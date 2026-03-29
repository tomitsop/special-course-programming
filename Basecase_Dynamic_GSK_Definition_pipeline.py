import numpy as np
import pandas as pd

from input_data_base_functions import *


###############################################################
# Utility
###############################################################

def normalize_columns(mat: np.ndarray) -> np.ndarray:
    """
    Normalize each column to sum to 1 when the column has positive mass.
    Keep zero columns unchanged.
    """
    out = np.array(mat, dtype=float, copy=True)
    col_sums = out.sum(axis=0)
    nonzero = col_sums > 0.0
    out[:, nonzero] = out[:, nonzero] / col_sums[nonzero]
    return out


def _get_zone_of_bus(bus_id):
    return df_bus.loc[df_bus["BusID"] == bus_id, "Zone"].iloc[0]


def _get_bus_of_gen(gen_id):
    return df_plants.loc[df_plants["GenID"] == gen_id, "OnBus"].iloc[0]


def _get_fbmc_plants_in_zone(zone: str):
    """
    Plants from P that belong to this FBMC zone.
    """
    return df_plants.loc[
        (df_plants["Zone"] == zone) & (df_plants["GenID"].isin(P)),
        "GenID"
    ].tolist()


def _build_pairwise_z2z_matrix(ptdf_z: np.ndarray) -> np.ndarray:
    """
    Build all pairwise zonal PTDF differences:
        PTDF_Z[:, i] - PTDF_Z[:, j] for i < j
    """
    n_z = ptdf_z.shape[1]
    n_pairs = int(n_z * (n_z - 1) / 2)

    if n_pairs == 0:
        return np.zeros((ptdf_z.shape[0], 0), dtype=float)

    z2z = np.zeros((ptdf_z.shape[0], n_pairs), dtype=float)

    counter = 0
    for i_z in range(n_z - 1):
        for j_z in range(i_z + 1, n_z):
            z2z[:, counter] = ptdf_z[:, i_z] - ptdf_z[:, j_z]
            counter += 1

    return z2z


def _validate_gsk(gsk: np.ndarray) -> np.ndarray:
    """
    Validate and cast GSK to expected shape (|N_FBMC|, |Z_FBMC|).
    """
    gsk_arr = np.asarray(gsk, dtype=float)

    expected_shape = (len(N_FBMC), len(Z_FBMC))
    if gsk_arr.shape != expected_shape:
        raise ValueError(
            f"GSK shape mismatch. Expected {expected_shape}, got {gsk_arr.shape}."
        )

    return gsk_arr


def _validate_lodf(lodf: np.ndarray) -> np.ndarray:
    """
    Validate and cast LODF to expected shape (|L|, |L|).
    """
    lodf_arr = np.asarray(lodf, dtype=float)

    expected_shape = (len(L), len(L))
    if lodf_arr.shape != expected_shape:
        raise ValueError(
            f"LODF shape mismatch. Expected {expected_shape}, got {lodf_arr.shape}."
        )

    return lodf_arr


def _build_bad_k_mask(bad_k=None) -> np.ndarray:
    """
    Convert bad_k to a boolean mask of length |L|.

    bad_k[i] == True means contingency i should be skipped.
    """
    if bad_k is None:
        return np.zeros(len(L), dtype=bool)

    bad_arr = np.asarray(bad_k, dtype=bool).reshape(-1)

    if bad_arr.shape[0] != len(L):
        raise ValueError(
            f"bad_k length mismatch. Expected {len(L)}, got {bad_arr.shape[0]}."
        )

    return bad_arr


def _line_ids_from_indices(indices):
    return [L[int(i)] for i in indices]


def _compute_max_abs_z2z(ptdf_z: np.ndarray) -> np.ndarray:
    """
    For a zonal PTDF matrix (n_lines, n_zones), compute for each line the
    maximum absolute zonal difference over all zone pairs.
    """
    z2z = _build_pairwise_z2z_matrix(ptdf_z)

    if z2z.shape[1] == 0:
        return np.zeros(ptdf_z.shape[0], dtype=float)

    return np.max(np.abs(z2z), axis=1)


def _select_monitored_line_indices_from_ptdf_z(
    ptdf_z: np.ndarray,
    cne_alpha: float,
    include_cb_lines: bool = True,
) -> list:
    """
    Base-case style line selection:
      - select lines with max abs zonal-pair PTDF diff >= cne_alpha
      - optionally always include cross-border lines
      - preserve original L ordering
    """
    maximum_abs_z2z = _compute_max_abs_z2z(ptdf_z)

    selected_set = {
        i for i, x in enumerate(maximum_abs_z2z)
        if x >= float(cne_alpha)
    }

    if include_cb_lines:
        cb_lines = find_cross_border_lines()
        for line in cb_lines:
            selected_set.add(L_idx[line])

    selected_idx = [i for i in range(len(L)) if i in selected_set]
    return selected_idx


def _normalize_selected_contingencies(selected_contingencies=None) -> np.ndarray | None:
    """
    Normalize selected contingencies to a unique integer array or None.

    Notes
    -----
    - Removes -1 if present, since -1 corresponds to the optional base-case block.
    - Preserves sorted unique indices.
    """
    if selected_contingencies is None:
        return None

    selected = np.asarray(selected_contingencies, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        return np.zeros(0, dtype=np.int64)

    if np.any((selected < -1) | (selected >= len(L))):
        bad_vals = selected[(selected < -1) | (selected >= len(L))]
        raise ValueError(
            f"selected_contingencies contains out-of-range values: {bad_vals.tolist()}"
        )

    selected = selected[selected != -1]
    return np.unique(selected)


def _build_contingency_iterator(selected_contingencies=None, bad_k=None) -> list:
    """
    Return the valid contingency indices to build in the N-1 block.

    If selected_contingencies is None, iterate over all non-bad contingencies.
    Otherwise, iterate only over the selected non-bad contingencies.
    """
    bad_mask = _build_bad_k_mask(bad_k)
    selected = _normalize_selected_contingencies(selected_contingencies)

    if selected is None:
        return [k for k in range(len(L)) if not bad_mask[k]]

    return [int(k) for k in selected if not bad_mask[int(k)]]


def build_post_contingency_zonal_ptdf_block(
    ptdf_z_base: np.ndarray,
    lodf: np.ndarray,
    contingency_idx: int,
) -> np.ndarray:
    """
    Efficient N-1 zonal PTDF update:

        PTDF_Z^(k) = PTDF_Z + LODF[:, k] outer PTDF_Z[k, :]
    """
    if contingency_idx < 0 or contingency_idx >= len(L):
        raise IndexError(f"contingency_idx {contingency_idx} out of range.")

    lodf_col = lodf[:, contingency_idx].reshape(-1, 1)
    outage_row = ptdf_z_base[contingency_idx, :].reshape(1, -1)

    return ptdf_z_base + (lodf_col @ outage_row)


def compute_post_contingency_line_flows(
    line_f_base: np.ndarray,
    lodf: np.ndarray,
    contingency_idx: int,
) -> np.ndarray:
    """
    Efficient N-1 line flow update:

        f^(k) = f + LODF[:, k] * f[k]
    """
    f = np.asarray(line_f_base, dtype=float).reshape(-1)
    lodf_arr = _validate_lodf(lodf)

    if f.shape[0] != len(L):
        raise ValueError(
            f"line_f_base length mismatch. Expected {len(L)}, got {f.shape[0]}."
        )

    if contingency_idx < 0 or contingency_idx >= len(L):
        raise IndexError(f"contingency_idx {contingency_idx} out of range.")

    return f + lodf_arr[:, contingency_idx] * f[contingency_idx]


def compute_selected_post_contingency_line_flows(
    line_f_base: np.ndarray,
    lodf: np.ndarray,
    monitored_idx: list,
    contingency_idx: int,
) -> np.ndarray:
    """
    Return only the monitored-line post-contingency flows for one outage.
    """
    f_k = compute_post_contingency_line_flows(
        line_f_base=line_f_base,
        lodf=lodf,
        contingency_idx=contingency_idx,
    )
    return f_k[np.asarray(monitored_idx, dtype=int)]


###############################################################
# Static GSK builders
###############################################################

def build_flat_gsk():
    gsk = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone = _get_zone_of_bus(n)

        if zone not in Z_FBMC:
            continue

        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone]

        nodes_in_zone = df_bus.loc[df_bus["Zone"] == zone, "BusID"].tolist()
        if len(nodes_in_zone) == 0:
            continue

        gsk[n_index, z_index] = 1.0 / len(nodes_in_zone)

    return gsk


def build_flat_unit_gsk():
    gsk = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone = _get_zone_of_bus(n)

        if zone not in Z_FBMC:
            continue

        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone]

        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()

        if n in conv_nodes_in_zone and len(conv_nodes_in_zone) > 0:
            gsk[n_index, z_index] = 1.0 / len(conv_nodes_in_zone)

    return gsk


def build_pmax_gsk():
    gsk = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone = _get_zone_of_bus(n)

        if zone not in Z_FBMC:
            continue

        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone]

        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()

        if n not in conv_nodes_in_zone:
            continue

        conv_pmax_in_zone = float(
            df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) &
                (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()
        )

        conv_pmax_at_node = float(
            df_plants.loc[
                (df_plants["OnBus"] == n) &
                (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()
        )

        if conv_pmax_in_zone > 0:
            gsk[n_index, z_index] = conv_pmax_at_node / conv_pmax_in_zone

    return gsk


def build_pmax_sub_gsk():
    p_sub = df_plants.loc[
        (df_plants["Type"].isin(["Hard Coal", "Gas/CCGT"])) &
        (df_plants["Zone"].isin(Z_FBMC)),
        "GenID"
    ].tolist()

    gsk = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for n in N_FBMC:
        zone = _get_zone_of_bus(n)

        if zone not in Z_FBMC:
            continue

        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone]

        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone) &
            (df_plants["GenID"].isin(p_sub)),
            "OnBus"
        ].unique()

        if n not in conv_nodes_in_zone:
            continue

        conv_pmax_in_zone = float(
            df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) &
                (df_plants["GenID"].isin(p_sub)),
                "Pmax"
            ].sum()
        )

        conv_pmax_at_node = float(
            df_plants.loc[
                (df_plants["OnBus"] == n) &
                (df_plants["GenID"].isin(p_sub)),
                "Pmax"
            ].sum()
        )

        if conv_pmax_in_zone > 0:
            gsk[n_index, z_index] = conv_pmax_at_node / conv_pmax_in_zone

    return gsk


###############################################################
# Dynamic GSK builders
###############################################################

def build_dynamic_headroom_gsk(d2_gen_t: pd.Series):
    gsk = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for z in Z_FBMC:
        plants = _get_fbmc_plants_in_zone(z)

        if len(plants) == 0:
            continue

        weights = []
        buses = []

        for p in plants:
            node = _get_bus_of_gen(p)
            buses.append(node)

            gen_val = float(d2_gen_t.get(p, 0.0))
            headroom = max(float(get_gen_up(p)) - gen_val, 0.0)
            weights.append(headroom)

        total = float(sum(weights))
        if total <= 0.0:
            continue

        zi = Z_FBMC_idx[z]

        for p, node, w in zip(plants, buses, weights):
            if node not in N_FBMC or w <= 0.0:
                continue

            ni = N_FBMC_idx[node]
            gsk[ni, zi] += w / total

    return normalize_columns(gsk)


def build_dynamic_gen_gsk(d2_gen_t: pd.Series):
    gsk = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    for z in Z_FBMC:
        plants = _get_fbmc_plants_in_zone(z)

        if len(plants) == 0:
            continue

        weights = []
        buses = []

        for p in plants:
            node = _get_bus_of_gen(p)
            buses.append(node)

            gen_val = max(float(d2_gen_t.get(p, 0.0)), 0.0)
            weights.append(gen_val)

        total = float(sum(weights))
        if total <= 0.0:
            continue

        zi = Z_FBMC_idx[z]

        for p, node, w in zip(plants, buses, weights):
            if node not in N_FBMC or w <= 0.0:
                continue

            ni = N_FBMC_idx[node]
            gsk[ni, zi] += w / total

    return normalize_columns(gsk)


###############################################################
# Base-case CNEC computation
###############################################################

def compute_cnec_from_gsk(gsk, cne_alpha, include_cb_lines=True):
    """
    Base-case (N-0) computation:
      - PTDF_Z = PTDF_FBMC @ GSK
      - pairwise zonal PTDF differences (z-z)
      - CNEC set selected by max abs z-z >= cne_alpha
      - optionally add cross-border lines
      - preserve L ordering in final CNEC list
    """
    gsk_arr = _validate_gsk(gsk)
    ptdf_z = PTDF_FBMC @ gsk_arr

    cnec_idx = _select_monitored_line_indices_from_ptdf_z(
        ptdf_z=ptdf_z,
        cne_alpha=cne_alpha,
        include_cb_lines=include_cb_lines,
    )

    cnec = _line_ids_from_indices(cnec_idx)
    ptdf_z_cnec = ptdf_z[np.asarray(cnec_idx, dtype=int), :]

    return cnec, cnec_idx, ptdf_z, ptdf_z_cnec


###############################################################
# N-1 / contingency-aware CNEC computation
###############################################################

def compute_cnec_from_gsk_n1(
    gsk,
    lodf,
    bad_k=None,
    cne_alpha=0.05,
    include_cb_lines=True,
    include_basecase=True,
    selected_contingencies=None,
):
    """
    Contingency-aware (N-1) computation using direct zonal PTDF updates.

    For each valid contingency k:
        PTDF_Z^(k) = PTDF_Z + LODF[:, k] outer PTDF_Z[k, :]

    Then, for each contingency block:
      - compute pairwise zonal PTDF differences (z-z)
      - select monitored lines by max abs z-z >= cne_alpha
      - optionally add cross-border lines
      - store each selected (monitored line, contingency) pair

    Parameters
    ----------
    selected_contingencies : array-like of int, optional
        If provided, only these contingency indices are built in the N-1 block.
        The optional base-case block is still controlled separately by
        include_basecase.
    """
    gsk_arr = _validate_gsk(gsk)
    lodf_arr = _validate_lodf(lodf)
    bad_mask = _build_bad_k_mask(bad_k)
    contingency_iter = _build_contingency_iterator(
        selected_contingencies=selected_contingencies,
        bad_k=bad_mask,
    )

    ptdf_z_base = PTDF_FBMC @ gsk_arr

    rows = []
    monitored_line_ids = []
    monitored_line_idx = []
    contingency_ids = []
    contingency_idx = []
    meta = []

    if include_basecase:
        selected_idx_base = _select_monitored_line_indices_from_ptdf_z(
            ptdf_z=ptdf_z_base,
            cne_alpha=cne_alpha,
            include_cb_lines=include_cb_lines,
        )

        for ell in selected_idx_base:
            rows.append(ptdf_z_base[ell, :].copy())
            monitored_line_ids.append(L[ell])
            monitored_line_idx.append(ell)
            contingency_ids.append("basecase")
            contingency_idx.append(-1)
            meta.append(
                {
                    "cnec_id": L[ell],
                    "cnec_idx": ell,
                    "contingency_id": "basecase",
                    "contingency_idx": -1,
                }
            )

    for k in contingency_iter:
        if bad_mask[k]:
            continue

        ptdf_z_k = build_post_contingency_zonal_ptdf_block(
            ptdf_z_base=ptdf_z_base,
            lodf=lodf_arr,
            contingency_idx=k,
        )

        selected_idx_k = _select_monitored_line_indices_from_ptdf_z(
            ptdf_z=ptdf_z_k,
            cne_alpha=cne_alpha,
            include_cb_lines=include_cb_lines,
        )

        for ell in selected_idx_k:
            rows.append(ptdf_z_k[ell, :].copy())
            monitored_line_ids.append(L[ell])
            monitored_line_idx.append(ell)
            contingency_ids.append(L[k])
            contingency_idx.append(k)
            meta.append(
                {
                    "cnec_id": L[ell],
                    "cnec_idx": ell,
                    "contingency_id": L[k],
                    "contingency_idx": k,
                }
            )

    if len(rows) > 0:
        ptdf_z_cnec = np.vstack(rows).astype(float)
    else:
        ptdf_z_cnec = np.zeros((0, len(Z_FBMC)), dtype=float)

    return {
        "gsk": gsk_arr,
        "ptdf_z": ptdf_z_base,
        "ptdf_z_cnec": ptdf_z_cnec,
        "cnec": monitored_line_ids,
        "cnec_idx": monitored_line_idx,
        "contingency": contingency_ids,
        "contingency_idx": contingency_idx,
        "cnec_meta": meta,
        "n_rows": int(ptdf_z_cnec.shape[0]),
    }


###############################################################
# Strategy Manager
###############################################################

class GSKStrategyManager:
    def __init__(self, cne_alpha, include_cb_lines=True):
        self.cne_alpha = float(cne_alpha)
        self.include_cb_lines = bool(include_cb_lines)
        self.static_cache = {}

    def _build_gsk_only(self, strategy, t=None, df_d2_gen=None):
        if strategy == "flat":
            return build_flat_gsk()

        if strategy == "flat_unit":
            return build_flat_unit_gsk()

        if strategy == "pmax":
            return build_pmax_gsk()

        if strategy == "pmax_sub":
            return build_pmax_sub_gsk()

        if strategy in {"dynamic_headroom", "dynamic_gen"}:
            if df_d2_gen is None:
                raise ValueError("Dynamic GSK requires df_d2_gen.")

            if isinstance(df_d2_gen, pd.Series):
                d2_gen_t = df_d2_gen
            else:
                if t is None:
                    raise ValueError(
                        "Dynamic GSK requires t when df_d2_gen is a DataFrame."
                    )
                d2_gen_t = df_d2_gen.loc[t]

            if strategy == "dynamic_headroom":
                return build_dynamic_headroom_gsk(d2_gen_t)

            if strategy == "dynamic_gen":
                return build_dynamic_gen_gsk(d2_gen_t)

        raise ValueError(f"Unknown GSK strategy: {strategy}")

    def build_for_t(
        self,
        strategy,
        t=None,
        df_d2_gen=None,
        lodf=None,
        bad_k=None,
        fbmc_mode="basecase",
        include_basecase=True,
        selected_contingencies=None,
    ):
        """
        Build the GSK payload for one MTU.

        If selected_contingencies is provided in N-1 mode, only the base-case
        block and those contingency blocks are constructed.
        """
        mode = str(fbmc_mode).lower().strip()
        if mode not in {"basecase", "n1"}:
            raise ValueError(f"Unknown fbmc_mode: {fbmc_mode}")

        is_static = strategy in {"flat", "flat_unit", "pmax", "pmax_sub"}

        selected_contingencies_arr = _normalize_selected_contingencies(selected_contingencies)

        cache_key = None
        if is_static:
            bad_mask = _build_bad_k_mask(bad_k) if mode == "n1" else np.zeros(len(L), dtype=bool)
            selected_key = None if selected_contingencies_arr is None else tuple(selected_contingencies_arr.tolist())
            cache_key = (
                strategy,
                mode,
                bool(include_basecase),
                bool(self.include_cb_lines),
                float(self.cne_alpha),
                bad_mask.tobytes(),
                selected_key,
            )
            if cache_key in self.static_cache:
                return self.static_cache[cache_key]

        gsk = self._build_gsk_only(strategy=strategy, t=t, df_d2_gen=df_d2_gen)

        if mode == "basecase":
            cnec, cnec_idx, ptdf_z, ptdf_z_cnec = compute_cnec_from_gsk(
                gsk=gsk,
                cne_alpha=self.cne_alpha,
                include_cb_lines=self.include_cb_lines,
            )

            payload = {
                "gsk": gsk,
                "cnec": cnec,
                "cnec_idx": cnec_idx,
                "ptdf_z": ptdf_z,
                "ptdf_z_cnec": ptdf_z_cnec,
            }
        else:
            if lodf is None:
                raise ValueError("fbmc_mode='n1' requires lodf.")

            payload = compute_cnec_from_gsk_n1(
                gsk=gsk,
                lodf=lodf,
                bad_k=bad_k,
                cne_alpha=self.cne_alpha,
                include_cb_lines=self.include_cb_lines,
                include_basecase=include_basecase,
                selected_contingencies=selected_contingencies_arr,
            )

        if cache_key is not None:
            self.static_cache[cache_key] = payload

        return payload
