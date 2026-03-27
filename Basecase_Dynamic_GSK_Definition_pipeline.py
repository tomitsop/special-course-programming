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

    Parameters
    ----------
    ptdf_z : np.ndarray
        Shape (n_lines, n_zones)

    Returns
    -------
    np.ndarray
        Shape (n_lines, n_pairs)
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


def build_post_contingency_zonal_ptdf_block(
    ptdf_z_base: np.ndarray,
    lodf: np.ndarray,
    contingency_idx: int,
) -> np.ndarray:
    """
    Efficient N-1 zonal PTDF update:

        PTDF_Z^(k) = PTDF_Z + LODF[:, k] outer PTDF_Z[k, :]

    Parameters
    ----------
    ptdf_z_base : np.ndarray
        Base zonal PTDF matrix of shape (|L|, |Z_FBMC|)

    lodf : np.ndarray
        LODF matrix of shape (|L|, |L|)

    contingency_idx : int
        Outaged line index k

    Returns
    -------
    np.ndarray
        Post-contingency zonal PTDF block of shape (|L|, |Z_FBMC|)
    """
    if contingency_idx < 0 or contingency_idx >= len(L):
        raise IndexError(f"contingency_idx {contingency_idx} out of range.")

    lodf_col = lodf[:, contingency_idx].reshape(-1, 1)       # (L, 1)
    outage_row = ptdf_z_base[contingency_idx, :].reshape(1, -1)  # (1, Z)

    return ptdf_z_base + (lodf_col @ outage_row)


def compute_post_contingency_line_flows(
    line_f_base: np.ndarray,
    lodf: np.ndarray,
    contingency_idx: int,
) -> np.ndarray:
    """
    Efficient N-1 line flow update:

        f^(k) = f + LODF[:, k] * f[k]

    Parameters
    ----------
    line_f_base : np.ndarray
        Base line flow vector of shape (|L|,)

    lodf : np.ndarray
        LODF matrix of shape (|L|, |L|)

    contingency_idx : int
        Outaged line index k

    Returns
    -------
    np.ndarray
        Post-contingency line flow vector of shape (|L|,)
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
# These are written to match input_data_base_functions.py logic
###############################################################

def build_flat_gsk():
    """
    Flat GSK:
    each FBMC node in a zone gets 1 / (#nodes_in_zone) for that zone,
    matching get_gsk_flat() in input_data_base_functions.py.
    """
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
    """
    Flat unit GSK:
    equal weight for conventional nodes in each zone,
    matching get_gsk_flat_unit() in input_data_base_functions.py.
    """
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
    """
    Pmax-based GSK over conventional generators P,
    matching get_gsk_pmax() in input_data_base_functions.py.
    """
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
    """
    Pmax-based GSK on a subset of generators
    (Hard Coal, Gas/CCGT in FB zones),
    matching get_gsk_pmax_sub() in input_data_base_functions.py.
    """
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
    """
    Dynamic headroom GSK:
    zone weights proportional to available headroom (Pmax - dispatched gen)
    over conventional plants in the FBMC zone.
    """
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
    """
    Dynamic generation GSK:
    zone weights proportional to actual D-2 generation over conventional plants
    in the FBMC zone.
    """
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

    Returns
    -------
    tuple
        (cnec, cnec_idx, ptdf_z, ptdf_z_cnec)
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
    gsk : array-like
        Shape (|N_FBMC|, |Z_FBMC|)

    lodf : array-like
        Shape (|L|, |L|)

    bad_k : array-like of bool, optional
        Length |L|. True means skip this contingency.

    cne_alpha : float
        CNE/CNEC threshold.

    include_cb_lines : bool
        If True, always include cross-border monitored lines in each block.

    include_basecase : bool
        If True, prepend base-case (N-0) selected rows with contingency_idx = -1.

    Returns
    -------
    dict
        Keys:
          - gsk
          - ptdf_z
          - ptdf_z_cnec
          - cnec
          - cnec_idx
          - contingency
          - contingency_idx
          - cnec_meta
          - n_rows
    """
    gsk_arr = _validate_gsk(gsk)
    lodf_arr = _validate_lodf(lodf)
    bad_mask = _build_bad_k_mask(bad_k)

    ptdf_z_base = PTDF_FBMC @ gsk_arr

    rows = []
    monitored_line_ids = []
    monitored_line_idx = []
    contingency_ids = []
    contingency_idx = []
    meta = []

    # Optional N-0 block first
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

    # N-1 blocks
    for k in range(len(L)):
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
        """
        Build only the GSK for the requested strategy.
        """
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
    ):
        """
        Build the GSK payload for one MTU.

        Parameters
        ----------
        strategy : str
            One of:
              - flat
              - flat_unit
              - pmax
              - pmax_sub
              - dynamic_headroom
              - dynamic_gen

        t : optional
            Time index when using a full DataFrame for dynamic GSKs.

        df_d2_gen : DataFrame or Series, optional
            D-2 generation data for dynamic GSK strategies.

        lodf : np.ndarray, optional
            Required when fbmc_mode == "n1".

        bad_k : array-like of bool, optional
            Contingencies to skip when fbmc_mode == "n1".

        fbmc_mode : str
            "basecase" or "n1"

        include_basecase : bool
            Only used when fbmc_mode == "n1".
            If True, prepend N-0 rows before N-1 rows.

        Returns
        -------
        dict
            Base-case mode keys:
              - gsk
              - cnec
              - cnec_idx
              - ptdf_z
              - ptdf_z_cnec

            N-1 mode keys:
              - gsk
              - cnec
              - cnec_idx
              - contingency
              - contingency_idx
              - cnec_meta
              - ptdf_z
              - ptdf_z_cnec
              - n_rows
        """
        mode = str(fbmc_mode).lower().strip()
        if mode not in {"basecase", "n1"}:
            raise ValueError(f"Unknown fbmc_mode: {fbmc_mode}")

        is_static = strategy in {"flat", "flat_unit", "pmax", "pmax_sub"}

        cache_key = None
        if is_static:
            bad_mask = _build_bad_k_mask(bad_k) if mode == "n1" else np.zeros(len(L), dtype=bool)
            cache_key = (
                strategy,
                mode,
                bool(include_basecase),
                bool(self.include_cb_lines),
                float(self.cne_alpha),
                bad_mask.tobytes(),
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
            )

        if cache_key is not None:
            self.static_cache[cache_key] = payload

        return payload