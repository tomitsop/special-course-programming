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


def _get_pmax_sum_at_node_for_gens(bus_id, gen_ids):
    return float(
        df_plants.loc[
            (df_plants["OnBus"] == bus_id) & (df_plants["GenID"].isin(gen_ids)),
            "Pmax"
        ].sum()
    )


def _get_pmax_sum_in_zone_for_gens(zone, gen_ids):
    return float(
        df_plants.loc[
            (df_plants["Zone"] == zone) & (df_plants["GenID"].isin(gen_ids)),
            "Pmax"
        ].sum()
    )


def _build_pairwise_z2z_matrix(ptdf_z: np.ndarray) -> np.ndarray:
    """
    Build all pairwise zonal PTDF differences:
        PTDF_Z[:, i] - PTDF_Z[:, j] for i < j

    Returns array of shape (n_lines, n_pairs).
    """
    n_z = len(Z_FBMC)
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
# CNEC computation
# This now matches input_data_base_functions.py logic
###############################################################

def compute_cnec_from_gsk(gsk, cne_alpha, include_cb_lines=True):
    """
    Compute:
      - PTDF_Z = PTDF_FBMC @ GSK
      - pairwise zonal PTDF differences (z-z)
      - CNEC set selected by max abs z-z >= cne_alpha
      - optionally add cross-border lines
      - preserve L ordering in final CNEC list
    """
    ptdf_z = PTDF_FBMC @ np.asarray(gsk, dtype=float)

    z2z = _build_pairwise_z2z_matrix(ptdf_z)

    if z2z.shape[1] == 0:
        maximum_abs_z2z = np.zeros(len(L), dtype=float)
    else:
        maximum_abs_z2z = np.max(np.abs(z2z), axis=1)

    cnec_set = {L[i] for i, x in enumerate(maximum_abs_z2z) if x >= float(cne_alpha)}

    if include_cb_lines:
        cb_lines = find_cross_border_lines()
        for line in cb_lines:
            cnec_set.add(line)

    cnec = [l for l in L if l in cnec_set]
    cnec_idx = [L_idx[l] for l in cnec]
    ptdf_z_cnec = ptdf_z[cnec_idx, :]

    return cnec, cnec_idx, ptdf_z, ptdf_z_cnec


###############################################################
# Strategy Manager
###############################################################

class GSKStrategyManager:
    def __init__(self, cne_alpha, include_cb_lines=True):
        self.cne_alpha = cne_alpha
        self.include_cb_lines = include_cb_lines
        self.static_cache = {}

    def build_for_t(self, strategy, t=None, df_d2_gen=None):
        """
        Static strategies are cached once.
        Dynamic strategies can be built either from:
          - a full DataFrame indexed by t, or
          - a Series already corresponding to one MTU.
        """
        if strategy in {"flat", "flat_unit", "pmax", "pmax_sub"}:
            if strategy not in self.static_cache:
                if strategy == "flat":
                    gsk = build_flat_gsk()
                elif strategy == "flat_unit":
                    gsk = build_flat_unit_gsk()
                elif strategy == "pmax":
                    gsk = build_pmax_gsk()
                elif strategy == "pmax_sub":
                    gsk = build_pmax_sub_gsk()
                else:
                    raise ValueError(f"Unknown static GSK strategy: {strategy}")

                cnec, cnec_idx, ptdf_z, ptdf_z_cnec = compute_cnec_from_gsk(
                    gsk=gsk,
                    cne_alpha=self.cne_alpha,
                    include_cb_lines=self.include_cb_lines,
                )

                self.static_cache[strategy] = {
                    "gsk": gsk,
                    "cnec": cnec,
                    "cnec_idx": cnec_idx,
                    "ptdf_z": ptdf_z,
                    "ptdf_z_cnec": ptdf_z_cnec,
                }

            return self.static_cache[strategy]

        if strategy in {"dynamic_headroom", "dynamic_gen"}:
            if df_d2_gen is None:
                raise ValueError("Dynamic GSK requires df_d2_gen")

            if isinstance(df_d2_gen, pd.Series):
                d2_gen_t = df_d2_gen
            else:
                if t is None:
                    raise ValueError("Dynamic GSK requires t when df_d2_gen is a DataFrame")
                d2_gen_t = df_d2_gen.loc[t]

            if strategy == "dynamic_headroom":
                gsk = build_dynamic_headroom_gsk(d2_gen_t)
            elif strategy == "dynamic_gen":
                gsk = build_dynamic_gen_gsk(d2_gen_t)
            else:
                raise ValueError(f"Unknown dynamic GSK strategy: {strategy}")

            cnec, cnec_idx, ptdf_z, ptdf_z_cnec = compute_cnec_from_gsk(
                gsk=gsk,
                cne_alpha=self.cne_alpha,
                include_cb_lines=self.include_cb_lines,
            )

            return {
                "gsk": gsk,
                "cnec": cnec,
                "cnec_idx": cnec_idx,
                "ptdf_z": ptdf_z,
                "ptdf_z_cnec": ptdf_z_cnec,
            }

        raise ValueError(f"Unknown GSK strategy {strategy}")