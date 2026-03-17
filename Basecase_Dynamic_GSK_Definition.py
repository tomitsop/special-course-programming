import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from input_data_base_functions import *  # gets all the pre calculated data


# #########################################
# ###    SUSCEPTANCE / PTDF MATRICES   ###
# #########################################

# MWBase = 380 ** 2
# slack_node = 50
# slack_position = N.index(slack_node)

# line_sus_mat = np.matmul(susceptance.values / MWBase, incidence.values)
# node_sus_mat = np.matmul(
#     np.matmul(incidence.values.T, susceptance.values / MWBase),
#     incidence.values
# )

# line_sus_mat_ = np.delete(line_sus_mat, slack_position, axis=1)
# node_sus_mat_ = np.delete(np.delete(node_sus_mat, slack_position, axis=0), slack_position, axis=1)

# PTDF_full = np.matmul(line_sus_mat_, np.linalg.inv(node_sus_mat_))
# zero_column = np.zeros((len(L), 1))
# PTDF_full = np.hstack((PTDF_full[:, :slack_position], zero_column, PTDF_full[:, slack_position:]))

# N_FBMC_indices = [N_idx[n] for n in N_FBMC]
# PTDF_FBMC = PTDF_full[:, N_FBMC_indices]


#################################
###    CREATE GSK MATRICES    ###
#################################

N_FBMC_idx = {n: i for i, n in enumerate(N_FBMC)}
Z_FBMC_idx = {z: i for i, z in enumerate(Z_FBMC)}


# -------- STATIC GSKs --------
def get_gsk_flat():
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]
        nodes_in_zone = df_bus[df_bus["Zone"] == zone_temp]["BusID"].tolist()
        gsk_temp[n_index, z_index] = 1.0 / len(nodes_in_zone)
    return gsk_temp


def get_gsk_flat_unit():
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]
        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()
        if n in conv_nodes_in_zone:
            gsk_temp[n_index, z_index] = 1.0 / len(conv_nodes_in_zone)
    return gsk_temp


def get_gsk_pmax():
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]
        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()
        if n in conv_nodes_in_zone:
            conv_pmax_in_zone = df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) & (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()
            conv_pmax_at_node = df_plants.loc[
                (df_plants["OnBus"] == n) & (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()
            if conv_pmax_in_zone > 0:
                gsk_temp[n_index, z_index] = conv_pmax_at_node / conv_pmax_in_zone
    return gsk_temp


def get_gsk_pmax_sub():
    P_sub = df_plants.loc[
        (df_plants["Type"].isin(["Hard Coal", "Gas/CCGT"])) &
        (df_plants["Zone"].isin(Z_FBMC)),
        "GenID"
    ].tolist()

    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    for n in N_FBMC:
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        n_index = N_FBMC_idx[n]
        z_index = Z_FBMC_idx[zone_temp]
        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P_sub)),
            "OnBus"
        ].unique()
        if n in conv_nodes_in_zone:
            conv_pmax_in_zone = df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) & (df_plants["GenID"].isin(P_sub)),
                "Pmax"
            ].sum()
            conv_pmax_at_node = df_plants.loc[
                (df_plants["OnBus"] == n) & (df_plants["GenID"].isin(P_sub)),
                "Pmax"
            ].sum()
            if conv_pmax_in_zone > 0:
                gsk_temp[n_index, z_index] = conv_pmax_at_node / conv_pmax_in_zone
    return gsk_temp


###########################################
###  CNEC selection + histogram helpers  ###
###########################################

def compute_cnec_list_from_gsk(
    PTDF_FBMC: np.ndarray,
    gsk: np.ndarray,
    *,
    cne_alpha: float,
    include_cb_lines: bool,
):
    """
    Computes CNEC list for a given GSK matrix.
    Returns (CNEC_list, CNEC_indices).
    """

    PTDF_Z = PTDF_FBMC @ gsk  # (|L|, |Z_FBMC|)

    n_z = len(Z_FBMC)
    n_pairs = n_z * (n_z - 1) // 2
    z2z = np.zeros((len(L), n_pairs), dtype=float)

    counter = 0
    for i_z in range(n_z - 1):
        for j_z in range(i_z + 1, n_z):
            z2z[:, counter] = PTDF_Z[:, i_z] - PTDF_Z[:, j_z]
            counter += 1

    maximum_abs_z2z = np.max(np.abs(z2z), axis=1)
    CNEC = [L[i] for i, x in enumerate(maximum_abs_z2z) if x >= cne_alpha]

    if include_cb_lines:
        cb_lines = find_cross_border_lines()
        CNEC = set(CNEC).union(cb_lines)
        CNEC = [l for l in L if l in CNEC]  # keep original L order
    else:
        CNEC = [l for l in L if l in CNEC]

    CNEC_indices = [L_idx[l] for l in CNEC]
    return CNEC, CNEC_indices


def plot_histogram(counts: np.ndarray, title: str, bins="auto"):
    """
    Plot histogram of integer counts (e.g., #CNECs per MTU).
    """
    counts = np.asarray(counts).reshape(-1)
    counts = counts[np.isfinite(counts)].astype(int)

    plt.figure()
    plt.hist(counts, bins=bins)
    plt.title(title)
    plt.xlabel("#CNECs")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()


def summarize_and_plot_histogram(counts: np.ndarray, title: str, top_k: int = 25, bins="auto"):
    """
    Prints summary + (optional) compact frequency table, and plots histogram.
    """
    counts = np.asarray(counts).reshape(-1)
    counts = counts[np.isfinite(counts)].astype(int)

    s = pd.Series(counts)
    vc = s.value_counts().sort_index()

    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)
    print(f"MTUs: {len(counts)}")
    print(f"min={counts.min()}, p5={np.percentile(counts,5):.0f}, "
          f"median={np.percentile(counts,50):.0f}, p95={np.percentile(counts,95):.0f}, "
          f"max={counts.max()}, mean={counts.mean():.2f}")

    print("\nHistogram (#CNECs -> frequency):")
    if len(vc) <= top_k:
        for k, v in vc.items():
            print(f"{int(k):5d} -> {int(v):6d}")
    else:
        most_freq = s.value_counts().head(10)
        print("(Too many distinct values; showing 10 most frequent and edges)\n")

        print("10 most frequent (#CNECs -> frequency):")
        for k, v in most_freq.items():
            print(f"{int(k):5d} -> {int(v):6d}")

        print("\nSmallest values:")
        for k, v in vc.head(5).items():
            print(f"{int(k):5d} -> {int(v):6d}")

        print("\nLargest values:")
        for k, v in vc.tail(5).items():
            print(f"{int(k):5d} -> {int(v):6d}")

    # Plot at the end
    plot_histogram(counts, title=title, bins=bins)


###########################################
###   Dynamic GSK  ###
###########################################

def get_gsk_headroom_single_t(df_d2_gen, t, p_ids, p_node_idx, p_zone_idx, p_pmax, node_zone_idx):
    
    gen_vals = df_d2_gen.loc[t, p_ids].to_numpy(dtype=float) # generation schedules from D-2 CGM
    headroom_vals = p_pmax - gen_vals  # PMAX - PG 

    node_zone_sum = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    np.add.at(node_zone_sum, (p_node_idx, p_zone_idx), headroom_vals)

    zone_tot = node_zone_sum.sum(axis=0)
    node_tot = node_zone_sum[np.arange(len(N_FBMC)), node_zone_idx]

    denom = zone_tot[node_zone_idx]
    weights = np.zeros(len(N_FBMC), dtype=float)
    mask = denom > 0
    weights[mask] = node_tot[mask] / denom[mask]

    gsk_t = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    gsk_t[np.arange(len(N_FBMC)), node_zone_idx] = weights
    return gsk_t


def get_gsk_currentgen_single_t(df_d2_gen, t, p_ids, p_node_idx, p_zone_idx, node_zone_idx):
    
    gen_vals = df_d2_gen.loc[t, p_ids].to_numpy(dtype=float) # generation schedules from D-2 CGM
    gen_vals = gen_vals # generation schedule 

    node_zone_sum = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    np.add.at(node_zone_sum, (p_node_idx, p_zone_idx), gen_vals)

    zone_tot = node_zone_sum.sum(axis=0)
    node_tot = node_zone_sum[np.arange(len(N_FBMC)), node_zone_idx]

    denom = zone_tot[node_zone_idx]
    weights = np.zeros(len(N_FBMC), dtype=float)
    mask = denom > 0
    weights[mask] = node_tot[mask] / denom[mask]

    gsk_t = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)
    gsk_t[np.arange(len(N_FBMC)), node_zone_idx] = weights
    return gsk_t


def dynamic_cnec_counts(
    d2_results_dir: str,
    *,
    cne_alpha: float,
    include_cb_lines: bool,
):
    """
    Reads d_2_gen.parquet from d2_results_dir and computes #CNECs per t for:
      - headroom weights (Pmax - D2_GEN)
      - gen-only weights (D2_GEN)
    Returns: (counts_headroom, counts_genonly)
    """

    d2_gen_path = os.path.join(d2_results_dir, "d_2_gen.parquet")
    if not os.path.exists(d2_gen_path):
        raise FileNotFoundError(
            f"Missing {d2_gen_path}\n"
            f"Run your D-2 SCOPF script first (the one that writes d_2_gen.parquet)."
        )

    df_d2_gen = pd.read_parquet(d2_gen_path).set_index("t")
    t_used = df_d2_gen.index.to_numpy(dtype=int)
    print(t_used)

    P_sub = df_plants.loc[
        (df_plants["Type"].isin(["Hard Coal", "Gas/CCGT"])) &
        (df_plants["Zone"].isin(Z_FBMC)),
        "GenID"
    ].tolist()

    p_ids = [str(p) for p in P_sub]
    p_node_idx = np.zeros(len(P_sub), dtype=int)
    p_zone_idx = np.zeros(len(P_sub), dtype=int)
    p_pmax = np.zeros(len(P_sub), dtype=float)

    for i, p in enumerate(P_sub):
        prow = df_plants.loc[df_plants["GenID"] == p].iloc[0]
        p_node_idx[i] = N_FBMC_idx[int(prow["OnBus"])]
        p_zone_idx[i] = Z_FBMC_idx[prow["Zone"]]
        p_pmax[i] = float(prow["Pmax"])

    node_zone_idx = np.zeros(len(N_FBMC), dtype=int)
    for i, n in enumerate(N_FBMC):
        z = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]
        node_zone_idx[i] = Z_FBMC_idx[z]

    counts_headroom = np.zeros(len(t_used), dtype=int)
    counts_genonly = np.zeros(len(t_used), dtype=int)

    for it, t in enumerate(t_used):
        if t not in df_d2_gen.index:
            raise ValueError(f"t={t} missing from {d2_gen_path}")

        gsk_headroom_t = get_gsk_headroom_single_t(
            df_d2_gen, t, p_ids, p_node_idx, p_zone_idx, p_pmax, node_zone_idx
        )
        CNEC_h, _ = compute_cnec_list_from_gsk(
            PTDF_FBMC, gsk_headroom_t,
            cne_alpha=cne_alpha,
            include_cb_lines=include_cb_lines
        )
        counts_headroom[it] = len(CNEC_h)

        gsk_genonly_t = get_gsk_currentgen_single_t(
            df_d2_gen, t, p_ids, p_node_idx, p_zone_idx, node_zone_idx
        )
        CNEC_g, _ = compute_cnec_list_from_gsk(
            PTDF_FBMC, gsk_genonly_t,
            cne_alpha=cne_alpha,
            include_cb_lines=include_cb_lines
        )
        counts_genonly[it] = len(CNEC_g)

        if (it + 1) % 500 == 0:
            print(f"Computed dynamic CNEC counts for {it+1}/{len(t_used)} MTUs...")

    return counts_headroom, counts_genonly, t_used


#################################
###            MAIN           ###
#################################

def main():
    # # ---- your knobs ----
    # cne_alpha = 0.05
    # include_cb_lines = True

    # IMPORTANT: set this to the directory created by your SCOPF script
    # In your SCOPF script you used: Basecase_D_2_CGM_Results
    d2_results_dir = os.path.join(os.getcwd(), "BaseCase_D_2_CGM_Results")

    # ---- build static GSKs ----
    gsk_flat = get_gsk_flat()
    gsk_flat_unit = get_gsk_flat_unit()
    gsk_pmax = get_gsk_pmax()
    gsk_pmax_sub = get_gsk_pmax_sub()

    # ---- column sum checks ----
    print("\nAll STATIC GSKs built, with column sums:")
    print("1) GSK flat:",      np.round(np.sum(gsk_flat, axis=0), 2))
    print("2) GSK flat unit:", np.round(np.sum(gsk_flat_unit, axis=0), 2))
    print("3) GSK pmax:",      np.round(np.sum(gsk_pmax, axis=0), 2))
    print("4) GSK pmax sub:",  np.round(np.sum(gsk_pmax_sub, axis=0), 2))

    # ---- static CNEC counts ----
    static_strategies = {
        "flat": gsk_flat,
        "flat_unit": gsk_flat_unit,
        "pmax": gsk_pmax,
        "pmax_sub": gsk_pmax_sub,
    }

    print("\n" + "#" * 80)
    print("STATIC GSK STRATEGIES: CNEC COUNTS")
    print("#" * 80)

    for name, gsk in static_strategies.items():
        CNEC, _ = compute_cnec_list_from_gsk(
            PTDF_FBMC, gsk,
            cne_alpha=cne_alpha,
            include_cb_lines=include_cb_lines
        )
        print(f"{name:20s} -> #CNECs = {len(CNEC)}")

    # ---- dynamic CNEC histograms ----
    print("\n" + "#" * 80)
    print("DYNAMIC GSK STRATEGIES: CNEC COUNTS PER MTU (HISTOGRAM)")
    print("#" * 80)

    counts_headroom, counts_genonly, t_used = dynamic_cnec_counts(
        d2_results_dir,
        cne_alpha=cne_alpha,
        include_cb_lines=include_cb_lines
    )

    summarize_and_plot_histogram(counts_headroom, "DYNAMIC GSK: headroom (Pmax - D2 GEN)")
    summarize_and_plot_histogram(counts_genonly, "DYNAMIC GSK: gen-only (D2 GEN)")

    # ---- save per-MTU counts ----
    out_df = pd.DataFrame({
        "t": np.array(t_used, dtype=int),
        "cnec_count_headroom": counts_headroom,
        "cnec_count_genonly": counts_genonly,
    })
    out_path = os.path.join(d2_results_dir, "cnec_counts_dynamic.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved dynamic CNEC count time-series to: {out_path}")

    # optional: also save CSV for quick inspection
    out_csv = os.path.join(d2_results_dir, "cnec_counts_dynamic.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved CSV copy to: {out_csv}")


if __name__ == "__main__":
    main()
