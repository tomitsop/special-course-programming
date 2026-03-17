# Basecase_FBMC_Bins_FBME.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from input_data_base_functions import (
    L,
    Z_FBMC,
    n_in_z,
    get_dem,
    get_renew,
)

###############################################################################
# CONFIG
###############################################################################

RESULTS_BASE = Path("results")

RUNS = {
    "dynamic_gen": RESULTS_BASE / "pipeline_run_gurobi_dynamic_gen",
    # "SCOPF_dynamic_headroom": RESULTS_BASE / "results/pipeline_run_gurobi_SCOPF_dynamic_headroom",
    # # "flat_unit": RESULTS_BASE / "pipeline_run_gurobi_flat_unit",
    # "SCOPF_pmax_sub": RESULTS_BASE / "results/pipeline_run_gurobi_SCOPF_pmax_sub",
    # "SCOPF_dynamic_gen": RESULTS_BASE / "results/pipeline_run_gurobi_SCOPF_dynamic_gen",
}

OUTPUT_DIR = RESULTS_BASE / "fbmc_bins_fbme_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONGESTION_THRESHOLD = 0.85
BIN_LABELS = ["low", "mid", "high"]
PLOT_DPI = 200

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

###############################################################################
# IO HELPERS
###############################################################################

def read_df(run_dir: Path, rel_path: str) -> pd.DataFrame:
    path = run_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def ensure_numeric_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out.index = pd.Index(pd.to_numeric(out.index), name=out.index.name)
    except Exception:
        pass
    return out.sort_index()


def read_line_cap_margin(run_dir: Path) -> pd.Series:
    path = run_dir / "fb" / "line_cap_margin.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_parquet(path)
    if "line_cap_margin" not in df.columns:
        raise ValueError(f"'line_cap_margin' column not found in {path}")

    s = df["line_cap_margin"].copy()
    s.name = "line_cap_margin"
    return s


def align_line_cap_margin(line_cap_margin: pd.Series, flow_columns: pd.Index) -> pd.Series:
    aligned = line_cap_margin.reindex(flow_columns)

    if aligned.isna().any():
        tmp = line_cap_margin.copy()
        tmp.index = tmp.index.map(str)
        aligned = tmp.reindex(flow_columns.map(str))
        aligned.index = flow_columns

    missing = aligned[aligned.isna()].index.tolist()
    if missing:
        raise ValueError(
            "Missing line_cap_margin values for some flow columns. "
            f"Example missing lines: {missing[:10]}"
        )

    aligned.name = "line_cap_margin"
    return aligned


###############################################################################
# BIN HELPERS
###############################################################################

def qcut_3labels(series: pd.Series, labels=BIN_LABELS) -> pd.Series:
    """
    Robust tertile binning.
    """
    s = series.astype(float)
    non_na = s.dropna()

    if non_na.empty:
        return pd.Series(index=s.index, dtype="object", name=f"{s.name}_bin")

    try:
        out = pd.qcut(non_na, q=3, labels=labels, duplicates="drop")
        if len(pd.Index(out.cat.categories)) < len(labels):
            raise ValueError("qcut collapsed categories")
    except Exception:
        ranked = non_na.rank(method="first")
        out = pd.qcut(ranked, q=3, labels=labels)

    result = pd.Series(index=s.index, dtype="object")
    result.loc[out.index] = out.astype(str)
    result.name = f"{s.name}_bin"
    return result


###############################################################################
# SYSTEM STATE FEATURES (FBMC ONLY)
###############################################################################

def compute_fbmc_state_features(time_index) -> pd.DataFrame:
    fbmc_nodes = []
    for z in Z_FBMC:
        fbmc_nodes.extend(n_in_z[z])
    fbmc_nodes = list(dict.fromkeys(fbmc_nodes))

    rows = []
    for t in time_index:
        t_int = int(t)

        total_demand_fbmc = float(sum(get_dem(t_int, n) for n in fbmc_nodes))
        total_renewables_fbmc = float(sum(get_renew(t_int, n) for n in fbmc_nodes))
        renewable_share_fbmc = (
            total_renewables_fbmc / total_demand_fbmc
            if total_demand_fbmc != 0
            else np.nan
        )

        rows.append({
            "t": t_int,
            "total_demand_fbmc": total_demand_fbmc,
            "total_renewables_fbmc": total_renewables_fbmc,
            "renewable_share_fbmc": renewable_share_fbmc,
        })

    return pd.DataFrame(rows).set_index("t").sort_index()


###############################################################################
# CONGESTION FEATURES FROM D-1 CGM
###############################################################################

def compute_congestion_features(
    line_f_d1: pd.DataFrame,
    line_cap_margin: pd.Series,
    threshold: float = CONGESTION_THRESHOLD,
) -> pd.DataFrame:
    margin = align_line_cap_margin(line_cap_margin, line_f_d1.columns)
    loading = line_f_d1.abs().div(margin, axis=1)

    out = pd.DataFrame(index=line_f_d1.index)
    out["mean_loading_d1"] = loading.mean(axis=1)
    out["high_loading_share_d1"] = (loading > threshold).mean(axis=1)
    out["n_high_loading_lines_d1"] = (loading > threshold).sum(axis=1).astype(int)
    out["n_lines_total_d1"] = int(loading.shape[1])

    return out


###############################################################################
# FBME FROM SAVED PTDF RESULTS
###############################################################################

def prepare_ptdf_long(ptdf_long: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"t", "cnec_idx", "zone", "ptdf"}
    missing = required_cols - set(ptdf_long.columns)
    if missing:
        raise ValueError(f"ptdf_long is missing required columns: {sorted(missing)}")

    out = ptdf_long.copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce")
    out["cnec_idx"] = pd.to_numeric(out["cnec_idx"], errors="coerce")
    out["ptdf"] = pd.to_numeric(out["ptdf"], errors="coerce")

    out = out.dropna(subset=["t", "cnec_idx", "zone", "ptdf"]).copy()
    out["t"] = out["t"].astype(int)
    out["cnec_idx"] = out["cnec_idx"].astype(int)

    return out


def compute_fbme_hourly_from_saved_ptdf(
    line_f_d1: pd.DataFrame,
    line_f_d2: pd.DataFrame,
    np_d1: pd.DataFrame,
    np_d2: pd.DataFrame,
    ptdf_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uses the PTDF values already saved by the pipeline in:
        fb/ptdf_z_cnec_long.parquet

    FBME_{t,j} = F^{D1-CGM}_{t,j} - F^{D2-CGM}_{t,j}
                 - sum_z (NP^{D1-CGM}_{t,z} - NP^{D2-CGM}_{t,z}) * PTDF^Z_{t,j,z}

    Matching logic is the same idea as in Basecase_Results_Comparison.py:
        cnec_idx -> L[cnec_idx] -> line_f columns
    """
    line_f_d1 = ensure_numeric_time_index(line_f_d1)
    line_f_d2 = ensure_numeric_time_index(line_f_d2)
    np_d1 = ensure_numeric_time_index(np_d1)
    np_d2 = ensure_numeric_time_index(np_d2)
    ptdf_long = prepare_ptdf_long(ptdf_long)

    common_t = (
        line_f_d1.index
        .intersection(line_f_d2.index)
        .intersection(np_d1.index)
        .intersection(np_d2.index)
        .intersection(pd.Index(ptdf_long["t"].unique()))
        .sort_values()
    )

    idx_to_line = {int(i): L[i] for i in range(len(L))}
    hourly_rows = []

    for t in common_t:
        ptdf_hour = ptdf_long.loc[ptdf_long["t"] == t].copy()

        row = {
            "t": int(t),
            "n_ptdf_rows_raw": int(len(ptdf_hour)),
            "n_matched_lines": 0,
            "fbme_status": "ok",
            "fbme_reason": "",
            "mean_abs_fbme_d1_cgm": np.nan,
            "max_abs_fbme_d1_cgm": np.nan,
            "p95_abs_fbme_d1_cgm": np.nan,
        }

        if ptdf_hour.empty:
            row["fbme_status"] = "nan"
            row["fbme_reason"] = "no_ptdf_rows"
            hourly_rows.append(row)
            continue

        ptdf_t = ptdf_hour.pivot_table(
            index="cnec_idx",
            columns="zone",
            values="ptdf",
            aggfunc="first"
        )

        ptdf_t = ptdf_t.reindex(columns=Z_FBMC)
        ptdf_t.index = ptdf_t.index.map(idx_to_line)
        ptdf_t = ptdf_t[ptdf_t.index.notna()]
        ptdf_t = ptdf_t[~ptdf_t.index.duplicated(keep="first")]

        matched_lines = [
            c for c in ptdf_t.index
            if c in line_f_d1.columns and c in line_f_d2.columns
        ]
        row["n_matched_lines"] = int(len(matched_lines))

        if len(matched_lines) == 0:
            row["fbme_status"] = "nan"
            row["fbme_reason"] = "no_matched_lines"
            hourly_rows.append(row)
            continue

        ptdf_t = ptdf_t.loc[matched_lines]

        if ptdf_t.isna().any().any():
            row["fbme_status"] = "nan"
            row["fbme_reason"] = "ptdf_contains_nan"
            hourly_rows.append(row)
            continue

        missing_np_d1_cols = [z for z in Z_FBMC if z not in np_d1.columns]
        missing_np_d2_cols = [z for z in Z_FBMC if z not in np_d2.columns]
        if missing_np_d1_cols or missing_np_d2_cols:
            missing_msg = []
            if missing_np_d1_cols:
                missing_msg.append(f"np_d1 missing zones: {missing_np_d1_cols}")
            if missing_np_d2_cols:
                missing_msg.append(f"np_d2 missing zones: {missing_np_d2_cols}")
            raise ValueError(" ; ".join(missing_msg))

        delta_np = (
            np_d1.loc[t, Z_FBMC].to_numpy(dtype=float)
            - np_d2.loc[t, Z_FBMC].to_numpy(dtype=float)
        )

        expected_delta_flow = ptdf_t.to_numpy(dtype=float) @ delta_np
        actual_delta_flow = (
            line_f_d1.loc[t, matched_lines].to_numpy(dtype=float)
            - line_f_d2.loc[t, matched_lines].to_numpy(dtype=float)
        )

        if expected_delta_flow.shape != actual_delta_flow.shape:
            row["fbme_status"] = "nan"
            row["fbme_reason"] = (
                f"shape_mismatch_expected_{expected_delta_flow.shape}_actual_{actual_delta_flow.shape}"
            )
            hourly_rows.append(row)
            continue

        fbme = actual_delta_flow - expected_delta_flow
        abs_fbme = np.abs(fbme)

        if abs_fbme.size > 0:
            row["mean_abs_fbme_d1_cgm"] = float(abs_fbme.mean())
            row["max_abs_fbme_d1_cgm"] = float(abs_fbme.max())
            row["p95_abs_fbme_d1_cgm"] = float(np.percentile(abs_fbme, 95))

        hourly_rows.append(row)

    return pd.DataFrame(hourly_rows).set_index("t").sort_index()


###############################################################################
# BUILD MERGED DATASET
###############################################################################

def build_analysis_dataset(run_dir: Path) -> pd.DataFrame:
    line_f_d1 = ensure_numeric_time_index(read_df(run_dir, "d1_cgm/line_f.parquet"))
    line_f_d2 = ensure_numeric_time_index(read_df(run_dir, "d2/line_f.parquet"))
    np_d1 = ensure_numeric_time_index(read_df(run_dir, "d1_cgm/np.parquet"))
    np_d2 = ensure_numeric_time_index(read_df(run_dir, "d2/np.parquet"))
    ptdf_long = read_df(run_dir, "fb/ptdf_z_cnec_long.parquet")
    line_cap_margin = read_line_cap_margin(run_dir)

    idx = (
        line_f_d1.index
        .intersection(line_f_d2.index)
        .intersection(np_d1.index)
        .intersection(np_d2.index)
        .sort_values()
    )

    state_df = compute_fbmc_state_features(idx)
    congestion_df = compute_congestion_features(
        line_f_d1=line_f_d1.loc[idx],
        line_cap_margin=line_cap_margin,
        threshold=CONGESTION_THRESHOLD,
    )
    fbme_df = compute_fbme_hourly_from_saved_ptdf(
        line_f_d1=line_f_d1.loc[idx],
        line_f_d2=line_f_d2.loc[idx],
        np_d1=np_d1.loc[idx],
        np_d2=np_d2.loc[idx],
        ptdf_long=ptdf_long,
    )

    df = pd.concat([state_df, congestion_df, fbme_df], axis=1).sort_index()

    df["demand_bin"] = qcut_3labels(df["total_demand_fbmc"])
    df["renewables_bin"] = qcut_3labels(df["renewable_share_fbmc"])
    df["congestion_bin"] = qcut_3labels(df["high_loading_share_d1"])

    return df


###############################################################################
# SUMMARIES
###############################################################################

def make_1d_bin_summary(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    return (
        df.groupby(bin_col, dropna=False)
        .agg(
            n_mtus=("mean_abs_fbme_d1_cgm", "size"),
            n_valid_fbme=("mean_abs_fbme_d1_cgm", lambda s: int(s.notna().sum())),
            mean_fbme=("mean_abs_fbme_d1_cgm", "mean"),
            median_fbme=("mean_abs_fbme_d1_cgm", "median"),
            p95_fbme=("mean_abs_fbme_d1_cgm", lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan),
            max_fbme=("max_abs_fbme_d1_cgm", "max"),
            mean_demand_fbmc=("total_demand_fbmc", "mean"),
            mean_renewables_fbmc=("total_renewables_fbmc", "mean"),
            mean_res_share_fbmc=("renewable_share_fbmc", "mean"),
            mean_high_loading_share_d1=("high_loading_share_d1", "mean"),
        )
        .reset_index()
    )


def make_2d_bin_summary(df: pd.DataFrame, row_bin: str, col_bin: str) -> pd.DataFrame:
    return (
        df.groupby([row_bin, col_bin], dropna=False)
        .agg(
            n_mtus=("mean_abs_fbme_d1_cgm", "size"),
            n_valid_fbme=("mean_abs_fbme_d1_cgm", lambda s: int(s.notna().sum())),
            mean_fbme=("mean_abs_fbme_d1_cgm", "mean"),
            median_fbme=("mean_abs_fbme_d1_cgm", "median"),
            p95_fbme=("mean_abs_fbme_d1_cgm", lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan),
            mean_high_loading_share_d1=("high_loading_share_d1", "mean"),
            mean_res_share_fbmc=("renewable_share_fbmc", "mean"),
            mean_demand_fbmc=("total_demand_fbmc", "mean"),
        )
        .reset_index()
    )


###############################################################################
# PLOTS
###############################################################################

def save_plots(strategy: str, df: pd.DataFrame):
    out_dir = OUTPUT_DIR / strategy / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_df = df[df["mean_abs_fbme_d1_cgm"].notna()].copy()
    if valid_df.empty:
        return

    tmp = valid_df.groupby("renewables_bin")["mean_abs_fbme_d1_cgm"].mean().reindex(BIN_LABELS)
    plt.figure(figsize=(6, 4))
    plt.bar(tmp.index.astype(str), tmp.to_numpy())
    plt.xlabel("Renewables bin")
    plt.ylabel("Mean hourly mean-abs FBME [MW]")
    plt.title(f"{strategy}: FBME by renewables bin")
    plt.tight_layout()
    plt.savefig(out_dir / "fbme_by_renewables_bin.png", dpi=PLOT_DPI)
    plt.close()

    tmp = valid_df.groupby("demand_bin")["mean_abs_fbme_d1_cgm"].mean().reindex(BIN_LABELS)
    plt.figure(figsize=(6, 4))
    plt.bar(tmp.index.astype(str), tmp.to_numpy())
    plt.xlabel("Demand bin")
    plt.ylabel("Mean hourly mean-abs FBME [MW]")
    plt.title(f"{strategy}: FBME by demand bin")
    plt.tight_layout()
    plt.savefig(out_dir / "fbme_by_demand_bin.png", dpi=PLOT_DPI)
    plt.close()

    tmp = valid_df.groupby("congestion_bin")["mean_abs_fbme_d1_cgm"].mean().reindex(BIN_LABELS)
    plt.figure(figsize=(6, 4))
    plt.bar(tmp.index.astype(str), tmp.to_numpy())
    plt.xlabel("Congestion bin")
    plt.ylabel("Mean hourly mean-abs FBME [MW]")
    plt.title(f"{strategy}: FBME by congestion bin")
    plt.tight_layout()
    plt.savefig(out_dir / "fbme_by_congestion_bin.png", dpi=PLOT_DPI)
    plt.close()

    heat = (
        valid_df.groupby(["renewables_bin", "congestion_bin"])["mean_abs_fbme_d1_cgm"]
        .mean()
        .unstack("congestion_bin")
        .reindex(index=BIN_LABELS, columns=BIN_LABELS)
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(heat.to_numpy(dtype=float), aspect="auto")
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("Congestion bin")
    plt.ylabel("Renewables bin")
    plt.title(f"{strategy}: Mean FBME heatmap")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.iloc[i, j]
            txt = "nan" if pd.isna(val) else f"{val:.2f}"
            plt.text(j, i, txt, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_fbme_renewables_x_congestion.png", dpi=PLOT_DPI)
    plt.close()


###############################################################################
# REPORTING
###############################################################################

def save_outputs(strategy: str, df: pd.DataFrame):
    out_dir = OUTPUT_DIR / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = df.reset_index().rename(columns={"index": "t"})
    dataset.to_csv(out_dir / "fbmc_bins_fbme_dataset.csv", index=False)
    dataset.to_parquet(out_dir / "fbmc_bins_fbme_dataset.parquet", index=False)

    demand_summary = make_1d_bin_summary(df, "demand_bin")
    renewables_summary = make_1d_bin_summary(df, "renewables_bin")
    congestion_summary = make_1d_bin_summary(df, "congestion_bin")

    demand_summary.to_csv(out_dir / "fbme_by_demand_bin.csv", index=False)
    demand_summary.to_parquet(out_dir / "fbme_by_demand_bin.parquet", index=False)

    renewables_summary.to_csv(out_dir / "fbme_by_renewables_bin.csv", index=False)
    renewables_summary.to_parquet(out_dir / "fbme_by_renewables_bin.parquet", index=False)

    congestion_summary.to_csv(out_dir / "fbme_by_congestion_bin.csv", index=False)
    congestion_summary.to_parquet(out_dir / "fbme_by_congestion_bin.parquet", index=False)

    d_x_r = make_2d_bin_summary(df, "demand_bin", "renewables_bin")
    d_x_c = make_2d_bin_summary(df, "demand_bin", "congestion_bin")
    r_x_c = make_2d_bin_summary(df, "renewables_bin", "congestion_bin")

    d_x_r.to_csv(out_dir / "fbme_by_demand_x_renewables_bins.csv", index=False)
    d_x_r.to_parquet(out_dir / "fbme_by_demand_x_renewables_bins.parquet", index=False)

    d_x_c.to_csv(out_dir / "fbme_by_demand_x_congestion_bins.csv", index=False)
    d_x_c.to_parquet(out_dir / "fbme_by_demand_x_congestion_bins.parquet", index=False)

    r_x_c.to_csv(out_dir / "fbme_by_renewables_x_congestion_bins.csv", index=False)
    r_x_c.to_parquet(out_dir / "fbme_by_renewables_x_congestion_bins.parquet", index=False)

    save_plots(strategy, df)


def print_summary(strategy: str, df: pd.DataFrame):
    print("\n" + "=" * 100)
    print(f"RUN: {strategy}")
    print("=" * 100)

    print("\nOverall summary:")
    print(df[[
        "total_demand_fbmc",
        "total_renewables_fbmc",
        "renewable_share_fbmc",
        "high_loading_share_d1",
        "mean_abs_fbme_d1_cgm",
        "max_abs_fbme_d1_cgm",
    ]].describe().to_string())

    print("\nMean FBME by renewables bin:")
    print(
        df.groupby("renewables_bin")["mean_abs_fbme_d1_cgm"]
        .mean()
        .reindex(BIN_LABELS)
        .to_string()
    )

    print("\nMean FBME by demand bin:")
    print(
        df.groupby("demand_bin")["mean_abs_fbme_d1_cgm"]
        .mean()
        .reindex(BIN_LABELS)
        .to_string()
    )

    print("\nMean FBME by congestion bin:")
    print(
        df.groupby("congestion_bin")["mean_abs_fbme_d1_cgm"]
        .mean()
        .reindex(BIN_LABELS)
        .to_string()
    )


###############################################################################
# MAIN
###############################################################################

def main():
    all_rows = []

    for strategy, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"Skipping missing run: {run_dir}")
            continue

        df = build_analysis_dataset(run_dir)
        save_outputs(strategy, df)
        print_summary(strategy, df)

        tmp = df.reset_index().rename(columns={"index": "t"})
        tmp["strategy"] = strategy
        all_rows.append(tmp)

    if not all_rows:
        raise RuntimeError("No valid runs found.")

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUTPUT_DIR / "all_runs_fbmc_bins_fbme_dataset.csv", index=False)
    all_df.to_parquet(OUTPUT_DIR / "all_runs_fbmc_bins_fbme_dataset.parquet", index=False)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()