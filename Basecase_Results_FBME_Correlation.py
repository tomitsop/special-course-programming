# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from input_data_base_functions import (
    L,
    Z_FBMC,
    N,
    get_dem,
    get_renew,
)

###############################################################################
# CONFIG
###############################################################################

RESULTS_BASE = Path("results")

RUNS = {
    # "NO_SCOPF_50%_pmax_sub": RESULTS_BASE / "pipeline_run_gurobi_NO_SOPF_50%_CAP_pmax_sub",
    # "pmax_sub": RESULTS_BASE / "pipeline_run_gurobi",
    "dynamic_gen": RESULTS_BASE / "pipeline_run_gurobi_dynamic_gen",
    # "dynamic_headroom": RESULTS_BASE / "pipeline_run_gurobi_dynamic_headroom",
    # "flat_unit": RESULTS_BASE / "pipeline_run_gurobi_flat_unit",
    # "SCOPF_pmax_sub": RESULTS_BASE / "pipeline_run_gurobi_SCOPF_pmax_sub",
    # "SCOPF_dynamic_gen": RESULTS_BASE / "pipeline_run_gurobi_SCOPF_dynamic_gen",
}

OUTPUT_DIR = RESULTS_BASE / "fbme_correlation_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_SCATTERS = 6
TOP_N_PRINT = 10
PLOT_DPI = 200

# Set to True if you want extra diagnostics saved/printed
DEBUG_FBME = True

# Better terminal display
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

###############################################################################
# HELPERS
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

    line_cap_margin_df = pd.read_parquet(path)
    if "line_cap_margin" not in line_cap_margin_df.columns:
        raise ValueError(f"'line_cap_margin' column not found in {path}")

    line_cap_margin = line_cap_margin_df["line_cap_margin"].copy()
    line_cap_margin.name = "line_cap_margin"
    return line_cap_margin


def align_line_cap_margin(line_cap_margin: pd.Series, flow_columns: pd.Index) -> pd.Series:
    aligned_margin = line_cap_margin.reindex(flow_columns)
    if aligned_margin.isna().any():
        margin_by_str = line_cap_margin.copy()
        margin_by_str.index = margin_by_str.index.map(str)
        aligned_margin = margin_by_str.reindex(flow_columns.map(str))
        aligned_margin.index = flow_columns

    missing_lines = aligned_margin[aligned_margin.isna()].index.tolist()
    if missing_lines:
        raise ValueError(
            "Missing line_cap_margin values for flow columns. "
            f"Example missing lines: {missing_lines[:10]}"
        )

    aligned_margin.name = "line_cap_margin"
    return aligned_margin


def prepare_ptdf_long(ptdf_long: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"t", "cnec_idx", "zone", "ptdf"}
    missing = required_cols - set(ptdf_long.columns)
    if missing:
        raise ValueError(f"ptdf_long is missing required columns: {sorted(missing)}")

    out = ptdf_long.copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce")
    out["cnec_idx"] = pd.to_numeric(out["cnec_idx"], errors="coerce")

    out = out.dropna(subset=["t", "cnec_idx", "zone", "ptdf"]).copy()
    out["t"] = out["t"].astype(int)
    out["cnec_idx"] = out["cnec_idx"].astype(int)

    return out


def build_ptdf_matrix_for_hour(
    ptdf_hour: pd.DataFrame,
    line_f_d1_columns: pd.Index,
    line_f_d2_columns: pd.Index,
) -> tuple[pd.DataFrame, list]:
    """
    Build PTDF_t with rows indexed by physical line names, restricted to lines
    present in both D-1 and D-2 flow tables.
    """
    if ptdf_hour.empty:
        empty = pd.DataFrame(columns=Z_FBMC, dtype=float)
        return empty, []

    # Use pivot_table instead of pivot to be robust to duplicate keys
    ptdf_t = ptdf_hour.pivot_table(
        index="cnec_idx",
        columns="zone",
        values="ptdf",
        aggfunc="first",
    ).reindex(columns=Z_FBMC)

    # Drop rows where cnec_idx is outside the valid line index range
    valid_mask = ptdf_t.index.to_series().between(0, len(L) - 1)
    ptdf_t = ptdf_t.loc[valid_mask]

    if ptdf_t.empty:
        empty = pd.DataFrame(columns=Z_FBMC, dtype=float)
        return empty, []

    idx_to_line = {i: L[i] for i in range(len(L))}
    ptdf_t.index = ptdf_t.index.map(idx_to_line)

    # Remove any duplicated physical line labels after mapping
    # If multiple CNECs map to the same physical line index, keep the first one.
    ptdf_t = ptdf_t[~ptdf_t.index.duplicated(keep="first")]

    matched_lines = [
        c for c in ptdf_t.index
        if c in line_f_d1_columns and c in line_f_d2_columns
    ]

    if not matched_lines:
        empty = pd.DataFrame(columns=Z_FBMC, dtype=float)
        return empty, []

    ptdf_t = ptdf_t.loc[matched_lines].astype(float)

    return ptdf_t, matched_lines


def compute_fbme_hourly(
    line_f_d1: pd.DataFrame,
    line_f_d2: pd.DataFrame,
    np_d1: pd.DataFrame,
    np_d2: pd.DataFrame,
    ptdf_long: pd.DataFrame,
    return_debug: bool = False,
):
    """
    Computes hourly FBME according to:

        FBME_{t,j} = F^{D1-CGM}_{t,j} - F^{D2-CGM}_{t,j}
                     - sum_z (NP^{D1-CGM}_{t,z} - NP^{D2-CGM}_{t,z}) * PTDF_{j,z}

    and returns the hourly mean absolute FBME across the matched CNEC lines j.
    """
    line_f_d1 = ensure_numeric_time_index(line_f_d1)
    line_f_d2 = ensure_numeric_time_index(line_f_d2)
    np_d1 = ensure_numeric_time_index(np_d1)
    np_d2 = ensure_numeric_time_index(np_d2)
    ptdf_long = prepare_ptdf_long(ptdf_long)

    # Keep ALL hours that exist in the solved D1/D2/NP outputs.
    # Do NOT intersect with ptdf_long["t"].unique(), because that drops hours
    # when an hour has zero PTDF/CNEC rows saved in the long table.
    common_t = (
        line_f_d1.index
        .intersection(line_f_d2.index)
        .intersection(np_d1.index)
        .intersection(np_d2.index)
        .sort_values()
    )

    ptdf_by_t = {
        t: grp.copy()
        for t, grp in ptdf_long.groupby("t", sort=False)
    }

    hourly_out = []
    debug_rows = []

    for t in common_t:
        ptdf_hour = ptdf_by_t.get(int(t), pd.DataFrame(columns=ptdf_long.columns))

        ptdf_t, matched_lines = build_ptdf_matrix_for_hour(
            ptdf_hour=ptdf_hour,
            line_f_d1_columns=line_f_d1.columns,
            line_f_d2_columns=line_f_d2.columns,
        )

        debug_row = {
            "t": int(t),
            "n_ptdf_rows_raw": int(len(ptdf_hour)),
            "n_unique_cnec_idx": int(ptdf_hour["cnec_idx"].nunique()) if not ptdf_hour.empty else 0,
            "n_matched_lines": int(len(matched_lines)),
            "status": "ok",
            "reason": "",
        }

        # If there are no PTDF rows or no matched lines, FBME is undefined for that hour.
        if ptdf_t.empty or len(matched_lines) == 0:
            debug_row["status"] = "nan"
            debug_row["reason"] = "no_ptdf_rows_or_no_matched_lines"
            hourly_out.append((t, np.nan))
            debug_rows.append(debug_row)
            continue

        # Ensure FBMC zone columns exist
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
            debug_row["status"] = "nan"
            debug_row["reason"] = (
                f"shape_mismatch_expected_{expected_delta_flow.shape}_actual_{actual_delta_flow.shape}"
            )
            hourly_out.append((t, np.nan))
            debug_rows.append(debug_row)
            continue

        fbme = actual_delta_flow - expected_delta_flow

        if fbme.size == 0:
            debug_row["status"] = "nan"
            debug_row["reason"] = "empty_fbme_vector"
            hourly_out.append((t, np.nan))
            debug_rows.append(debug_row)
            continue

        debug_row["mean_abs_fbme_d1_cgm"] = float(np.abs(fbme).mean())
        hourly_out.append((t, float(np.abs(fbme).mean())))
        debug_rows.append(debug_row)

    hourly_series = pd.Series(
        [v for _, v in hourly_out],
        index=[t for t, _ in hourly_out],
        name="mean_abs_fbme_d1_cgm",
    ).sort_index()

    debug_df = pd.DataFrame(debug_rows).set_index("t").sort_index()

    if return_debug:
        return hourly_series, debug_df
    return hourly_series


def build_d1_overload_df(
    line_f_d1: pd.DataFrame,
    line_cap_margin: pd.Series,
) -> pd.DataFrame:
    """
    Overload severity above admissible margin, otherwise NaN.
    """
    margin = align_line_cap_margin(line_cap_margin, line_f_d1.columns)
    severity = line_f_d1.abs().sub(margin, axis=1)
    return severity.where(severity > 0)


def compute_hourly_overload_metrics(
    line_f_d1: pd.DataFrame,
    line_cap_margin: pd.Series,
) -> pd.DataFrame:
    overload_df = build_d1_overload_df(line_f_d1, line_cap_margin)

    out = pd.DataFrame(index=line_f_d1.index)
    out["hourly_d1_overloads"] = overload_df.count(axis=1).astype(int)
    out["hourly_d1_overload_mw"] = overload_df.fillna(0.0).sum(axis=1)

    return out


def compute_grid_metrics(
    line_f_d1: pd.DataFrame,
    np_d1: pd.DataFrame,
    np_d2: pd.DataFrame,
    obj: pd.DataFrame,
    line_cap_margin: pd.Series,
) -> pd.DataFrame:
    margin = align_line_cap_margin(line_cap_margin, line_f_d1.columns)
    loading = line_f_d1.abs().div(margin, axis=1)

    out = pd.DataFrame(index=line_f_d1.index)
    out["sum_abs_np_diff"] = (np_d1[Z_FBMC] - np_d2[Z_FBMC]).abs().sum(axis=1)
    out["mean_line_loading"] = loading.mean(axis=1)
    out["max_line_loading"] = loading.max(axis=1)

    if "d1_cgm" in obj.columns:
        out["d1_cgm_objective"] = obj["d1_cgm"]
    if "d0" in obj.columns:
        out["d0_objective"] = obj["d0"]

    return out


def compute_system_state_features(time_index) -> pd.DataFrame:
    rows = []
    for t in time_index:
        total_demand = float(sum(get_dem(int(t), n) for n in N))
        total_renewables = float(sum(get_renew(int(t), n) for n in N))
        renewable_share = total_renewables / total_demand if total_demand != 0 else np.nan

        rows.append({
            "t": int(t),
            "total_demand": total_demand,
            "total_renewables": total_renewables,
            "renewable_share": renewable_share,
        })

    return pd.DataFrame(rows).set_index("t").sort_index()


def build_analysis_df(run_dir: Path):
    obj = ensure_numeric_time_index(read_df(run_dir, "objectives.parquet"))
    line_f_d1 = ensure_numeric_time_index(read_df(run_dir, "d1_cgm/line_f.parquet"))
    line_f_d2 = ensure_numeric_time_index(read_df(run_dir, "d2/line_f.parquet"))

    # Keep D1-CGM NP here, because this matches the formula you want to compute.
    np_d1 = ensure_numeric_time_index(read_df(run_dir, "d1_cgm/np.parquet"))
    np_d2 = ensure_numeric_time_index(read_df(run_dir, "d2/np.parquet"))

    ptdf_long = read_df(run_dir, "fb/ptdf_z_cnec_long.parquet")
    line_cap_margin = read_line_cap_margin(run_dir)

    hourly_fbme, fbme_debug = compute_fbme_hourly(
        line_f_d1=line_f_d1,
        line_f_d2=line_f_d2,
        np_d1=np_d1,
        np_d2=np_d2,
        ptdf_long=ptdf_long,
        return_debug=True,
    )

    idx = hourly_fbme.index

    system_df = compute_system_state_features(idx)
    overload_df = compute_hourly_overload_metrics(
        line_f_d1.loc[idx],
        line_cap_margin,
    )
    grid_df = compute_grid_metrics(
        line_f_d1=line_f_d1.loc[idx],
        np_d1=np_d1.loc[idx],
        np_d2=np_d2.loc[idx],
        obj=obj.loc[idx],
        line_cap_margin=line_cap_margin,
    )

    analysis_df = pd.concat(
        [hourly_fbme, system_df, overload_df, grid_df],
        axis=1,
    ).sort_index()

    analysis_df = analysis_df.join(
        fbme_debug[["n_ptdf_rows_raw", "n_unique_cnec_idx", "n_matched_lines", "status", "reason"]],
        how="left",
    )

    return analysis_df, fbme_debug


def compute_correlation_table(
    analysis_df: pd.DataFrame,
    target_col: str = "mean_abs_fbme_d1_cgm",
) -> pd.DataFrame:
    numeric_df = analysis_df.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in numeric_df.columns if c != target_col]

    rows = []
    for col in feature_cols:
        tmp = numeric_df[[target_col, col]].dropna()

        if len(tmp) < 3:
            pearson_corr = np.nan
            spearman_corr = np.nan
        else:
            pearson_corr = tmp[target_col].corr(tmp[col], method="pearson")
            spearman_corr = tmp[target_col].corr(tmp[col], method="spearman")

        rows.append({
            "variable": col,
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
            "abs_pearson_corr": abs(pearson_corr) if pd.notna(pearson_corr) else np.nan,
            "abs_spearman_corr": abs(spearman_corr) if pd.notna(spearman_corr) else np.nan,
            "n_obs": len(tmp),
        })

    return pd.DataFrame(rows).sort_values(
        ["abs_spearman_corr", "abs_pearson_corr"],
        ascending=False,
    )


def save_run_plots(strategy: str, analysis_df: pd.DataFrame, corr_df: pd.DataFrame):
    run_plot_dir = OUTPUT_DIR / strategy / "plots"
    run_plot_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = analysis_df.select_dtypes(include=[np.number]).copy()

    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            numeric_df.corr(method="spearman"),
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
        )
        plt.title(f"Spearman correlation matrix - {strategy}")
        plt.tight_layout()
        plt.savefig(run_plot_dir / "correlation_heatmap_spearman.png", dpi=PLOT_DPI)
        plt.close()

    top_vars = corr_df["variable"].head(TOP_N_SCATTERS).tolist()

    for var in top_vars:
        tmp = analysis_df[[var, "mean_abs_fbme_d1_cgm"]].dropna()
        if len(tmp) < 3:
            continue

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=tmp,
            x=var,
            y="mean_abs_fbme_d1_cgm",
            alpha=0.4,
            s=25,
        )
        sns.regplot(
            data=tmp,
            x=var,
            y="mean_abs_fbme_d1_cgm",
            scatter=False,
            ci=None,
            line_kws={"linewidth": 2, "color": "red"},
        )
        plt.title(f"{strategy}: FBME vs {var}")
        plt.xlabel(var)
        plt.ylabel("Hourly mean absolute FBME [MW]")
        plt.tight_layout()
        safe_var = str(var).replace("/", "_").replace(" ", "_")
        plt.savefig(run_plot_dir / f"scatter_fbme_vs_{safe_var}.png", dpi=PLOT_DPI)
        plt.close()


def save_run_report(strategy: str, analysis_df: pd.DataFrame, corr_df: pd.DataFrame):
    run_out_dir = OUTPUT_DIR / strategy
    run_out_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_out_dir / "report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FBME CORRELATION REPORT\n")
        f.write(f"Strategy: {strategy}\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of MTUs (all common solved hours): {len(analysis_df)}\n")
        f.write(f"Number of MTUs with valid FBME: {analysis_df['mean_abs_fbme_d1_cgm'].notna().sum()}\n")
        f.write(f"Mean absolute FBME: {analysis_df['mean_abs_fbme_d1_cgm'].mean():.6f}\n")
        f.write(f"Mean hourly overload count: {analysis_df['hourly_d1_overloads'].mean():.6f}\n")
        f.write(f"Mean hourly overload MW: {analysis_df['hourly_d1_overload_mw'].mean():.6f}\n")
        f.write(f"Mean total demand: {analysis_df['total_demand'].mean():.6f}\n")
        f.write(f"Mean total renewables: {analysis_df['total_renewables'].mean():.6f}\n")
        f.write(f"Mean renewable share: {analysis_df['renewable_share'].mean():.6f}\n")
        f.write(f"Mean sum_abs_np_diff: {analysis_df['sum_abs_np_diff'].mean():.6f}\n")
        f.write(f"Mean max line loading: {analysis_df['max_line_loading'].mean():.6f}\n\n")

        if "reason" in analysis_df.columns:
            f.write("FBME INVALID-HOUR REASONS\n")
            f.write("-" * 80 + "\n")
            reason_counts = analysis_df["reason"].fillna("").replace("", "ok").value_counts()
            f.write(reason_counts.to_string())
            f.write("\n\n")

        f.write("TOP CORRELATION DRIVERS\n")
        f.write("-" * 80 + "\n")
        f.write(corr_df.head(TOP_N_PRINT).to_string(index=False))
        f.write("\n")


def print_run_summary(strategy: str, analysis_df: pd.DataFrame, corr_df: pd.DataFrame):
    print("\n" + "=" * 100)
    print(f"RUN: {strategy}")
    print("=" * 100)

    summary = {
        "n_mtus_total": int(len(analysis_df)),
        "n_mtus_valid_fbme": int(analysis_df["mean_abs_fbme_d1_cgm"].notna().sum()),
        "mean_abs_fbme_d1_cgm": float(analysis_df["mean_abs_fbme_d1_cgm"].mean()),
        "mean_hourly_d1_overloads": float(analysis_df["hourly_d1_overloads"].mean()),
        "mean_hourly_d1_overload_mw": float(analysis_df["hourly_d1_overload_mw"].mean()),
        "mean_sum_abs_np_diff": float(analysis_df["sum_abs_np_diff"].mean()),
        "mean_total_demand": float(analysis_df["total_demand"].mean()),
        "mean_total_renewables": float(analysis_df["total_renewables"].mean()),
        "mean_renewable_share": float(analysis_df["renewable_share"].mean()),
        "mean_max_line_loading": float(analysis_df["max_line_loading"].mean()),
    }

    print("Summary metrics:")
    for k, v in summary.items():
        print(f"  {k:<30} {v:.6f}" if isinstance(v, float) else f"  {k:<30} {v}")

    if "reason" in analysis_df.columns:
        print("\nFBME invalid-hour reasons:")
        print(analysis_df["reason"].fillna("").replace("", "ok").value_counts().to_string())

    print("\nTop correlation drivers:")
    print(corr_df[["variable", "pearson_corr", "spearman_corr", "n_obs"]].head(TOP_N_PRINT).to_string(index=False))


###############################################################################
# MAIN
###############################################################################

def main():
    sns.set_style("whitegrid")

    all_analysis = []
    all_corr_summary = []
    run_summary_rows = []

    for strategy, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"Skipping missing folder: {run_dir}")
            continue

        print(f"\nAnalyzing run: {strategy}")

        try:
            analysis_df, fbme_debug = build_analysis_df(run_dir)
        except Exception as e:
            print(f"Failed for {strategy}: {e}")
            continue

        if analysis_df.empty:
            print(f"No analysis data for {strategy}")
            continue

        analysis_df = analysis_df.copy()
        analysis_df["strategy"] = strategy
        analysis_df["t"] = analysis_df.index

        corr_df = compute_correlation_table(
            analysis_df.drop(columns=["strategy", "t", "status", "reason"], errors="ignore")
        )
        corr_df["strategy"] = strategy

        save_run_plots(
            strategy,
            analysis_df.drop(columns=["strategy", "t", "status", "reason"], errors="ignore"),
            corr_df
        )
        save_run_report(
            strategy,
            analysis_df.drop(columns=["strategy", "t"], errors="ignore"),
            corr_df.drop(columns=["strategy"], errors="ignore")
        )
        print_run_summary(
            strategy,
            analysis_df.drop(columns=["strategy", "t"], errors="ignore"),
            corr_df.drop(columns=["strategy"], errors="ignore")
        )

        run_summary_rows.append({
            "strategy": strategy,
            "n_mtus_total": int(len(analysis_df)),
            "n_mtus_valid_fbme": int(analysis_df["mean_abs_fbme_d1_cgm"].notna().sum()),
            "mean_abs_fbme_d1_cgm": float(analysis_df["mean_abs_fbme_d1_cgm"].mean()),
            "mean_hourly_d1_overloads": float(analysis_df["hourly_d1_overloads"].mean()),
            "mean_hourly_d1_overload_mw": float(analysis_df["hourly_d1_overload_mw"].mean()),
            "mean_sum_abs_np_diff": float(analysis_df["sum_abs_np_diff"].mean()),
            "mean_total_demand": float(analysis_df["total_demand"].mean()),
            "mean_total_renewables": float(analysis_df["total_renewables"].mean()),
            "mean_renewable_share": float(analysis_df["renewable_share"].mean()),
            "mean_max_line_loading": float(analysis_df["max_line_loading"].mean()),
            "top_spearman_driver": corr_df.iloc[0]["variable"] if not corr_df.empty else np.nan,
            "top_spearman_value": float(corr_df.iloc[0]["spearman_corr"]) if not corr_df.empty else np.nan,
            "top_pearson_driver": corr_df.iloc[0]["variable"] if not corr_df.empty else np.nan,
            "top_pearson_value": float(corr_df.iloc[0]["pearson_corr"]) if not corr_df.empty else np.nan,
        })

        run_out_dir = OUTPUT_DIR / strategy
        run_out_dir.mkdir(parents=True, exist_ok=True)

        analysis_df.to_csv(run_out_dir / "fbme_analysis_dataset.csv", index=False)
        analysis_df.to_parquet(run_out_dir / "fbme_analysis_dataset.parquet", index=False)

        corr_df.to_csv(run_out_dir / "fbme_correlations.csv", index=False)
        corr_df.to_parquet(run_out_dir / "fbme_correlations.parquet", index=False)

        if DEBUG_FBME:
            fbme_debug.to_csv(run_out_dir / "fbme_debug.csv", index=True)
            fbme_debug.to_parquet(run_out_dir / "fbme_debug.parquet", index=True)

        all_analysis.append(analysis_df.reset_index(drop=True))
        all_corr_summary.append(corr_df)

    if not all_analysis:
        raise RuntimeError("No valid runs were analyzed.")

    combined_analysis_df = pd.concat(all_analysis, ignore_index=True)
    combined_corr_df = pd.concat(all_corr_summary, ignore_index=True)
    summary_df = pd.DataFrame(run_summary_rows).sort_values("strategy")

    combined_analysis_df.to_csv(OUTPUT_DIR / "all_runs_fbme_analysis_dataset.csv", index=False)
    combined_analysis_df.to_parquet(OUTPUT_DIR / "all_runs_fbme_analysis_dataset.parquet", index=False)

    combined_corr_df.to_csv(OUTPUT_DIR / "all_runs_fbme_correlations.csv", index=False)
    combined_corr_df.to_parquet(OUTPUT_DIR / "all_runs_fbme_correlations.parquet", index=False)

    summary_df.to_csv(OUTPUT_DIR / "all_runs_summary.csv", index=False)
    summary_df.to_parquet(OUTPUT_DIR / "all_runs_summary.parquet", index=False)

    print("\n" + "=" * 100)
    print("FBME CORRELATION SUMMARY ACROSS RUNS")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("TRANSPOSED SUMMARY")
    print("=" * 100)
    print(summary_df.set_index("strategy").T.to_string())

    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()