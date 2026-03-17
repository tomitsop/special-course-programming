# Basecase_Results_FBME_Diagnostics.py

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
    "dynamic_gen": RESULTS_BASE / "pipeline_run_gurobi_dynamic_gen",
    # "dynamic_headroom": RESULTS_BASE / "pipeline_run_gurobi_dynamic_headroom",
    # "flat_unit": RESULTS_BASE / "pipeline_run_gurobi_flat_unit",
    # "pmax_sub": RESULTS_BASE / "pipeline_run_gurobi",
    # "SCOPF_dynamic_gen": RESULTS_BASE / "pipeline_run_gurobi_SCOPF_dynamic_gen",
}

OUTPUT_DIR = RESULTS_BASE / "fbme_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_MTUS = 50
TOP_N_LINES_PER_MTU = 25
PLOT_DPI = 200

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", None)
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

###############################################################################
# PTDF / FBME HELPERS
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


def prepare_ram_long(ram_long: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"t", "cnec", "cnec_idx", "ram_pos", "ram_neg"}
    missing = required_cols - set(ram_long.columns)
    if missing:
        raise ValueError(f"ram_long is missing required columns: {sorted(missing)}")

    out = ram_long.copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce")
    out["cnec_idx"] = pd.to_numeric(out["cnec_idx"], errors="coerce")
    out["ram_pos"] = pd.to_numeric(out["ram_pos"], errors="coerce")
    out["ram_neg"] = pd.to_numeric(out["ram_neg"], errors="coerce")

    out = out.dropna(subset=["t", "cnec_idx", "ram_pos", "ram_neg"]).copy()
    out["t"] = out["t"].astype(int)
    out["cnec_idx"] = out["cnec_idx"].astype(int)

    return out


def build_ptdf_matrix_for_hour(
    ptdf_hour: pd.DataFrame,
    line_f_d1_columns: pd.Index,
    line_f_d2_columns: pd.Index,
):
    """
    Returns
    -------
    ptdf_t : DataFrame
        index = matched physical line labels
        columns = Z_FBMC
    matched_lines : list[str]
    cnec_meta : DataFrame
        columns = ['line', 'cnec_idx']
        one row per matched line retained in ptdf_t
    """
    empty_ptdf = pd.DataFrame(columns=Z_FBMC, dtype=float)
    empty_meta = pd.DataFrame(columns=["line", "cnec_idx"])

    if ptdf_hour.empty:
        return empty_ptdf, [], empty_meta

    ptdf_t = ptdf_hour.pivot_table(
        index="cnec_idx",
        columns="zone",
        values="ptdf",
        aggfunc="first",
    ).reindex(columns=Z_FBMC)

    valid_mask = ptdf_t.index.to_series().between(0, len(L) - 1)
    ptdf_t = ptdf_t.loc[valid_mask]

    if ptdf_t.empty:
        return empty_ptdf, [], empty_meta

    idx_to_line = {i: L[i] for i in range(len(L))}
    mapped_lines = ptdf_t.index.map(idx_to_line)
    ptdf_t.index = mapped_lines

    # preserve one row per physical line
    keep_mask = ~ptdf_t.index.duplicated(keep="first")
    kept_cnec_idx = pd.Index(
        [idx for idx, keep in zip(ptdf_t.index, keep_mask) if keep]
    )  # only used for length sanity if needed

    original_cnec_idx = ptdf_t.reset_index(drop=True)
    ptdf_t = ptdf_t.loc[keep_mask]

    matched_lines = [
        line for line in ptdf_t.index
        if line in line_f_d1_columns and line in line_f_d2_columns
    ]

    if not matched_lines:
        return empty_ptdf, [], empty_meta

    ptdf_t = ptdf_t.loc[matched_lines].astype(float)

    # rebuild cnec_meta consistently from the original hour data
    tmp = (
        ptdf_hour[["cnec_idx"]]
        .drop_duplicates()
        .copy()
    )
    tmp = tmp[tmp["cnec_idx"].between(0, len(L) - 1)]
    tmp["line"] = tmp["cnec_idx"].map(idx_to_line)
    tmp = tmp.drop_duplicates(subset=["line"], keep="first")
    tmp = tmp[tmp["line"].isin(matched_lines)].copy()
    tmp = tmp.set_index("line").reindex(matched_lines).reset_index()

    return ptdf_t, matched_lines, tmp


def compute_hourly_and_line_fbme(
    line_f_d1: pd.DataFrame,
    line_f_d2: pd.DataFrame,
    np_d1: pd.DataFrame,
    np_d2: pd.DataFrame,
    ptdf_long: pd.DataFrame,
    ram_long: pd.DataFrame,
    line_cap_margin: pd.Series,
):
    """
    Builds:
      - hourly_df: one row per MTU
      - line_df: one row per (MTU, matched line)

    Formula:
        FBME_{t,j} = F^{D1-CGM}_{t,j} - F^{D2-CGM}_{t,j}
                     - sum_z (NP^{D1-CGM}_{t,z} - NP^{D2-CGM}_{t,z}) * PTDF_{j,z}
    """
    line_f_d1 = ensure_numeric_time_index(line_f_d1)
    line_f_d2 = ensure_numeric_time_index(line_f_d2)
    np_d1 = ensure_numeric_time_index(np_d1)
    np_d2 = ensure_numeric_time_index(np_d2)

    ptdf_long = prepare_ptdf_long(ptdf_long)
    ram_long = prepare_ram_long(ram_long)

    margin_aligned = align_line_cap_margin(line_cap_margin, line_f_d1.columns)

    common_t = (
        line_f_d1.index
        .intersection(line_f_d2.index)
        .intersection(np_d1.index)
        .intersection(np_d2.index)
        .sort_values()
    )

    ptdf_by_t = {int(t): grp.copy() for t, grp in ptdf_long.groupby("t", sort=False)}
    ram_by_t = {int(t): grp.copy() for t, grp in ram_long.groupby("t", sort=False)}

    hourly_rows = []
    line_rows = []

    for t in common_t:
        t_int = int(t)
        ptdf_hour = ptdf_by_t.get(t_int, pd.DataFrame(columns=ptdf_long.columns))
        ram_hour = ram_by_t.get(t_int, pd.DataFrame(columns=ram_long.columns))

        ptdf_t, matched_lines, cnec_meta = build_ptdf_matrix_for_hour(
            ptdf_hour=ptdf_hour,
            line_f_d1_columns=line_f_d1.columns,
            line_f_d2_columns=line_f_d2.columns,
        )

        base_row = {
            "t": t_int,
            "n_ptdf_rows_raw": int(len(ptdf_hour)),
            "n_ram_rows_raw": int(len(ram_hour)),
            "n_matched_lines": int(len(matched_lines)),
            "status": "ok",
            "reason": "",
        }

        if ptdf_t.empty or len(matched_lines) == 0:
            base_row.update({
                "mean_abs_fbme": np.nan,
                "max_abs_fbme": np.nan,
                "std_abs_fbme": np.nan,
                "sum_abs_fbme": np.nan,
                "mean_abs_actual_delta_flow": np.nan,
                "mean_abs_expected_delta_flow": np.nan,
                "sum_abs_np_diff": float((np_d1.loc[t, Z_FBMC] - np_d2.loc[t, Z_FBMC]).abs().sum()),
                "max_line_loading_d1": np.nan,
                "n_d1_overloaded_matched_lines": np.nan,
            })
            base_row["status"] = "nan"
            base_row["reason"] = "no_ptdf_rows_or_no_matched_lines"
            hourly_rows.append(base_row)
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
            base_row.update({
                "mean_abs_fbme": np.nan,
                "max_abs_fbme": np.nan,
                "std_abs_fbme": np.nan,
                "sum_abs_fbme": np.nan,
                "mean_abs_actual_delta_flow": np.nan,
                "mean_abs_expected_delta_flow": np.nan,
                "sum_abs_np_diff": float((np_d1.loc[t, Z_FBMC] - np_d2.loc[t, Z_FBMC]).abs().sum()),
                "max_line_loading_d1": np.nan,
                "n_d1_overloaded_matched_lines": np.nan,
            })
            base_row["status"] = "nan"
            base_row["reason"] = (
                f"shape_mismatch_expected_{expected_delta_flow.shape}_actual_{actual_delta_flow.shape}"
            )
            hourly_rows.append(base_row)
            continue

        fbme = actual_delta_flow - expected_delta_flow
        abs_fbme = np.abs(fbme)

        d1_flow = line_f_d1.loc[t, matched_lines].to_numpy(dtype=float)
        d2_flow = line_f_d2.loc[t, matched_lines].to_numpy(dtype=float)
        line_margin = margin_aligned.reindex(matched_lines).to_numpy(dtype=float)
        d1_loading = np.abs(d1_flow) / line_margin
        d1_overloaded = np.abs(d1_flow) > line_margin

        # RAM metadata by cnec_idx
        ram_info = (
            ram_hour[["cnec_idx", "cnec", "ram_pos", "ram_neg"]]
            .drop_duplicates(subset=["cnec_idx"], keep="first")
            .copy()
        )
        cnec_meta = cnec_meta.rename(columns={"cnec_idx": "meta_cnec_idx"}).copy()
        if "meta_cnec_idx" in cnec_meta.columns:
            cnec_meta["meta_cnec_idx"] = pd.to_numeric(cnec_meta["meta_cnec_idx"], errors="coerce")

        # build line-level rows
        for i, line_name in enumerate(matched_lines):
            meta_row = cnec_meta.loc[cnec_meta["line"] == line_name]
            cnec_idx_val = int(meta_row["meta_cnec_idx"].iloc[0]) if not meta_row.empty else np.nan

            ram_row = ram_info.loc[ram_info["cnec_idx"] == cnec_idx_val]
            cnec_name = ram_row["cnec"].iloc[0] if not ram_row.empty else line_name
            ram_pos = float(ram_row["ram_pos"].iloc[0]) if not ram_row.empty else np.nan
            ram_neg = float(ram_row["ram_neg"].iloc[0]) if not ram_row.empty else np.nan

            row = {
                "t": t_int,
                "line": line_name,
                "cnec": cnec_name,
                "cnec_idx": cnec_idx_val,
                "actual_delta_flow": float(actual_delta_flow[i]),
                "expected_delta_flow": float(expected_delta_flow[i]),
                "fbme": float(fbme[i]),
                "abs_fbme": float(abs_fbme[i]),
                "flow_d1": float(d1_flow[i]),
                "flow_d2": float(d2_flow[i]),
                "d1_loading": float(d1_loading[i]),
                "is_d1_overloaded": bool(d1_overloaded[i]),
                "line_cap_margin": float(line_margin[i]),
                "ram_pos": ram_pos,
                "ram_neg": ram_neg,
            }

            for z in Z_FBMC:
                row[f"delta_np__{z}"] = float(np_d1.loc[t, z] - np_d2.loc[t, z])
                row[f"ptdf__{z}"] = float(ptdf_t.loc[line_name, z])

            line_rows.append(row)

        # hourly row
        base_row.update({
            "mean_abs_fbme": float(abs_fbme.mean()),
            "max_abs_fbme": float(abs_fbme.max()),
            "std_abs_fbme": float(abs_fbme.std(ddof=0)),
            "sum_abs_fbme": float(abs_fbme.sum()),
            "mean_abs_actual_delta_flow": float(np.abs(actual_delta_flow).mean()),
            "mean_abs_expected_delta_flow": float(np.abs(expected_delta_flow).mean()),
            "sum_abs_np_diff": float((np_d1.loc[t, Z_FBMC] - np_d2.loc[t, Z_FBMC]).abs().sum()),
            "max_line_loading_d1": float(d1_loading.max()),
            "n_d1_overloaded_matched_lines": int(d1_overloaded.sum()),
        })
        hourly_rows.append(base_row)

    hourly_df = pd.DataFrame(hourly_rows).sort_values("t").reset_index(drop=True)
    line_df = pd.DataFrame(line_rows).sort_values(["t", "abs_fbme"], ascending=[True, False]).reset_index(drop=True)

    return hourly_df, line_df

###############################################################################
# FEATURE HELPERS
###############################################################################

def compute_system_state_features(time_index) -> pd.DataFrame:
    rows = []
    for t in time_index:
        t_int = int(t)
        total_demand = float(sum(get_dem(t_int, n) for n in N))
        total_renewables = float(sum(get_renew(t_int, n) for n in N))
        renewable_share = total_renewables / total_demand if total_demand != 0 else np.nan

        rows.append({
            "t": t_int,
            "total_demand": total_demand,
            "total_renewables": total_renewables,
            "renewable_share": renewable_share,
        })

    return pd.DataFrame(rows)


def attach_hourly_context(
    hourly_df: pd.DataFrame,
    obj: pd.DataFrame,
    np_d1: pd.DataFrame,
    np_d2: pd.DataFrame,
):
    out = hourly_df.copy()

    obj_local = ensure_numeric_time_index(obj)
    obj_local = obj_local.reset_index().rename(columns={obj_local.index.name or "index": "t"})
    obj_local["t"] = pd.to_numeric(obj_local["t"], errors="coerce").astype("Int64")

    state_df = compute_system_state_features(out["t"].tolist())

    np_tmp = (
        ensure_numeric_time_index(np_d1)[Z_FBMC]
        .sub(ensure_numeric_time_index(np_d2)[Z_FBMC], fill_value=np.nan)
        .abs()
        .sum(axis=1)
        .rename("sum_abs_np_diff_check")
        .reset_index()
    )
    np_tmp = np_tmp.rename(columns={np_tmp.columns[0]: "t"})
    np_tmp["t"] = pd.to_numeric(np_tmp["t"], errors="coerce").astype("Int64")

    out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("Int64")
    out = out.merge(state_df, on="t", how="left")
    out = out.merge(obj_local, on="t", how="left")
    out = out.merge(np_tmp, on="t", how="left")

    return out

###############################################################################
# REPORTING / PLOTS
###############################################################################

def save_top_mtu_report(strategy: str, hourly_df: pd.DataFrame, line_df: pd.DataFrame):
    run_out_dir = OUTPUT_DIR / strategy
    run_out_dir.mkdir(parents=True, exist_ok=True)

    valid_hourly = hourly_df[hourly_df["mean_abs_fbme"].notna()].copy()
    top_mtus = valid_hourly.nlargest(TOP_N_MTUS, "mean_abs_fbme").copy()

    report_path = run_out_dir / "top_mtu_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FBME DIAGNOSTICS REPORT\n")
        f.write(f"Strategy: {strategy}\n")
        f.write("=" * 100 + "\n\n")

        f.write("GLOBAL SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"Number of MTUs total: {len(hourly_df)}\n")
        f.write(f"Number of MTUs with valid FBME: {valid_hourly['mean_abs_fbme'].notna().sum()}\n")
        f.write(f"Mean hourly mean_abs_fbme: {valid_hourly['mean_abs_fbme'].mean():.6f}\n")
        f.write(f"Mean hourly max_abs_fbme: {valid_hourly['max_abs_fbme'].mean():.6f}\n")
        f.write(f"Max hourly mean_abs_fbme: {valid_hourly['mean_abs_fbme'].max():.6f}\n")
        f.write(f"Max line abs_fbme overall: {line_df['abs_fbme'].max():.6f}\n\n")

        f.write("INVALID-HOUR REASONS\n")
        f.write("-" * 100 + "\n")
        f.write(hourly_df["reason"].fillna("").replace("", "ok").value_counts().to_string())
        f.write("\n\n")

        f.write(f"TOP {min(TOP_N_MTUS, len(top_mtus))} MTUS BY mean_abs_fbme\n")
        f.write("-" * 100 + "\n")
        cols = [
            "t",
            "mean_abs_fbme",
            "max_abs_fbme",
            "std_abs_fbme",
            "n_matched_lines",
            "sum_abs_np_diff",
            "total_demand",
            "total_renewables",
            "renewable_share",
            "max_line_loading_d1",
            "n_d1_overloaded_matched_lines",
        ]
        cols = [c for c in cols if c in top_mtus.columns]
        f.write(top_mtus[cols].to_string(index=False))
        f.write("\n\n")

        f.write("TOP LINES WITH HIGHEST abs_fbme ACROSS ALL HOURS\n")
        f.write("-" * 100 + "\n")
        top_lines = line_df.nlargest(50, "abs_fbme").copy()
        cols = [
            "t",
            "line",
            "cnec",
            "cnec_idx",
            "abs_fbme",
            "fbme",
            "actual_delta_flow",
            "expected_delta_flow",
            "flow_d1",
            "flow_d2",
            "d1_loading",
            "is_d1_overloaded",
        ]
        cols = [c for c in cols if c in top_lines.columns]
        f.write(top_lines[cols].to_string(index=False))
        f.write("\n")


def save_top_mtu_line_tables(strategy: str, hourly_df: pd.DataFrame, line_df: pd.DataFrame):
    run_out_dir = OUTPUT_DIR / strategy
    run_out_dir.mkdir(parents=True, exist_ok=True)

    valid_hourly = hourly_df[hourly_df["mean_abs_fbme"].notna()].copy()
    top_mtus = valid_hourly.nlargest(TOP_N_MTUS, "mean_abs_fbme").copy()
    top_mtus.to_csv(run_out_dir / "top_mtus.csv", index=False)
    top_mtus.to_parquet(run_out_dir / "top_mtus.parquet", index=False)

    top_lines_global = line_df.nlargest(500, "abs_fbme").copy()
    top_lines_global.to_csv(run_out_dir / "top_lines_global.csv", index=False)
    top_lines_global.to_parquet(run_out_dir / "top_lines_global.parquet", index=False)

    # one compact table with top lines per selected MTU
    top_line_rows = []
    selected_t = top_mtus["t"].tolist()

    for t in selected_t:
        tmp = line_df.loc[line_df["t"] == t].nlargest(TOP_N_LINES_PER_MTU, "abs_fbme").copy()
        top_line_rows.append(tmp)

    if top_line_rows:
        top_lines_per_mtu = pd.concat(top_line_rows, ignore_index=True)
    else:
        top_lines_per_mtu = pd.DataFrame(columns=line_df.columns)

    top_lines_per_mtu.to_csv(run_out_dir / "top_lines_per_top_mtu.csv", index=False)
    top_lines_per_mtu.to_parquet(run_out_dir / "top_lines_per_top_mtu.parquet", index=False)


def save_summary_plots(strategy: str, hourly_df: pd.DataFrame, line_df: pd.DataFrame):
    run_plot_dir = OUTPUT_DIR / strategy / "plots"
    run_plot_dir.mkdir(parents=True, exist_ok=True)

    valid_hourly = hourly_df[hourly_df["mean_abs_fbme"].notna()].copy()
    if valid_hourly.empty:
        return

    # Time series of hourly mean FBME
    plt.figure(figsize=(12, 5))
    plt.plot(valid_hourly["t"].to_numpy(), valid_hourly["mean_abs_fbme"].to_numpy())
    plt.xlabel("MTU")
    plt.ylabel("Hourly mean absolute FBME [MW]")
    plt.title(f"{strategy}: hourly mean absolute FBME")
    plt.tight_layout()
    plt.savefig(run_plot_dir / "hourly_mean_abs_fbme.png", dpi=PLOT_DPI)
    plt.close()

    # Top MTUs bar plot
    top_mtus = valid_hourly.nlargest(min(TOP_N_MTUS, len(valid_hourly)), "mean_abs_fbme").copy()
    plt.figure(figsize=(12, 6))
    plt.bar(top_mtus["t"].astype(str), top_mtus["mean_abs_fbme"].to_numpy())
    plt.xlabel("MTU")
    plt.ylabel("Hourly mean absolute FBME [MW]")
    plt.title(f"{strategy}: top MTUs by mean absolute FBME")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(run_plot_dir / "top_mtus_by_mean_abs_fbme.png", dpi=PLOT_DPI)
    plt.close()

    # Scatter: mean FBME vs max loading
    if "max_line_loading_d1" in valid_hourly.columns:
        tmp = valid_hourly[["mean_abs_fbme", "max_line_loading_d1"]].dropna()
        if len(tmp) >= 3:
            plt.figure(figsize=(7, 5))
            plt.scatter(
                tmp["max_line_loading_d1"].to_numpy(),
                tmp["mean_abs_fbme"].to_numpy(),
                alpha=0.5,
            )
            plt.xlabel("Max D-1 line loading")
            plt.ylabel("Hourly mean absolute FBME [MW]")
            plt.title(f"{strategy}: FBME vs max D-1 line loading")
            plt.tight_layout()
            plt.savefig(run_plot_dir / "scatter_fbme_vs_max_loading.png", dpi=PLOT_DPI)
            plt.close()

    # Top lines frequency
    if not line_df.empty:
        top_lines = line_df.nlargest(min(300, len(line_df)), "abs_fbme").copy()
        freq = top_lines["line"].value_counts().head(20)

        plt.figure(figsize=(10, 6))
        plt.bar(freq.index.astype(str), freq.to_numpy())
        plt.xlabel("Line")
        plt.ylabel("Count in top abs FBME rows")
        plt.title(f"{strategy}: lines most often appearing in top FBME events")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(run_plot_dir / "top_line_frequency.png", dpi=PLOT_DPI)
        plt.close()


def print_run_summary(strategy: str, hourly_df: pd.DataFrame, line_df: pd.DataFrame):
    valid_hourly = hourly_df[hourly_df["mean_abs_fbme"].notna()].copy()

    print("\n" + "=" * 100)
    print(f"FBME DIAGNOSTICS: {strategy}")
    print("=" * 100)

    print(f"n_mtus_total              {len(hourly_df)}")
    print(f"n_mtus_valid_fbme         {valid_hourly['mean_abs_fbme'].notna().sum()}")
    print(f"mean_hourly_mean_abs_fbme {valid_hourly['mean_abs_fbme'].mean():.6f}")
    print(f"max_hourly_mean_abs_fbme  {valid_hourly['mean_abs_fbme'].max():.6f}")
    print(f"max_line_abs_fbme         {line_df['abs_fbme'].max():.6f}" if not line_df.empty else "max_line_abs_fbme         nan")

    print("\nInvalid-hour reasons:")
    print(hourly_df["reason"].fillna("").replace("", "ok").value_counts().to_string())

    print("\nTop 10 MTUs by hourly mean_abs_fbme:")
    cols = ["t", "mean_abs_fbme", "max_abs_fbme", "n_matched_lines", "sum_abs_np_diff", "renewable_share", "max_line_loading_d1"]
    cols = [c for c in cols if c in valid_hourly.columns]
    print(valid_hourly.nlargest(10, "mean_abs_fbme")[cols].to_string(index=False))

###############################################################################
# MAIN
###############################################################################

def run_diagnostics_for_strategy(strategy: str, run_dir: Path):
    obj = ensure_numeric_time_index(read_df(run_dir, "objectives.parquet"))
    line_f_d1 = ensure_numeric_time_index(read_df(run_dir, "d1_cgm/line_f.parquet"))
    line_f_d2 = ensure_numeric_time_index(read_df(run_dir, "d2/line_f.parquet"))
    np_d1 = ensure_numeric_time_index(read_df(run_dir, "d1_cgm/np.parquet"))
    np_d2 = ensure_numeric_time_index(read_df(run_dir, "d2/np.parquet"))
    ptdf_long = read_df(run_dir, "fb/ptdf_z_cnec_long.parquet")
    ram_long = read_df(run_dir, "fb/ram_long.parquet")
    line_cap_margin = read_line_cap_margin(run_dir)

    hourly_df, line_df = compute_hourly_and_line_fbme(
        line_f_d1=line_f_d1,
        line_f_d2=line_f_d2,
        np_d1=np_d1,
        np_d2=np_d2,
        ptdf_long=ptdf_long,
        ram_long=ram_long,
        line_cap_margin=line_cap_margin,
    )

    hourly_df = attach_hourly_context(
        hourly_df=hourly_df,
        obj=obj,
        np_d1=np_d1,
        np_d2=np_d2,
    )

    run_out_dir = OUTPUT_DIR / strategy
    run_out_dir.mkdir(parents=True, exist_ok=True)

    hourly_df.to_csv(run_out_dir / "hourly_fbme_diagnostics.csv", index=False)
    hourly_df.to_parquet(run_out_dir / "hourly_fbme_diagnostics.parquet", index=False)

    line_df.to_csv(run_out_dir / "line_fbme_diagnostics.csv", index=False)
    line_df.to_parquet(run_out_dir / "line_fbme_diagnostics.parquet", index=False)

    save_top_mtu_line_tables(strategy, hourly_df, line_df)
    save_top_mtu_report(strategy, hourly_df, line_df)
    save_summary_plots(strategy, hourly_df, line_df)
    print_run_summary(strategy, hourly_df, line_df)

    return hourly_df, line_df


def main():
    all_hourly = []
    all_lines = []

    for strategy, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"Skipping missing folder: {run_dir}")
            continue

        hourly_df, line_df = run_diagnostics_for_strategy(strategy, run_dir)

        hourly_df = hourly_df.copy()
        hourly_df["strategy"] = strategy
        line_df = line_df.copy()
        line_df["strategy"] = strategy

        all_hourly.append(hourly_df)
        all_lines.append(line_df)

    if not all_hourly:
        raise RuntimeError("No valid runs were analyzed.")

    all_hourly_df = pd.concat(all_hourly, ignore_index=True)
    all_lines_df = pd.concat(all_lines, ignore_index=True)

    all_hourly_df.to_csv(OUTPUT_DIR / "all_runs_hourly_fbme_diagnostics.csv", index=False)
    all_hourly_df.to_parquet(OUTPUT_DIR / "all_runs_hourly_fbme_diagnostics.parquet", index=False)

    all_lines_df.to_csv(OUTPUT_DIR / "all_runs_line_fbme_diagnostics.csv", index=False)
    all_lines_df.to_parquet(OUTPUT_DIR / "all_runs_line_fbme_diagnostics.parquet", index=False)

    print(f"\nSaved all FBME diagnostics to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()