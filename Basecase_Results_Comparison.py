import numpy as np
import pandas as pd
from pathlib import Path

from input_data_base_functions import (
    L,
    Z_FBMC,
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

OUTPUT_DIR = RESULTS_BASE / "comparison_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# HELPERS
###############################################################################

def read_df(run_dir: Path, rel_path: str) -> pd.DataFrame:
    path = run_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


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


def compute_total_d1_overloads(
    line_f_d1: pd.DataFrame,
    line_cap_margin: pd.Series,
) -> tuple[int, pd.Series]:
    """
    Count overloads where |flow| > line_cap * (1 - frm)
    """
    margin = align_line_cap_margin(line_cap_margin, line_f_d1.columns)
    overloaded = line_f_d1.abs().gt(margin, axis=1)

    hourly_overloads = overloaded.sum(axis=1).astype(int)
    total_overloads = int(hourly_overloads.sum())

    return total_overloads, hourly_overloads


def compute_fbme_d1_metrics(
    line_f_d1: pd.DataFrame,
    line_f_d2: pd.DataFrame,
    np_d1: pd.DataFrame,
    np_d2: pd.DataFrame,
    ptdf_long: pd.DataFrame,
):
    """
    FBME_{t,j} = F^{D1_CGM}_{t,j} - F^{D2_CGM}_{t,j}
                 - sum_z (NP^{D1_CGM}_{t,z} - NP^{D2_CGM}_{t,z}) * PTDF^Z_{t,j,z}

    Matching is done robustly via:
        cnec_idx -> L[cnec_idx] -> line_f columns

    Returns:
      hourly_mean_abs_fbme : Series indexed by t
      global_mean_abs_fbme : scalar over all MTUs and all matched CNECs together
    """

    if "cnec_idx" not in ptdf_long.columns:
        raise ValueError(
            "ptdf_long must contain 'cnec_idx'. "
            "Merge it from ram_long.parquet before calling this function."
        )

    common_t = (
        line_f_d1.index
        .intersection(line_f_d2.index)
        .intersection(np_d1.index)
        .intersection(np_d2.index)
        .intersection(pd.Index(ptdf_long["t"].unique()))
    )

    # robust mapping from cnec_idx to actual line label
    idx_to_line = {int(i): L[i] for i in range(len(L))}

    hourly_out = []
    all_abs_fbme = []

    for t in common_t:
        ptdf_hour = ptdf_long.loc[ptdf_long["t"] == t].copy()

        # drop rows where cnec_idx is missing
        ptdf_hour = ptdf_hour.dropna(subset=["cnec_idx"])

        if ptdf_hour.empty:
            hourly_out.append((t, np.nan))
            continue

        ptdf_hour["cnec_idx"] = ptdf_hour["cnec_idx"].astype(int)

        # pivot by cnec_idx, not by cnec text
        ptdf_t = ptdf_hour.pivot(
            index="cnec_idx",
            columns="zone",
            values="ptdf"
        )

        # ensure zones are aligned
        ptdf_t = ptdf_t.reindex(columns=Z_FBMC)

        # map cnec_idx -> actual line label used in line_f
        ptdf_t.index = ptdf_t.index.map(idx_to_line)

        # remove rows that failed mapping
        ptdf_t = ptdf_t[ptdf_t.index.notna()]

        matched_lines = [
            c for c in ptdf_t.index
            if c in line_f_d1.columns and c in line_f_d2.columns
        ]

        if len(matched_lines) == 0:
            hourly_out.append((t, np.nan))
            continue

        ptdf_t = ptdf_t.loc[matched_lines]

        # optional safety: skip hour if PTDF still contains NaNs
        if ptdf_t.isna().any().any():
            hourly_out.append((t, np.nan))
            continue

        delta_np = (
            np_d1.loc[t, Z_FBMC].to_numpy(dtype=float)
            - np_d2.loc[t, Z_FBMC].to_numpy(dtype=float)
        )

        expected_delta_flow = ptdf_t.to_numpy(dtype=float) @ delta_np

        actual_delta_flow = (
            line_f_d1.loc[t, matched_lines].to_numpy(dtype=float)
            - line_f_d2.loc[t, matched_lines].to_numpy(dtype=float)
        )

        fbme = actual_delta_flow - expected_delta_flow
        abs_fbme = np.abs(fbme)

        hourly_out.append((t, float(abs_fbme.mean())))
        all_abs_fbme.extend(abs_fbme.tolist())

    hourly_mean_abs_fbme = pd.Series(
        data=[v for _, v in hourly_out],
        index=[t for t, _ in hourly_out],
        name="mean_abs_fbme_d1_cgm"
    ).sort_index()

    global_mean_abs_fbme = (
        float(np.mean(all_abs_fbme)) if len(all_abs_fbme) > 0 else np.nan
    )

    return hourly_mean_abs_fbme, global_mean_abs_fbme


###############################################################################
# PER-RUN ANALYSIS
###############################################################################

def analyze_run(strategy_name: str, run_dir: Path) -> tuple[dict, dict]:
    print(f"Analyzing {strategy_name} from {run_dir}")

    obj = read_df(run_dir, "objectives.parquet")
    line_f_d1 = read_df(run_dir, "d1_cgm/line_f.parquet")
    line_f_d2 = read_df(run_dir, "d2/line_f.parquet")
    np_d1 = read_df(run_dir, "d1_cgm/np.parquet")
    np_d2 = read_df(run_dir, "d2/np.parquet")
    line_cap_margin = read_line_cap_margin(run_dir)
    
    ptdf_long = read_df(run_dir, "fb/ptdf_z_cnec_long.parquet")

    if "cnec_idx" not in ptdf_long.columns:
        ram_long = read_df(run_dir, "fb/ram_long.parquet")
        ptdf_long = ptdf_long.merge(
            ram_long[["t", "cnec", "cnec_idx"]].drop_duplicates(),
            on=["t", "cnec"],
            how="left"
        )

    missing_idx = ptdf_long["cnec_idx"].isna().sum()
    if missing_idx > 0:
        print(f"[WARN] {strategy_name}: {missing_idx} PTDF rows are missing cnec_idx.")

    total_overloads_d1, hourly_overloads_d1 = compute_total_d1_overloads(
        line_f_d1,
        line_cap_margin,
    )

    hourly_abs_fbme_d1, global_mean_abs_fbme_d1 = compute_fbme_d1_metrics(
        line_f_d1=line_f_d1,
        line_f_d2=line_f_d2,
        np_d1=np_d1,
        np_d2=np_d2,
        ptdf_long=ptdf_long,
    )

    summary = {
        "strategy": strategy_name,
        "n_mtus": int(len(obj)),
        "d1_cgm_objective_mean": float(obj["d1_cgm"].mean()),
        "d1_cgm_objective_sum": float(obj["d1_cgm"].sum()),
        "d0_objective_mean": float(obj["d0"].mean()),
        "d0_objective_sum": float(obj["d0"].sum()),
        "total_d1_overloads": int(total_overloads_d1),
        # "mean_hourly_d1_overloads": float(hourly_overloads_d1.mean()),
        "mean_abs_fbme_d1_cgm": global_mean_abs_fbme_d1,
    }

    detailed = {
        "objectives": obj[["d1_cgm", "d0"]].rename(
            columns={
                "d1_cgm": "d1_cgm_objective",
                "d0": "d0_objective",
            }
        ),
        "hourly_overloads_d1": hourly_overloads_d1.rename("hourly_d1_overloads"),
        "hourly_abs_fbme_d1": hourly_abs_fbme_d1.rename("mean_abs_fbme_d1_cgm"),
    }

    return summary, detailed


###############################################################################
# MAIN
###############################################################################

def main():
    all_summary = []
    all_objectives = []
    all_overloads = []
    all_fbme_d1 = []

    for strategy, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"Skipping missing folder: {run_dir}")
            continue

        summary, detailed = analyze_run(strategy, run_dir)
        all_summary.append(summary)

        obj_df = detailed["objectives"].copy()
        obj_df["strategy"] = strategy
        obj_df["t"] = obj_df.index
        all_objectives.append(obj_df.reset_index(drop=True))

        ov_df = detailed["hourly_overloads_d1"].to_frame()
        ov_df["strategy"] = strategy
        ov_df["t"] = ov_df.index
        all_overloads.append(ov_df.reset_index(drop=True))

        fbme_d1_df = detailed["hourly_abs_fbme_d1"].to_frame()
        fbme_d1_df["strategy"] = strategy
        fbme_d1_df["t"] = fbme_d1_df.index
        all_fbme_d1.append(fbme_d1_df.reset_index(drop=True))

    if not all_summary:
        raise RuntimeError("No valid run folders found.")

    summary_df = pd.DataFrame(all_summary).sort_values("strategy")
    objectives_df = pd.concat(all_objectives, ignore_index=True)
    overloads_df = pd.concat(all_overloads, ignore_index=True)
    fbme_d1_df = pd.concat(all_fbme_d1, ignore_index=True)

    print("\n=== SUMMARY ===")
    print(summary_df)

    summary_df.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    summary_df.to_parquet(OUTPUT_DIR / "summary_metrics.parquet", index=False)

    objectives_df.to_csv(OUTPUT_DIR / "hourly_objectives.csv", index=False)
    objectives_df.to_parquet(OUTPUT_DIR / "hourly_objectives.parquet", index=False)

    overloads_df.to_csv(OUTPUT_DIR / "hourly_d1_overloads.csv", index=False)
    overloads_df.to_parquet(OUTPUT_DIR / "hourly_d1_overloads.parquet", index=False)

    fbme_d1_df.to_csv(OUTPUT_DIR / "hourly_mean_abs_fbme_d1_cgm.csv", index=False)
    fbme_d1_df.to_parquet(OUTPUT_DIR / "hourly_mean_abs_fbme_d1_cgm.parquet", index=False)

    print(f"\nSaved comparison outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
