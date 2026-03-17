# Basecase_CNEC_Histogram.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_BASE = Path("results")

RUNS = {
    # "dynamic_gen": RESULTS_BASE / "pipeline_run_gurobi_dynamic_gen",
    # "dynamic_headroom": RESULTS_BASE / "pipeline_run_gurobi_dynamic_headroom",
    # "flat_unit": RESULTS_BASE / "pipeline_run_gurobi_flat_unit",
    "pmax_sub": RESULTS_BASE / "pipeline_run_gurobi",
    "scopf_dynamic_gen":RESULTS_BASE / "pipeline_run_gurobi_SCOPF_dynamic_gen",
    "scopf_dynamic_headroom":RESULTS_BASE / "pipeline_run_gurobi_SCOPF_dynamic_headroom",
}

OUTPUT_DIR = RESULTS_BASE / "cnec_histograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_cnec_counts(run_dir: Path):
    path = run_dir / "fb" / "cnec_info.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_parquet(path)

    if "n_cnec" not in df.columns:
        raise ValueError("Column 'n_cnec' not found in cnec_info.parquet")

    return df["n_cnec"]


def plot_histogram(strategy: str, n_cnec: pd.Series):

    plt.figure(figsize=(8, 5))

    plt.hist(
        n_cnec,
        bins=40,
        edgecolor="black"
    )

    plt.xlabel("Number of CNECs per MTU")
    plt.ylabel("Frequency (MTUs)")
    plt.title(f"CNEC Distribution per MTU ({strategy})")

    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{strategy}_cnec_histogram.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved histogram → {out_path}")


def main():

    for strategy, run_dir in RUNS.items():

        if not run_dir.exists():
            print(f"Skipping missing run: {run_dir}")
            continue

        print(f"\nAnalyzing run: {strategy}")

        n_cnec = read_cnec_counts(run_dir)

        print("Summary statistics:")
        print(n_cnec.describe())

        plot_histogram(strategy, n_cnec)


if __name__ == "__main__":
    main()