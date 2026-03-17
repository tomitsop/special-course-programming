# compute_overloads_and_severity.py
# TOTAL overload events + overload severity distribution (no per-timestamp outputs)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from input_data_base_functions import *  # provides df_branch, get_line_cap(l)

# ----------------------------
# Config
# ----------------------------
Number_of_hours = 4320
pipeline_results_path = f"ML_results/{Number_of_hours}_hours/pipeline_results"
flow_csv = os.path.join(pipeline_results_path, "d1_cgm_line_flows.csv")

# Optional: margin (FRM). Set to 0.0 if not needed
frm = 0.0

# Whether to make plots (distribution of MW above limit)
make_plots = True

# ----------------------------
# 1) Load flows (T × L)
# ----------------------------
df_flow = pd.read_csv(flow_csv, index_col=0)

# Unify column types to match BranchID
try:
    df_flow.columns = df_flow.columns.astype(int)
    branch_id_type = "int"
except Exception:
    df_flow.columns = df_flow.columns.astype(str)
    branch_id_type = "str"

# ----------------------------
# 2) Build capacity vector aligned to flow columns
# ----------------------------
cap_map_raw = df_branch.set_index("BranchID")["Pmax"].to_dict()

if branch_id_type == "int":
    cap_map = {int(k): float(v) for k, v in cap_map_raw.items()}
else:
    cap_map = {str(k): float(v) for k, v in cap_map_raw.items()}

missing_caps = [l for l in df_flow.columns if l not in cap_map]
if missing_caps:
    raise ValueError(
        f"Missing capacities for {len(missing_caps)} lines. "
        f"Example: {missing_caps[:10]}"
    )

line_caps = pd.Series({l: cap_map[l] for l in df_flow.columns}, index=df_flow.columns)

# Apply FRM if desired
effective_caps = line_caps * (1.0 - float(frm))

# ----------------------------
# 3) Compute overloads + severity
# ----------------------------
abs_flow = df_flow.abs()

# Boolean overload matrix: True if |flow| > capacity
overload_bool = abs_flow.gt(effective_caps, axis=1)

# Severity in MW above limit (T × L), 0 if not overloaded
severity = (abs_flow.sub(effective_caps, axis=1)).clip(lower=0)

# Vector of all overload magnitudes across all lines/timestamps
sev_vec = severity.to_numpy().ravel()
sev_vec = sev_vec[sev_vec > 0]

# ----------------------------
# 4) Totals + useful summary stats
# ----------------------------
total_overloads = int(overload_bool.values.sum())

total_severity_mw = float(sev_vec.sum()) if sev_vec.size else 0.0
max_severity_mw   = float(sev_vec.max()) if sev_vec.size else 0.0
p95 = float(np.percentile(sev_vec, 95)) if sev_vec.size else 0.0
p99 = float(np.percentile(sev_vec, 99)) if sev_vec.size else 0.0

print("\n========== OVERLOAD SUMMARY ==========")
print("Flow shape (T × L):", df_flow.shape)
print("TOTAL overload events:", total_overloads)
# print("TOTAL severity (sum MW above limit):", total_severity_mw)
print("MAX severity (MW above limit):", max_severity_mw)
# print("95th percentile severity (MW):", p95)
# print("99th percentile severity (MW):", p99)
print("=====================================")

# ----------------------------
# 5) Save (no per-timestamp files)
# ----------------------------
# (Optional) per-line totals (count)
overloads_per_line = overload_bool.sum(axis=0).sort_values(ascending=False)
overloads_per_line.to_csv(
    os.path.join(pipeline_results_path, "overloads_per_line_total.csv"),
    header=["n_overloads"]
)

# (Optional) per-line severity (sum MW above limit)
severity_per_line = severity.sum(axis=0).sort_values(ascending=False)
severity_per_line.to_csv(
    os.path.join(pipeline_results_path, "overload_severity_per_line_total.csv"),
    header=["severity_MW"]
)

# (Optional) save the overload magnitudes vector (for later comparisons)
pd.Series(sev_vec, name="overload_MW").to_csv(
    os.path.join(pipeline_results_path, "overload_severity_values_all.csv"),
    index=False
)

print("\nSaved:")
print(" - overloads_per_line_total.csv")
print(" - overload_severity_per_line_total.csv")
print(" - overload_severity_values_all.csv")

# ----------------------------
# 6) Plots: distribution over ALL lines/timestamps
# ----------------------------
if make_plots:
    if sev_vec.size == 0:
        print("\nNo overloads found -> nothing to plot.")
    else:
        # Histogram of overload magnitude
        plt.figure(figsize=(7, 4))
        plt.hist(sev_vec, bins=60)
        plt.title("Distribution of overload magnitude (MW above capacity)")
        plt.xlabel("MW above limit")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # ECDF (nice for comparing scenarios)
        x = np.sort(sev_vec)
        y = np.arange(1, x.size + 1) / x.size
        plt.figure(figsize=(7, 4))
        plt.plot(x, y)
        plt.title("ECDF of overload magnitude (MW above capacity)")
        plt.xlabel("MW above limit")
        plt.ylabel("Fraction ≤ x")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
