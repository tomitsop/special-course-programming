import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# folder shown in your screenshot
base_dir = "pipeline_results_absfbme_dirgrad_slackfiltered"

csv_files = [
    "abs_median_dirpd_zone_1_slack0.csv", 
    "abs_median_dirpd_zone_2_slack0.csv",
    "abs_median_dirpd_zone_3_slack0.csv",
]

titles = [
    "CGMA error in Zone 1 NP",
    "CGMA error in Zone 2 NP",
    "CGMA error in Zone 3 NP",
]

metric_col = "abs_median_dir_dAbsFBME_dNPpred"  # absolute PD
ylabel = "Median Partial Derivatives [MW/MW]"
xlabel = "CNE"

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharey=True)

for ax, fname, title in zip(axes, csv_files, titles):
    df = pd.read_csv(os.path.join(base_dir, fname))

    # sort by importance
    vals = df[metric_col].values
    order = np.argsort(-vals)
    vals_sorted = vals[order]

    # color gradient by rank
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(vals_sorted)))

    ax.bar(np.arange(len(vals_sorted)), vals_sorted, color=colors, width=0.9)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

axes[1].set_ylabel(ylabel, fontsize=13)
axes[2].set_xlabel(xlabel, fontsize=13)
axes[0].set_ylim(bottom=0)

fig.suptitle("GSK: Flat", fontsize=18, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    os.path.join(base_dir, "median_partial_derivatives_subplots.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
