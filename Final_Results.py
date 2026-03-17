import numpy as np
import pandas as pd
import os

from input_data_base_functions import *  # PTDF_Z_CNEC, CNEC, Z_FBMC, etc.

pipeline_results_path = "ML_results/4320_hours/pipeline_results"

# ----------------------------
# 1) Read CSVs
# ----------------------------
df_d2_flow  = pd.read_csv(f"{pipeline_results_path}/d2_line_flows.csv", index_col=0)
df_cgm_flow = pd.read_csv(f"{pipeline_results_path}/d1_cgm_line_flows.csv", index_col=0)

df_d2_np  = pd.read_csv(f"{pipeline_results_path}/d2_np.csv", index_col=0)
df_cgm_np = pd.read_csv(f"{pipeline_results_path}/d1_cgm_np.csv", index_col=0)

# Ensure indices align as strings (optional, but helps if some read as int vs str)
df_d2_flow.index  = df_d2_flow.index.astype(str)
df_cgm_flow.index = df_cgm_flow.index.astype(str)
df_d2_np.index    = df_d2_np.index.astype(str)
df_cgm_np.index   = df_cgm_np.index.astype(str)

# ----------------------------
# 2) Ensure PTDF is a DataFrame
# ----------------------------
if not isinstance(PTDF_Z_CNEC, pd.DataFrame):
    PTDF_Z_CNEC = pd.DataFrame(PTDF_Z_CNEC, index=CNEC, columns=Z_FBMC)

# CRITICAL: force consistent label types (strings) for matching
PTDF_Z_CNEC.index = PTDF_Z_CNEC.index.astype(str)
PTDF_Z_CNEC.columns = PTDF_Z_CNEC.columns.astype(str)

df_d2_flow.columns  = df_d2_flow.columns.astype(str)
df_cgm_flow.columns = df_cgm_flow.columns.astype(str)

df_d2_np.columns  = df_d2_np.columns.astype(str)
df_cgm_np.columns = df_cgm_np.columns.astype(str)

# ----------------------------
# 3) Build exact orders from PTDF (best practice)
# ----------------------------
cnec_order = list(PTDF_Z_CNEC.index)      # J CNECs in PTDF order
z_order    = list(PTDF_Z_CNEC.columns)    # Z zones in PTDF order

# ----------------------------
# 4) Filter flows to ONLY CNECs (T × J)
# ----------------------------
missing_cnec_in_flow = set(cnec_order) - set(df_cgm_flow.columns)
if missing_cnec_in_flow:
    raise ValueError(
        f"Some CNEC IDs from PTDF are missing in flow columns. "
        f"Example missing: {list(sorted(missing_cnec_in_flow))[:10]}"
    )

D1_flow_cnec = df_cgm_flow[cnec_order]
D2_flow_cnec = df_d2_flow[cnec_order]

# ----------------------------
# 5) Align NPs to PTDF zone order (T × Z)
# ----------------------------
missing_z_in_np = set(z_order) - set(df_cgm_np.columns)
if missing_z_in_np:
    raise ValueError(
        f"Some zone IDs from PTDF are missing in NP columns. "
        f"Example missing: {list(sorted(missing_z_in_np))[:10]}"
    )

D1_np = df_cgm_np[z_order]
D2_np = df_d2_np[z_order]

# ----------------------------
# 6) Common time index (only timestamps that exist everywhere)
# ----------------------------
common_idx = (
    D1_flow_cnec.index
    .intersection(D2_flow_cnec.index)
    .intersection(D1_np.index)
    .intersection(D2_np.index)
)

if len(common_idx) == 0:
    raise ValueError("No common timestamps found across flows and NPs.")

D1_flow_cnec = D1_flow_cnec.loc[common_idx]
D2_flow_cnec = D2_flow_cnec.loc[common_idx]
D1_np = D1_np.loc[common_idx]
D2_np = D2_np.loc[common_idx]

# ----------------------------
# 7) Compute FBME (T × J)
# FBME = (D1_flow - D2_flow) - (ΔNP @ PTDF.T)
# PTDF is (J × Z) so PTDF.T is (Z × J)
# ----------------------------
PTDF_T = PTDF_Z_CNEC.loc[cnec_order, z_order].values.T  # (Z × J)

fbme = (D1_flow_cnec.values - D2_flow_cnec.values) - ((D1_np.values - D2_np.values) @ PTDF_T)

df_fbme = pd.DataFrame(fbme, index=common_idx, columns=cnec_order)

# ----------------------------
# 8) Save + diagnostics
# ----------------------------
out_path = os.path.join(pipeline_results_path, "fb_market_error.csv")
df_fbme.to_csv(out_path, index=True, index_label="TimeStep")

print("Saved:", out_path)
print("FBME shape:", df_fbme.shape)
print("Mean |FBME|:", df_fbme.abs().to_numpy().mean())
# print("First rows:\n", df_fbme.head())
