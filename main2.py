# run_fbme_pipeline.py

import os
import math
import numpy as np
import pandas as pd
import torch
import cvxpy as cp


from input_data_base_functions import *  # all sets, mappings, PTDF_Z_CNEC, etc.

from Differentiable_D_2_CGM import build_d2_cgm_layer
from Differentiable_D_1_MC  import build_d1_mc_layer
from Differentiable_D_1_CGM import build_d1_cgm_layer

# If you want to use the NN forward pass instead of predictions_NP.csv:
# from Neural_Network_NP import NeuralNet  # same architecture as training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CVXPY installed solvers:", cp.installed_solvers())

# Use DIFFCP-style arguments:
solver_args = {
    "solve_method": "SCS",   
    "eps": 1e-8,
    "max_iters": 200000,
    "verbose": False,
}

#########################################
#  1. Helper: demand & renewable vectors
#########################################

def build_dem_renew_t(t: int):
    # Use float64 + torch.double for cvxpylayers stability
    dem_np   = np.array([get_dem(t, n)   for n in N], dtype=np.float64)
    renew_np = np.array([get_renew(t, n) for n in N], dtype=np.float64)
    dem_t   = torch.from_numpy(dem_np).to(device=device, dtype=torch.double)
    renew_t = torch.from_numpy(renew_np).to(device=device, dtype=torch.double)
    return dem_t, renew_t

#########################################
#  2. Compute RAM from D-2 flows on CNEC
#########################################

# CNEC indices in full line index set
CNEC_indices = [L_idx[l] for l in CNEC]
CNEC_idx_t = torch.tensor(CNEC_indices, dtype=torch.long, device=device)

# Line capacities on CNEC lines (double)
line_cap_cnec = torch.tensor(
    [line_cap[i] for i in CNEC_indices],
    dtype=torch.double,
    device=device,
)

frm_t = torch.tensor(float(frm), dtype=torch.double, device=device)

def compute_ram_from_d2_torch(line_f_d2_cnec: torch.Tensor):
    # ensure double
    line_f_d2_cnec = line_f_d2_cnec.to(dtype=torch.double)
    base = line_cap_cnec * (1.0 - frm_t)
    RAM_pos = base - line_f_d2_cnec
    RAM_neg = -base - line_f_d2_cnec
    return RAM_pos, RAM_neg

#########################################
#  3. Load features X and predictions_NP
#########################################

Number_of_days = 30 * 6
Number_of_hours = 24 * Number_of_days

ML_results_path = f"ML_results/{Number_of_hours}_hours"

# Features (if you want to run the NN forward again)
X = pd.read_csv("data/X.csv", index_col=0)
if X.index.min() == 0:
    X.index = X.index + 1  # make it 1..T

# NP predictions
predictions_np = pd.read_csv(
    f"{ML_results_path}/predictions_NP.csv",
    index_col=0
)

# Ensure indices are consistent (use intersection)
common_index = X.index.intersection(predictions_np.index)
print(common_index)
X = X.loc[common_index]
predictions_np = predictions_np.loc[common_index]

time_index = common_index.values
time_index_subset = time_index


#########################################
#  4. Build differentiable layers
#########################################

cost_curt = math.ceil(find_maximum_mc())
cost_curt_mc = 0
max_ntc = 1000.0
export_eps = 1e-7

d2_layer = build_d2_cgm_layer(cost_curt=cost_curt)
d1_mc_layer = build_d1_mc_layer(
    cost_curt_mc=cost_curt_mc,
    max_ntc=max_ntc,
    export_eps=export_eps,
)
d1_cgm_layer = build_d1_cgm_layer(
    cost_curt=cost_curt,
    max_ntc=max_ntc,
)

#########################################
#  5. (Optional) Load trained NN model
#########################################

# use_nn_forward = False

# if use_nn_forward:
#     input_size = X.shape[1]
#     output_size = len(Z_FBMC)
#     model = NeuralNet(input_size, output_size).to(device)
#     # model.load_state_dict(torch.load(f"{ML_results_path}/model_NP.pt", map_location=device))
#     model.eval()

#########################################
#  6. Run pipeline over selected MTUs
#########################################

results_d2_np    = []
results_d1_np    = []
results_cgm_np   = []

results_d2_flow  = []
results_cgm_flow = []

results_d2_gen   = []
results_d2_curt  = []

results_d1_gen   = []
results_d1_curt  = []

results_cgm_gen  = []
results_cgm_curt = []

successful_ts    = []
infeasible_ts    = []


# --- Store inputs actually used by the layers ---
results_dem   = []
results_renew = []

# Optional: store quick checks
results_dem_sum   = []
results_renew_sum = []
results_samples   = []  # e.g., a few buses to spot-check

for t in time_index_subset:
    print(f"\nProcessing time step t = {t}")

    dem_t, renew_t = build_dem_renew_t(int(t))

    # # NP input for D-2 CGM
    # if use_nn_forward:
    #     x_t_np = X.loc[t].values.astype(np.float64)
    #     x_t = torch.from_numpy(x_t_np).to(device=device, dtype=torch.double)
    #     with torch.no_grad():
    #         np_pred_t = model(x_t.unsqueeze(0)).squeeze(0).to(dtype=torch.double)
    # else:
    np_pred_np = predictions_np.loc[t, Z_FBMC].values.astype(np.float64)
    np_pred_t = torch.from_numpy(np_pred_np).to(device=device, dtype=torch.double)

    try:
        with torch.no_grad():
            # 3) D-2 CGM
            GEN_d2, CURT_d2, NOD_INJ_d2, LINE_F_d2, NP_d2, EXPORT_d2, \
            DELTA_d2, slack_pos_d2, slack_neg_d2 = d2_layer(
                np_pred_t,
                dem_t,
                renew_t,
                solver_args=solver_args,
            )

            LINE_F_d2_CNEC = LINE_F_d2[CNEC_idx_t].to(dtype=torch.double)

            # 4) RAM from D-2 flows
            RAM_pos_t, RAM_neg_t = compute_ram_from_d2_torch(LINE_F_d2_CNEC)

            # 5) D-1 MC
            GEN_d1, CURT_d1, NP_d1, EXPORT_d1 = d1_mc_layer(
                dem_t,
                renew_t,
                NP_d2.to(dtype=torch.double),
                RAM_pos_t,
                RAM_neg_t,
                solver_args=solver_args,
            )

            # 6) D-1 CGM
            GEN_cgm, CURT_cgm, DELTA_cgm, NOD_INJ_cgm, LINE_F_cgm, \
            NP_cgm, EXPORT_cgm = d1_cgm_layer(
                dem_t,
                renew_t,
                GEN_d1.to(dtype=torch.double),
                CURT_d1.to(dtype=torch.double),
                solver_args=solver_args,
            )

    except Exception as e:
        print(f"  -> Solver error / infeasible at t = {t}: {e}")
        infeasible_ts.append(t)
        continue

    successful_ts.append(t)

    results_d2_np.append(NP_d2.detach().cpu().numpy())
    results_d1_np.append(NP_d1.detach().cpu().numpy())
    results_cgm_np.append(NP_cgm.detach().cpu().numpy())

    results_d2_flow.append(LINE_F_d2.detach().cpu().numpy())
    results_cgm_flow.append(LINE_F_cgm.detach().cpu().numpy())

    results_d2_gen.append(GEN_d2.detach().cpu().numpy())
    results_d2_curt.append(CURT_d2.detach().cpu().numpy())

    results_d1_gen.append(GEN_d1.detach().cpu().numpy())
    results_d1_curt.append(CURT_d1.detach().cpu().numpy())

    results_cgm_gen.append(GEN_cgm.detach().cpu().numpy())
    results_cgm_curt.append(CURT_cgm.detach().cpu().numpy())

#########################################
#  7. Save results as DataFrames
#########################################

if len(successful_ts) == 0:
    print("\nNo feasible MTUs were solved successfully. Nothing to save.")
else:
    index = np.array(successful_ts)

    df_d2_np    = pd.DataFrame(np.vstack(results_d2_np),    index=index, columns=Z_FBMC)
    df_d1_np    = pd.DataFrame(np.vstack(results_d1_np),    index=index, columns=Z_FBMC)
    df_cgm_np   = pd.DataFrame(np.vstack(results_cgm_np),   index=index, columns=Z_FBMC)

    df_d2_flow  = pd.DataFrame(np.vstack(results_d2_flow),  index=index, columns=L)
    df_cgm_flow = pd.DataFrame(np.vstack(results_cgm_flow), index=index, columns=L)

    df_d2_gen   = pd.DataFrame(np.vstack(results_d2_gen),   index=index, columns=P)
    df_d2_gen = df_d2_gen.reset_index().rename(columns={"index": "TimeStep"})
    df_d1_gen   = pd.DataFrame(np.vstack(results_d1_gen),   index=index, columns=P)
    df_d1_gen = df_d1_gen.reset_index().rename(columns={"index": "TimeStep"})
    df_cgm_gen  = pd.DataFrame(np.vstack(results_cgm_gen),  index=index, columns=P)
    df_cgm_gen = df_cgm_gen.reset_index().rename(columns={"index": "TimeStep"})


    df_d2_curt  = pd.DataFrame(np.vstack(results_d2_curt),  index=index, columns=N)
    df_d1_curt  = pd.DataFrame(np.vstack(results_d1_curt),  index=index, columns=N)
    df_cgm_curt = pd.DataFrame(np.vstack(results_cgm_curt), index=index, columns=N)

    pipeline_results_path = f"{ML_results_path}/pipeline_results"
    os.makedirs(pipeline_results_path, exist_ok=True)

    df_d2_np.to_csv(f"{pipeline_results_path}/d2_np.csv")
    df_d1_np.to_csv(f"{pipeline_results_path}/d1_mc_np.csv")
    df_cgm_np.to_csv(f"{pipeline_results_path}/d1_cgm_np.csv")

    df_d2_flow.to_csv(f"{pipeline_results_path}/d2_line_flows.csv")
    df_cgm_flow.to_csv(f"{pipeline_results_path}/d1_cgm_line_flows.csv")

    df_d2_gen.to_csv(f"{pipeline_results_path}/d2_gen.csv")
    df_d1_gen.to_csv(f"{pipeline_results_path}/d1_mc_gen.csv")
    df_cgm_gen.to_csv(f"{pipeline_results_path}/d1_cgm_gen.csv")

    df_d2_curt.to_csv(f"{pipeline_results_path}/d2_curt.csv")
    df_d1_curt.to_csv(f"{pipeline_results_path}/d1_mc_curt.csv")
    df_cgm_curt.to_csv(f"{pipeline_results_path}/d1_cgm_curt.csv")

    print("\nPipeline finished.")

print("\nSummary:")
print(f"  Total MTUs attempted: {len(time_index_subset)}")
print(f"  Feasible MTUs:        {len(successful_ts)}")
print(f"  Infeasible MTUs:      {len(infeasible_ts)}")

if len(infeasible_ts) > 0:
    print("  Infeasible at time steps:", infeasible_ts)



# difference of forecast with d-1 cgm flow based error install gurobi cvxpy train new NN for inputs ID CGMs DIFFERENT FORECAST MODELS