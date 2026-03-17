# # run_fbme_pipeline.py

# import os
# import math
# import numpy as np
# import pandas as pd
# import torch

# from input_data_base_functions import *  # all sets, mappings, PTDF_Z_CNEC, etc.

# from Differentiable_D_2_CGM import build_d2_cgm_layer
# from Differentiable_D_1_MC  import build_d1_mc_layer
# from Differentiable_D_1_CGM import build_d1_cgm_layer

# # If you want to use the NN forward pass instead of predictions_NP.csv:
# from Neural_Network_NP import NeuralNet  # same architecture as training

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


# #########################################
# #  1. Helper: demand & renewable vectors
# #########################################

# def build_dem_renew_t(t: int):
#     dem_np   = np.array([get_dem(t, n)   for n in N], dtype=np.float32)
#     renew_np = np.array([get_renew(t, n) for n in N], dtype=np.float32)
#     dem_t   = torch.from_numpy(dem_np).to(device)
#     renew_t = torch.from_numpy(renew_np).to(device)
#     return dem_t, renew_t


# #########################################
# #  2. Compute RAM from D-2 flows on CNEC
# #########################################

# # CNEC indices in full line index set
# CNEC_indices = [L_idx[l] for l in CNEC]
# CNEC_idx_t = torch.tensor(CNEC_indices, dtype=torch.long, device=device)

# # Line capacities on CNEC lines
# line_cap_cnec = torch.tensor(
#     [line_cap[i] for i in CNEC_indices],
#     dtype=torch.float32,
#     device=device,
# )

# frm_t = torch.tensor(frm, dtype=torch.float32, device=device)

# def compute_ram_from_d2_torch(line_f_d2_cnec: torch.Tensor):
#     base = line_cap_cnec * (1.0 - frm_t)
#     RAM_pos = base - line_f_d2_cnec
#     RAM_neg = -base - line_f_d2_cnec
#     return RAM_pos, RAM_neg


# #########################################
# #  3. Load features X and predictions_NP
# #########################################

# Number_of_days = 30 * 6
# Number_of_hours = 24 * Number_of_days

# ML_results_path = f"ML_results/{Number_of_hours}_hours"

# # Features (if you want to run the NN forward again)
# X = pd.read_csv("data/X.csv", index_col=0)
# if X.index.min() == 0:
#     X.index = X.index + 1  # make it 1..T

# # NP predictions from your existing NN pipeline (original scale)
# predictions_np = pd.read_csv(
#     f"{ML_results_path}/predictions_NP.csv",
#     index_col=0
# )

# # Ensure indices are consistent (use intersection)
# common_index = X.index.intersection(predictions_np.index)
# X = X.loc[common_index]
# predictions_np = predictions_np.loc[common_index]

# time_index = common_index.values  # numpy array of time steps in your process

# # >>> Only use first 5 MTUs <<<
# time_index_subset = time_index[:5]
# print("Running pipeline for MTUs:", time_index_subset)


# #########################################
# #  4. Build differentiable layers
# #########################################

# cost_curt = math.ceil(find_maximum_mc())
# cost_curt_mc = 0
# max_ntc = 1000.0
# export_eps = 1e-7

# d2_layer = build_d2_cgm_layer(cost_curt=cost_curt)
# d1_mc_layer = build_d1_mc_layer(
#     cost_curt_mc=cost_curt_mc,
#     max_ntc=max_ntc,
#     export_eps=export_eps,
# )
# d1_cgm_layer = build_d1_cgm_layer(
#     cost_curt=cost_curt,
#     max_ntc=max_ntc,
# )


# #########################################
# #  5. (Optional) Load trained NN model  #
# #     if you want to re-run it instead  #
# #     of using predictions_NP.csv       #
# #########################################

# use_nn_forward = False  # set True if you want to recompute NPs via NN

# if use_nn_forward:
#     # You must ensure these scalers / preprocessing are consistent with training
#     # For now we'll assume X is already scaled appropriately
#     input_size = X.shape[1]
#     output_size = len(Z_FBMC)

#     model = NeuralNet(input_size, output_size).to(device)
#     # Load weights if you saved them, e.g.:
#     # model.load_state_dict(torch.load(f"{ML_results_path}/model_NP.pt", map_location=device))
#     model.eval()


# #########################################
# #  6. Run pipeline over selected MTUs  #
# #########################################

# # Store results
# results_d2_np    = []
# results_d1_np    = []
# results_cgm_np   = []

# results_d2_flow  = []
# results_cgm_flow = []

# results_d2_gen   = []
# results_d2_curt  = []

# results_d1_gen   = []
# results_d1_curt  = []

# results_cgm_gen  = []
# results_cgm_curt = []

# for t in time_index_subset:
#     print(f"Processing time step t = {t}")

#     # 1) Build dem and renew for this hour
#     dem_t, renew_t = build_dem_renew_t(int(t))

#     # 2) NP input for D-2 CGM
#     if use_nn_forward:
#         # run NN on features X_t
#         x_t_np = X.loc[t].values.astype(np.float32)
#         x_t = torch.from_numpy(x_t_np).to(device)
#         with torch.no_grad():
#             np_pred_t = model(x_t.unsqueeze(0)).squeeze(0)  # (|Z_FBMC|,)
#     else:
#         # use predictions_NP.csv directly
#         np_pred_np = predictions_np.loc[t, Z_FBMC].values.astype(np.float32)
#         np_pred_t = torch.from_numpy(np_pred_np).to(device)

#     # 3) D-2 CGM
#     with torch.no_grad():
#         GEN_d2, CURT_d2, NOD_INJ_d2, LINE_F_d2, NP_d2, EXPORT_d2, \
#         DELTA_d2, slack_pos_d2, slack_neg_d2 = d2_layer(
#             np_pred_t,   # pred_np
#             dem_t,       # dem
#             renew_t,     # renew
#         )

#         LINE_F_d2_CNEC = LINE_F_d2[CNEC_idx_t]  # (nCNEC,)

#         # 4) RAM from D-2 flows
#         RAM_pos_t, RAM_neg_t = compute_ram_from_d2_torch(LINE_F_d2_CNEC)

#         # 5) D-1 MC
#         GEN_d1, CURT_d1, NP_d1, EXPORT_d1 = d1_mc_layer(
#             dem_t,       # dem
#             renew_t,     # renew
#             NP_d2,       # np_d2_fb
#             RAM_pos_t,   # ram_pos
#             RAM_neg_t,   # ram_neg
#         )

#         # 6) D-1 CGM
#         GEN_cgm, CURT_cgm, DELTA_cgm, NOD_INJ_cgm, LINE_F_cgm, \
#         NP_cgm, EXPORT_cgm = d1_cgm_layer(
#             dem_t,       # dem
#             renew_t,     # renew
#             GEN_d1,      # gen_sched
#             CURT_d1,     # curt_sched
#         )

#     # 7) Collect results on CPU as numpy
#     # --- NPs ---
#     results_d2_np.append(NP_d2.detach().cpu().numpy())
#     results_d1_np.append(NP_d1.detach().cpu().numpy())
#     results_cgm_np.append(NP_cgm.detach().cpu().numpy())

#     # --- Line flows (D-2 CGM & D-1 CGM) ---
#     results_d2_flow.append(LINE_F_d2.detach().cpu().numpy())
#     results_cgm_flow.append(LINE_F_cgm.detach().cpu().numpy())

#     # --- GEN / CURT from each model ---
#     results_d2_gen.append(GEN_d2.detach().cpu().numpy())
#     results_d2_curt.append(CURT_d2.detach().cpu().numpy())

#     results_d1_gen.append(GEN_d1.detach().cpu().numpy())
#     results_d1_curt.append(CURT_d1.detach().cpu().numpy())

#     results_cgm_gen.append(GEN_cgm.detach().cpu().numpy())
#     results_cgm_curt.append(CURT_cgm.detach().cpu().numpy())


# #########################################
# #  7. Save results as DataFrames       #
# #########################################

# index = time_index_subset

# # NPs
# df_d2_np    = pd.DataFrame(np.vstack(results_d2_np),    index=index, columns=Z_FBMC)
# df_d1_np    = pd.DataFrame(np.vstack(results_d1_np),    index=index, columns=Z_FBMC)
# df_cgm_np   = pd.DataFrame(np.vstack(results_cgm_np),   index=index, columns=Z_FBMC)

# # Line flows (all lines)
# df_d2_flow  = pd.DataFrame(np.vstack(results_d2_flow),  index=index, columns=L)
# df_cgm_flow = pd.DataFrame(np.vstack(results_cgm_flow), index=index, columns=L)

# # GEN schedules
# df_d2_gen   = pd.DataFrame(np.vstack(results_d2_gen),   index=index, columns=P)
# df_d1_gen   = pd.DataFrame(np.vstack(results_d1_gen),   index=index, columns=P)
# df_cgm_gen  = pd.DataFrame(np.vstack(results_cgm_gen),  index=index, columns=P)

# # CURT schedules
# df_d2_curt  = pd.DataFrame(np.vstack(results_d2_curt),  index=index, columns=N)
# df_d1_curt  = pd.DataFrame(np.vstack(results_d1_curt),  index=index, columns=N)
# df_cgm_curt = pd.DataFrame(np.vstack(results_cgm_curt), index=index, columns=N)

# # Save
# pipeline_results_path = f"{ML_results_path}/pipeline_results"
# os.makedirs(pipeline_results_path, exist_ok=True)

# # NPs
# df_d2_np.to_csv(f"{pipeline_results_path}/d2_np.csv")
# df_d1_np.to_csv(f"{pipeline_results_path}/d1_mc_np.csv")
# df_cgm_np.to_csv(f"{pipeline_results_path}/d1_cgm_np.csv")

# # Flows
# df_d2_flow.to_csv(f"{pipeline_results_path}/d2_line_flows.csv")
# df_cgm_flow.to_csv(f"{pipeline_results_path}/d1_cgm_line_flows.csv")

# # GEN / CURT
# df_d2_gen.to_csv(f"{pipeline_results_path}/d2_gen.csv")
# df_d1_gen.to_csv(f"{pipeline_results_path}/d1_mc_gen.csv")
# df_cgm_gen.to_csv(f"{pipeline_results_path}/d1_cgm_gen.csv")

# df_d2_curt.to_csv(f"{pipeline_results_path}/d2_curt.csv")
# df_d1_curt.to_csv(f"{pipeline_results_path}/d1_mc_curt.csv")
# df_cgm_curt.to_csv(f"{pipeline_results_path}/d1_cgm_curt.csv")

# print("Pipeline finished. Results saved in:", pipeline_results_path)


import pandas as pd

df = pd.read_parquet("data/Y_NP_FBMC.parquet")
print(df[2160:].head(20))