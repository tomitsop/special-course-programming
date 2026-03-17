###############################
#  D-2 CGM + Preventive SCOPF (N-1 via LODF) — NO pred_np / NO slack variables
###############################

import math
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from input_data_base_functions import *
# Uses: P,N,L,Z_FBMC,Z_not_in_FBMC,
#       N_idx,L_idx,Z_fb_idx,Z_not_idx,
#       plants_at_node_idx,nodes_in_zone_idx,z_to_z,
#       cost_gen,gmax,line_cap,B_matrix,H_matrix,
#       find_maximum_mc, etc.


# -----------------------------
# LODF computation from PTDF (same logic you used before)
# -----------------------------
def compute_lodf_from_ptdf(df_branch, PTDF_full, N_idx, L_idx, L, eps=1e-8):
    """
    Returns:
      LODF: (nL, nL) where LODF[l,k] gives change on line l due to outage of k,
            normalized by pre-outage flow on k.
      bad_k: (nL,) boolean for contingencies with near-zero denom (islanding/invalid)
    """
    nL = len(L)
    lodf = np.zeros((nL, nL), dtype=float)

    from_bus = np.zeros(nL, dtype=int)
    to_bus = np.zeros(nL, dtype=int)
    for l in L:
        li = L_idx[l]
        from_bus[li] = int(df_branch.loc[df_branch["BranchID"] == l, "FromBus"].iloc[0])
        to_bus[li]   = int(df_branch.loc[df_branch["BranchID"] == l, "ToBus"].iloc[0])

    # Transaction PTDF for each line-endpoint transaction (from_k -> to_k)
    ptdf_txn = np.zeros((nL, nL), dtype=float)
    for k in range(nL):
        i = N_idx[from_bus[k]]
        j = N_idx[to_bus[k]]
        ptdf_txn[:, k] = PTDF_full[:, i] - PTDF_full[:, j]

    denom = 1.0 - np.diag(ptdf_txn)
    bad_k = np.abs(denom) < eps

    for k in range(nL):
        if bad_k[k]:
            lodf[k, k] = -1.0
            continue
        lodf[:, k] = ptdf_txn[:, k] / denom[k]
        lodf[k, k] = -1.0

    return lodf, bad_k


# -----------------------------
# Core builder: D-2 nodal CGM + optional preventive N-1 constraints (LODF-only)
# NO pred_np, NO slack_pos/slack_neg anywhere.
# -----------------------------
def build_d2_cgm_core(
    *,
    cost_curt=None,
    frm=0.05,
    max_ntc=1000.0,
    preventive=False,
    LODF=None,                     # (nL,nL) numpy array (constant)
    bad_k=None,                    # (nL,) boolean array
    contingencies_idx=None,        # optional k indices; default all lines
    monitored_idx=None,            # optional monitored indices; default all lines
):
    """
    Returns:
      objective, constraints, params_dict, vars_dict, cost_curt_used

    Model:
      - D-2 nodal DC-OPF with exports and FBMC-zone NPs as derived variables
      - Optional preventive SCOPF (N-1) using LODF-only:
            f_post(k) = f + LODF[:,k] * f[k]
        enforce limits on monitored lines excluding the outaged line k.
    """

    nP = len(P)
    nN = len(N)
    nL = len(L)
    nZ_fb = len(Z_FBMC)
    nZ_not = len(Z_not_in_FBMC)

    # Slack node
    slack_node = 50
    slack_idx = N_idx[slack_node]

    # Curtailment cost
    if cost_curt is None:
        cost_curt = math.ceil(find_maximum_mc())

    # -----------------------
    # Parameters (forward pass)
    # -----------------------
    params = {
        "dem": cp.Parameter(nN),
        "renew": cp.Parameter(nN),
        "line_cap": cp.Parameter(nL),  # allow time-varying caps if you ever want; can pass constant too
    }

    # -----------------------
    # Variables
    # -----------------------
    vars_ = {
        "GEN": cp.Variable(nP),
        "CURT": cp.Variable(nN),
        "DELTA": cp.Variable(nN),
        "NOD_INJ": cp.Variable(nN),
        "LINE_F": cp.Variable(nL),
        "NP": cp.Variable(nZ_fb),
        "EXPORT": cp.Variable(nZ_not),
    }

    constraints = []

    # -----------------------
    # Bounds
    # -----------------------
    constraints += [vars_["GEN"] >= 0, vars_["GEN"] <= gmax]
    constraints += [vars_["CURT"] >= 0, vars_["CURT"] <= params["renew"]]
    constraints += [vars_["EXPORT"] <= max_ntc, vars_["EXPORT"] >= -max_ntc]

    # -----------------------
    # Nodal balance:
    #   sum GEN at node + renew - NOD_INJ - CURT = dem
    # -----------------------
    for n in N:
        ni = N_idx[n]
        p_list = plants_at_node_idx[n]  # indices of plants at node
        gen_sum = cp.sum(vars_["GEN"][p_list]) if len(p_list) > 0 else 0
        constraints += [
            gen_sum + params["renew"][ni] - vars_["NOD_INJ"][ni] - vars_["CURT"][ni] == params["dem"][ni]
        ]

    # -----------------------
    # Export balance (non-FBMC zones):
    #   EXPORT[z] = sum NOD_INJ[n in z]
    # -----------------------
    for z in Z_not_in_FBMC:
        zi = Z_not_idx[z]
        n_list = nodes_in_zone_idx[z]
        constraints += [vars_["EXPORT"][zi] == cp.sum(vars_["NOD_INJ"][n_list])]

    # -----------------------
    # Net positions (FBMC zones):
    #   NP[z] = sum NOD_INJ[n in z] + sum EXPORT[neighbor zones outside FBMC]
    # -----------------------
    for z in Z_FBMC:
        zi = Z_fb_idx[z]
        n_list = nodes_in_zone_idx[z]
        zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]
        export_indices = [Z_not_idx[zz] for zz in zz_list]

        if len(export_indices) > 0:
            constraints += [
                vars_["NP"][zi] == cp.sum(vars_["NOD_INJ"][n_list]) + cp.sum(vars_["EXPORT"][export_indices])
            ]
        else:
            constraints += [vars_["NP"][zi] == cp.sum(vars_["NOD_INJ"][n_list])]

    # -----------------------
    # DC physics
    # -----------------------
    constraints += [vars_["NOD_INJ"] == B_matrix @ vars_["DELTA"]]
    constraints += [vars_["LINE_F"] == H_matrix @ vars_["DELTA"]]
    constraints += [vars_["DELTA"][slack_idx] == 0]

    # -----------------------
    # Base-case line limits (with FRM)
    # -----------------------
    cap_margin = cp.multiply(params["line_cap"], (1.0 - frm))
    constraints += [
        vars_["LINE_F"] <= cap_margin,
        vars_["LINE_F"] >= -cap_margin
    ]

    # -----------------------
    # Preventive SCOPF (N-1) constraints via LODF (LODF-only)
    #   For each outage k:
    #     f_post = f + LODF[:,k] * f[k]
    #     enforce limits on monitored lines, excluding k
    # -----------------------
    if preventive:
        if LODF is None or bad_k is None:
            raise ValueError("preventive=True requires precomputed LODF (nL,nL) and bad_k (nL,).")
        if LODF.shape != (nL, nL):
            raise ValueError(f"LODF must have shape ({nL},{nL}), got {LODF.shape}")
        if bad_k.shape[0] != nL:
            raise ValueError(f"bad_k must have length {nL}, got {bad_k.shape[0]}")

        if contingencies_idx is None:
            contingencies_idx = np.arange(nL, dtype=int)
        else:
            contingencies_idx = np.asarray(contingencies_idx, dtype=int)

        if monitored_idx is None:
            monitored_idx = np.arange(nL, dtype=int)
        else:
            monitored_idx = np.asarray(monitored_idx, dtype=int)

        monitored_set = set(monitored_idx.tolist())
        f = vars_["LINE_F"]

        for k in contingencies_idx.tolist():
            if bad_k[k]:
                continue

            f_post = f + cp.multiply(LODF[:, k], f[k])

            # enforce on monitored lines except the outaged one
            if k in monitored_set:
                idxs = np.array([i for i in monitored_idx if i != k], dtype=int)
            else:
                idxs = monitored_idx

            constraints += [
                f_post[idxs] <= cap_margin[idxs],
                f_post[idxs] >= -cap_margin[idxs],
            ]

    # -----------------------
    # Objective (same as your old "components" objective)
    # -----------------------
    objective = cp.Minimize(cost_gen @ vars_["GEN"] + cost_curt * cp.sum(vars_["CURT"]))

    return objective, constraints, params, vars_, cost_curt


# -----------------------------
# Differentiable layer wrapper (NO pred_np)
# -----------------------------
def build_d2_cgm_layer(
    *,
    cost_curt=None,
    frm=0.05,
    max_ntc=1000.0,
    preventive=False,
    LODF=None,
    bad_k=None,
    contingencies_idx=None,
    monitored_idx=None,
):
    """
    Differentiable D-2 CGM layer (NO pred_np), optionally preventive N-1 via LODF-only.

    Forward-pass parameters:
      dem:   (|N|,)
      renew: (|N|,)
      line_cap: (|L|,)  (you can pass constant line_cap each call)
    """
    objective, constraints, params, vars_, _ = build_d2_cgm_core(
        cost_curt=cost_curt,
        frm=frm,
        max_ntc=max_ntc,
        preventive=preventive,
        LODF=LODF,
        bad_k=bad_k,
        contingencies_idx=contingencies_idx,
        monitored_idx=monitored_idx,
    )

    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[params["dem"], params["renew"], params["line_cap"]],
        variables=[
            vars_["GEN"],
            vars_["CURT"],
            vars_["NOD_INJ"],
            vars_["LINE_F"],
            vars_["NP"],
            vars_["EXPORT"],
            vars_["DELTA"],
        ],
    )
    return layer


# -----------------------------
# Components wrapper (for scripts that solve + export)
# -----------------------------
def build_d2_cgm_problem_components(
    *,
    cost_curt=None,
    frm=0.05,
    max_ntc=1000.0,
    preventive=True,
    LODF=None,
    bad_k=None,
    contingencies_idx=None,
    monitored_idx=None,
):
    """
    Returns:
      objective, constraints, params_list, vars_list, cost_curt_used, params_dict, vars_dict
    """
    objective, constraints, params, vars_, cost_curt_used = build_d2_cgm_core(
        cost_curt=cost_curt,
        frm=frm,
        max_ntc=max_ntc,
        preventive=preventive,
        LODF=LODF,
        bad_k=bad_k,
        contingencies_idx=contingencies_idx,
        monitored_idx=monitored_idx,
    )

    params_list = [params["dem"], params["renew"], params["line_cap"]]
    vars_list = [
        vars_["GEN"], vars_["CURT"], vars_["NOD_INJ"], vars_["LINE_F"],
        vars_["NP"], vars_["EXPORT"], vars_["DELTA"]
    ]

    return objective, constraints, params_list, vars_list, cost_curt_used, params, vars_