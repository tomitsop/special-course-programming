###############################
#  D-2 CGM Differentiable Layer
###############################

import math
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer  # or .jax for JAX backend

from input_data_base_functions import *


def build_d2_cgm_layer(cost_curt=None, frm=0.05):
    """
    Parameters (fixed at layer construction):
        cost_curt : float or None
            Curtailment cost. If None, it is set to ceil(max marginal cost among P_RD)
        frm : float
            Flow reliability margin

    Layer parameters (forward pass):
        pred_np : tensor/array (|Z_FBMC|,)
            Predicted NPs per FBMC zone from the NN.
        dem : tensor/array (|N|,)
            Nodal demand per time stemp.
        renew : tensor/array (|N|,)
            Available renewable generation per time step.

    Layer outputs:
        GEN, CURT, NOD_INJ, LINE_F, NP, EXPORT, DELTA, slack_pos, slack_neg,
        GEN_COSTS, CURT_COSTS, OBJECTIVE_PER_HOUR
    """

    # Dimensions
    nP = len(P)
    nN = len(N)
    nL = len(L)
    nZ_fb = len(Z_FBMC)
    nZ_not = len(Z_not_in_FBMC)
    max_ntc = 1000.0  # NTC bound on EXPORT

    # Slack node index
    slack_node = 50
    slack_idx = N_idx[slack_node]

    # Curtailment cost
    if cost_curt is None:
        max_mc = find_maximum_mc()
        cost_curt = math.ceil(max_mc)

    # --------- Parameters (cvxpy) ---------
    # Predicted NPs (from NN) for FBMC zones
    pred_np = cp.Parameter(nZ_fb)       # shape (|Z_FBMC|,)

    # Demand and renewable availability at nodes
    dem = cp.Parameter(nN)              # shape (|N|,)
    renew = cp.Parameter(nN)            # shape (|N|,)

    # --------- Variables (cvxpy) ---------
    GEN = cp.Variable(nP)               # generation at conventional plants
    CURT = cp.Variable(nN)              # curtailment at nodes
    DELTA = cp.Variable(nN)             # voltage angles
    NOD_INJ = cp.Variable(nN)           # nodal injections
    LINE_F = cp.Variable(nL)            # line flows

    NP = cp.Variable(nZ_fb)             # net positions for FBMC zones
    EXPORT = cp.Variable(nZ_not)        # exports for non-FBMC zones

    # NP slack variables (for matching predictions)
    slack_pos = cp.Variable(nZ_fb)
    slack_neg = cp.Variable(nZ_fb)

    # # Zonal generation and curtailment costs (auxiliary)
    # GEN_COSTS = cp.Variable((len(Z),))      # one per zone
    # CURT_COSTS = cp.Variable((len(Z),))     # one per zone

    # # # Objective per hour (auxiliary,OBJECTIVE_PER_HOUR[t])
    # OBJECTIVE_PER_HOUR = cp.Variable(1)

    constraints = []

    # --------- Bounds ---------
    constraints += [GEN >= 0, GEN <= gmax]
    constraints += [CURT >= 0, CURT <= renew]
    constraints += [slack_pos >= 0, slack_neg >= 0]

    # --------- OBJECTIVE_PER_HOUR definition ---------
    # OBJECTIVE_PER_HOUR = sum GEN*cost + sum CURT*cost_curt
    # constraints += [
    #     OBJECTIVE_PER_HOUR[0] == cost_gen @ GEN + cost_curt * cp.sum(CURT)
    # ]

    # # --------- Zonal generation costs (GEN_COSTS) ---------
    # # For each zone z: sum_{p in p_in_z[z]} GEN[p]*mc[p] == GEN_COSTS[z]
    # for z in Z:
    #     z_idx = Z_idx[z]
    #     p_list = [P_idx[p] for p in p_in_z[z]]  # plants in this zone, indices
    #     if len(p_list) > 0:
    #         constraints += [
    #             GEN_COSTS[z_idx] == cost_gen[p_list] @ GEN[p_list]
    #         ]
    #     else:
    #         constraints += [
    #             GEN_COSTS[z_idx] == 0
    #         ]

    # --------- Zonal curtailment costs (CURT_COSTS) ---------
    # For each zone z: sum_{n in n_in_z[z]} CURT[n]*cost_curt == CURT_COSTS[z]
    # for z in Z:
    #     z_idx = Z_idx[z]
    #     n_list = nodes_in_zone_idx[z]   # indices of nodes in this zone
    #     if len(n_list) > 0:
    #         constraints += [
    #             CURT_COSTS[z_idx] == cost_curt * cp.sum(CURT[n_list])
    #         ]
    #     else:
    #         constraints += [
    #             CURT_COSTS[z_idx] == 0
    #         ]

    # --------- Nodal balance constraints ---------
    # For each node n:
    #   sum_{p in p_at_n[n]} GEN[p] + renew[n] - NOD_INJ[n] - CURT[n] == dem[n]
    for n in N:
        ni = N_idx[n]
        p_list = plants_at_node_idx[n]  # plant indices at this node
        gen_sum = cp.sum(GEN[p_list]) if len(p_list) > 0 else 0
        constraints += [
            gen_sum + renew[ni] - NOD_INJ[ni] - CURT[ni] == dem[ni]
        ]

    # --------- Export balance constraints (non-FBMC zones) ---------
    # For z in Z_not_in_FBMC:
    #   EXPORT[z] == sum_{n in n_in_z[z]} NOD_INJ[n]
    for z in Z_not_in_FBMC:
        zi = Z_not_idx[z]
        n_list = nodes_in_zone_idx[z]
        constraints += [
            EXPORT[zi] == cp.sum(NOD_INJ[n_list])
        ]

    # --------- Net position constraints (FBMC zones) ---------
    # NP[z] = sum_{n in n_in_z[z]} NOD_INJ[n] + sum EXPORT[z'] where z' connected and outside FBMC
    for z in Z_FBMC:
        zi = Z_fb_idx[z]
        n_list = nodes_in_zone_idx[z]
        # zones outside FBMC that are neighbors of z
        zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]
        export_indices = [Z_not_idx[zz] for zz in zz_list]
        if len(export_indices) > 0:
            constraints += [
                NP[zi] == cp.sum(NOD_INJ[n_list]) + cp.sum(EXPORT[export_indices])
            ]
        else:
            constraints += [
                NP[zi] == cp.sum(NOD_INJ[n_list])
            ]
        
    # EXPORT NTC limits    
    constraints += [
                EXPORT <= max_ntc,
                EXPORT >= -max_ntc,
    ]

    # --------- NP matching prediction + slack ---------
    # NP[z] = pred_np[z] + slack_pos[z] - slack_neg[z]
    constraints += [NP == pred_np + slack_pos - slack_neg]

    # --------- Nodal injection via susceptance ---------
    # NOD_INJ = B * DELTA
    constraints += [NOD_INJ == B_matrix @ DELTA]

    # --------- Line flows via line susceptance ---------
    # LINE_F = H * DELTA
    constraints += [LINE_F == H_matrix @ DELTA]

    # --------- Line capacity limits ---------
    constraints += [
        LINE_F <= line_cap * (1.0 - frm),
        LINE_F >= -line_cap * (1.0 - frm)
    ]

    # --------- slack node angle ---------
    constraints += [
        DELTA[slack_idx] == 0
    ]

    # --------- Objective function ---------
    objective = cp.Minimize(
        cost_gen @ GEN
        + cost_curt * cp.sum(CURT)
        + cost_curt * 1.1 * (cp.sum(slack_pos) + cp.sum(slack_neg))
    )

    problem = cp.Problem(objective, constraints)

    # --------- Wrap as CvxpyLayer ---------
    layer = CvxpyLayer(
        problem,
        parameters=[pred_np, dem, renew],
        variables=[
            GEN,
            CURT,
            NOD_INJ,
            LINE_F,
            NP,
            EXPORT,
            DELTA,
            slack_pos,
            slack_neg,
            # GEN_COSTS,
            # CURT_COSTS,
            # OBJECTIVE_PER_HOUR,
        ],
    )

    return layer


def build_d2_cgm_problem_components(cost_curt=None, frm=0.05):
    """
    Same model as build_d2_cgm_layer, but returns CVXPY components
    so other scripts (SCOPF) can extend constraints before creating cp.Problem.
    """
    nP = len(P)
    nN = len(N)
    nL = len(L)
    nZ_fb = len(Z_FBMC)
    nZ_not = len(Z_not_in_FBMC)
    max_ntc = 1000.0

    slack_node = 50
    slack_idx = N_idx[slack_node]

    if cost_curt is None:
        max_mc = find_maximum_mc()
        cost_curt = math.ceil(max_mc)

    # Parameters
    # pred_np = cp.Parameter(nZ_fb)
    dem = cp.Parameter(nN)
    renew = cp.Parameter(nN)

    # Variables
    GEN = cp.Variable(nP)
    CURT = cp.Variable(nN)
    DELTA = cp.Variable(nN)
    NOD_INJ = cp.Variable(nN)
    LINE_F = cp.Variable(nL)

    NP = cp.Variable(nZ_fb)
    EXPORT = cp.Variable(nZ_not)

    # slack_pos = cp.Variable(nZ_fb)
    # slack_neg = cp.Variable(nZ_fb)

    constraints = []

    # Bounds
    constraints += [GEN >= 0, GEN <= gmax]
    constraints += [CURT >= 0, CURT <= renew]
    # constraints += [slack_pos >= 0, slack_neg >= 0]

    # Nodal balance
    for n in N:
        ni = N_idx[n]
        p_list = plants_at_node_idx[n]
        gen_sum = cp.sum(GEN[p_list]) if len(p_list) > 0 else 0
        constraints += [gen_sum + renew[ni] - NOD_INJ[ni] - CURT[ni] == dem[ni]]

    # Export balance (non-FBMC zones)
    for z in Z_not_in_FBMC:
        zi = Z_not_idx[z]
        n_list = nodes_in_zone_idx[z]
        constraints += [EXPORT[zi] == cp.sum(NOD_INJ[n_list])]

    constraints += [EXPORT <= max_ntc, EXPORT >= -max_ntc]

    # Net positions (FBMC zones)
    for z in Z_FBMC:
        zi = Z_fb_idx[z]
        n_list = nodes_in_zone_idx[z]
        zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]
        export_indices = [Z_not_idx[zz] for zz in zz_list]
        if len(export_indices) > 0:
            constraints += [NP[zi] == cp.sum(NOD_INJ[n_list]) + cp.sum(EXPORT[export_indices])]
        else:
            constraints += [NP[zi] == cp.sum(NOD_INJ[n_list])]

    # NP matching prediction + slack
    # constraints += [NP == pred_np + slack_pos - slack_neg]

    # DC physics
    constraints += [NOD_INJ == B_matrix @ DELTA]
    constraints += [LINE_F == H_matrix @ DELTA]

    # Line limits
    constraints += [
        LINE_F <= line_cap * (1.0 - frm),
        LINE_F >= -line_cap * (1.0 - frm)
    ]

    # slack node
    constraints += [DELTA[slack_idx] == 0]

    # Objective
    objective = cp.Minimize(
        cost_gen @ GEN
        + cost_curt * cp.sum(CURT)
    )

    # params = [pred_np, dem, renew]
    params = [dem, renew]

    # vars_ = [GEN, CURT, NOD_INJ, LINE_F, NP, EXPORT, DELTA, slack_pos, slack_neg]
    vars_ = [GEN, CURT, NOD_INJ, LINE_F, NP, EXPORT, DELTA]


    return objective, constraints, params, vars_, cost_curt