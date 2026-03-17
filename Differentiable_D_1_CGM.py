###############################
#  D-1 CGM Differentiable DC Flow Layer
###############################

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from input_data_base_functions import (
    # sets
    P, N, L, Z, Z_FBMC, Z_not_in_FBMC,
    # index maps
    P_idx, N_idx, L_idx, Z_idx, Z_fb_idx, Z_not_idx,
    # mappings
    p_at_n, p_in_z, n_in_z, z_to_z,
    # parameters / arrays
    cost_gen, gmax,
    B_matrix, H_matrix,
    # slack node info
    slack_node,
)


def build_d1_cgm_problem_components(cost_curt, max_ntc):
    """
    Build plain CVXPY problem components for the D-1 CGM model.

    This preserves the optimization exactly as in the original layer version,
    but exposes:
        - problem
        - parameters
        - variables
        - metadata

    so that the forward pass can be solved directly with Gurobi, e.g.:

        comps = build_d1_cgm_problem_components(...)
        params = comps["parameters"]

        params["dem"].value = ...
        params["renew"].value = ...
        params["gen_sched"].value = ...
        params["curt_sched"].value = ...

        comps["problem"].solve(solver=cp.GUROBI, verbose=False)

    Fixed args:
        cost_curt : curtailment cost
        max_ntc   : export bound for non-FBMC zones

    Per-solve parameters:
        dem        : (|N|,) nodal demand
        renew      : (|N|,) nodal renewable availability
        gen_sched  : (|P|,) fixed generation schedule from D-1 MC
        curt_sched : (|N|,) fixed curtailment schedule from D-1 MC
    """
    nP = len(P)
    nN = len(N)
    nL = len(L)
    nZ = len(Z)
    nZ_fb = len(Z_FBMC)
    nZ_not = len(Z_not_in_FBMC)
    slack_idx = N_idx[slack_node]

    # ---- Parameters ----
    dem = cp.Parameter(nN, name="dem")               # get_dem(t, ·)
    renew = cp.Parameter(nN, name="renew")           # get_renew(t, ·)
    gen_sched = cp.Parameter(nP, name="gen_sched")   # generation.loc[t, :]
    curt_sched = cp.Parameter(nN, name="curt_sched") # curtailment.loc[t, :]

    # ---- Variables ----
    GEN = cp.Variable(nP, name="GEN")
    CURT = cp.Variable(nN, name="CURT")
    DELTA = cp.Variable(nN, name="DELTA")
    NOD_INJ = cp.Variable(nN, name="NOD_INJ")
    LINE_F = cp.Variable(nL, name="LINE_F")

    NP = cp.Variable(nZ_fb, name="NP")
    EXPORT = cp.Variable(nZ_not, name="EXPORT")

    # GEN_COSTS = cp.Variable(nZ)
    # CURT_COSTS = cp.Variable(nZ)
    # OBJECTIVE_PER_HOUR = cp.Variable(1)

    constraints = []

    # ---- Bounds (same as Gurobi) ----
    constraints += [GEN >= 0, GEN <= gmax]          # 0 <= GEN <= get_gen_up
    constraints += [CURT >= 0, CURT <= renew]       # 0 <= CURT <= get_renew
    constraints += [EXPORT >= -max_ntc, EXPORT <= max_ntc]

    # # 1) OBJECTIVE_PER_HOUR definition
    # constraints += [
    #     OBJECTIVE_PER_HOUR[0] ==
    #     cost_gen @ GEN + cost_curt * cp.sum(CURT)
    # ]

    # # 2) GEN_COSTS[z] == Σ GEN*mc
    # # 3) CURT_COSTS[z] == Σ CURT*cost_curt
    # for z in Z:
    #     zi = Z_idx[z]
    #     plants = [P_idx[p] for p in p_in_z[z]]
    #     nodes = [N_idx[n] for n in n_in_z[z]]

    #     if plants:
    #         constraints += [
    #             GEN_COSTS[zi] == cost_gen[plants] @ GEN[plants]
    #         ]
    #     else:
    #         constraints += [GEN_COSTS[zi] == 0]

    #     if nodes:
    #         constraints += [
    #             CURT_COSTS[zi] == cost_curt * cp.sum(CURT[nodes])
    #         ]
    #     else:
    #         constraints += [CURT_COSTS[zi] == 0]

    # 4) Nodal balance: Σ GEN + RE - NOD_INJ - CURT == DEM
    for n in N:
        ni = N_idx[n]
        plants_at_node = [P_idx[p] for p in p_at_n[n]]
        gen_sum = cp.sum(GEN[plants_at_node]) if plants_at_node else 0
        constraints += [
            gen_sum + renew[ni] - NOD_INJ[ni] - CURT[ni] == dem[ni]
        ]

    # 5) Export balance abroad: EXPORT[z_not] == Σ NOD_INJ
    for z in Z_not_in_FBMC:
        zi_not = Z_not_idx[z]
        nodes = [N_idx[n] for n in n_in_z[z]]
        if nodes:
            constraints += [
                EXPORT[zi_not] == cp.sum(NOD_INJ[nodes])
            ]
        else:
            constraints += [EXPORT[zi_not] == 0]

    # 6) NP definition for FBMC zones
    for z in Z_FBMC:
        zfi = Z_fb_idx[z]
        nodes = [N_idx[n] for n in n_in_z[z]]
        zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]
        export_terms = [
            EXPORT[Z_not_idx[zz]] for zz in zz_list
        ] if zz_list else []

        lhs = cp.sum(NOD_INJ[nodes]) if nodes else 0
        if export_terms:
            lhs = lhs + cp.sum(export_terms)

        constraints += [NP[zfi] == lhs]

    # 7) NOD_INJ = B * DELTA
    constraints += [NOD_INJ == B_matrix @ DELTA]

    # 8) LINE_F = H * DELTA
    constraints += [LINE_F == H_matrix @ DELTA]

    # 9) Slack node angle
    constraints += [DELTA[slack_idx] == 0]

    # 10–11) Fix GEN and CURT to D-1 MC schedules
    constraints += [GEN == gen_sched]
    constraints += [CURT == curt_sched]

    objective = cp.Minimize(cost_gen @ GEN + cost_curt * cp.sum(CURT))
    problem = cp.Problem(objective, constraints)

    return {
        "problem": problem,
        "parameters": {
            "dem": dem,
            "renew": renew,
            "gen_sched": gen_sched,
            "curt_sched": curt_sched,
        },
        "variables": {
            "GEN": GEN,
            "CURT": CURT,
            "DELTA": DELTA,
            "NOD_INJ": NOD_INJ,
            "LINE_F": LINE_F,
            "NP": NP,
            "EXPORT": EXPORT,
            # "GEN_COSTS": GEN_COSTS,
            # "CURT_COSTS": CURT_COSTS,
            # "OBJECTIVE_PER_HOUR": OBJECTIVE_PER_HOUR,
        },
        "metadata": {
            "nP": nP,
            "nN": nN,
            "nL": nL,
            "nZ": nZ,
            "nZ_fb": nZ_fb,
            "nZ_not": nZ_not,
            "slack_idx": slack_idx,
        },
    }


def build_d1_cgm_layer(cost_curt, max_ntc):
    """
    Differentiable D-1 CGM layer.

    This reuses build_d1_cgm_problem_components() so the optimization
    remains identical between:
      - differentiable layer usage
      - direct forward solve usage with a plain CVXPY problem
    """
    comps = build_d1_cgm_problem_components(
        cost_curt=cost_curt,
        max_ntc=max_ntc,
    )

    problem = comps["problem"]
    parameters = comps["parameters"]
    variables = comps["variables"]

    layer = CvxpyLayer(
        problem,
        parameters=[
            parameters["dem"],
            parameters["renew"],
            parameters["gen_sched"],
            parameters["curt_sched"],
        ],
        variables=[
            variables["GEN"],
            variables["CURT"],
            variables["DELTA"],
            variables["NOD_INJ"],
            variables["LINE_F"],
            variables["NP"],
            variables["EXPORT"],
            # variables["GEN_COSTS"],
            # variables["CURT_COSTS"],
            # variables["OBJECTIVE_PER_HOUR"],
        ],
    )

    return layer