import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np


def build_d0_redispatch_problem_components(
    *,
    N, L, P, P_RD,
    N_idx, L_idx, P_idx,
    p_at_n,
    p_rd_at_n,
    B_matrix,
    H_matrix,
    slack_node,
    cost_curt,
):
    """
    Build plain CVXPY problem components for the D-0 redispatch model.

    This preserves the optimization exactly as in the original layer version,
    but exposes:
        - problem
        - parameters
        - variables
        - metadata

    so the forward pass can be solved directly with Gurobi, e.g.:

        comps = build_d0_redispatch_problem_components(...)
        params = comps["parameters"]

        params["dem"].value = ...
        params["renew"].value = ...
        params["gen_d1"].value = ...
        params["curt_d1"].value = ...
        params["mc_rd"].value = ...
        params["gmax_rd"].value = ...
        params["line_cap"].value = ...

        comps["problem"].solve(solver=cp.GUROBI, verbose=False)

    Parameters per solve:
        dem      : (|N|,) nodal demand
        renew    : (|N|,) nodal renewable availability
        gen_d1   : (|P|,) generation schedule from D-1
        curt_d1  : (|N|,) curtailment schedule from D-1
        mc_rd    : (|P_RD|,) redispatch marginal costs
        gmax_rd  : (|P_RD|,) redispatchable max generation
        line_cap : (|L|,) line capacities

    Variables:
        CURT_RD : (|N|,) additional curtailment redispatch
        RD_POS  : (|P_RD|,) upward redispatch
        RD_NEG  : (|P_RD|,) downward redispatch
        DELTA   : (|N|,) voltage angles
        NOD_INJ : (|N|,) nodal injections
        LINE_F  : (|L|,) line flows
    """

    nN = len(N)
    nL = len(L)
    nP = len(P)
    nP_RD = len(P_RD)

    slack_idx = N_idx[slack_node]

    # RD plant indexing
    P_RD_idx = {p: i for i, p in enumerate(P_RD)}

    # parameters
    dem = cp.Parameter(nN, name="dem")
    renew = cp.Parameter(nN, name="renew")

    gen_d1 = cp.Parameter(nP, name="gen_d1")    # GEN from D-1
    curt_d1 = cp.Parameter(nN, name="curt_d1")  # CURT from D-1

    mc_rd = cp.Parameter(nP_RD, name="mc_rd")
    gmax_rd = cp.Parameter(nP_RD, name="gmax_rd")

    line_cap = cp.Parameter(nL, name="line_cap")

    # variables
    CURT_RD = cp.Variable(nN, name="CURT_RD")

    RD_POS = cp.Variable(nP_RD, name="RD_POS")
    RD_NEG = cp.Variable(nP_RD, name="RD_NEG")

    DELTA = cp.Variable(nN, name="DELTA")
    NOD_INJ = cp.Variable(nN, name="NOD_INJ")
    LINE_F = cp.Variable(nL, name="LINE_F")

    constraints = []

    # ---- Curtailment bounds ----
    constraints += [
        CURT_RD >= -curt_d1,
        CURT_RD <= renew - curt_d1
    ]

    # ---- Redispatch bounds ----
    gen_d1_rd = cp.hstack([gen_d1[P_idx[p]] for p in P_RD])

    constraints += [
        RD_POS >= 0,
        RD_NEG >= 0,
        RD_POS <= gmax_rd - gen_d1_rd,
        RD_NEG <= gen_d1_rd
    ]

    # ---- DC power flow ----
    constraints += [NOD_INJ == B_matrix @ DELTA]
    constraints += [LINE_F == H_matrix @ DELTA]

    # ---- Slack ----
    constraints += [DELTA[slack_idx] == 0]

    # ---- Line limits ----
    constraints += [
        LINE_F <= line_cap,
        -LINE_F <= line_cap
    ]

    # ---- Nodal balance ----
    for n in N:
        ni = N_idx[n]

        plants = p_at_n[n]

        if plants:
            base_gen = cp.sum(cp.hstack([gen_d1[P_idx[p]] for p in plants]))
        else:
            base_gen = 0

        rd_plants = p_rd_at_n[n]

        if rd_plants:
            rd_pos = cp.sum(cp.hstack([RD_POS[P_RD_idx[p]] for p in rd_plants]))
            rd_neg = cp.sum(cp.hstack([RD_NEG[P_RD_idx[p]] for p in rd_plants]))
        else:
            rd_pos = 0
            rd_neg = 0

        constraints += [
            base_gen
            + rd_pos
            - rd_neg
            - NOD_INJ[ni]
            + renew[ni]
            - curt_d1[ni]
            - CURT_RD[ni]
            == dem[ni]
        ]

    # ---- Objective (exactly as original) ----
    objective = cp.Minimize(
        (100 + mc_rd) @ RD_POS
        + (100 + cost_curt - mc_rd) @ RD_NEG
        + (100 + cost_curt) * cp.sum(CURT_RD)
    )

    problem = cp.Problem(objective, constraints)

    return {
        "problem": problem,
        "parameters": {
            "dem": dem,
            "renew": renew,
            "gen_d1": gen_d1,
            "curt_d1": curt_d1,
            "mc_rd": mc_rd,
            "gmax_rd": gmax_rd,
            "line_cap": line_cap,
        },
        "variables": {
            "CURT_RD": CURT_RD,
            "RD_POS": RD_POS,
            "RD_NEG": RD_NEG,
            "DELTA": DELTA,
            "NOD_INJ": NOD_INJ,
            "LINE_F": LINE_F,
        },
        "metadata": {
            "nN": nN,
            "nL": nL,
            "nP": nP,
            "nP_RD": nP_RD,
            "slack_idx": slack_idx,
            "P_RD_idx": P_RD_idx,
        },
    }


def build_d0_redispatch_layer(
    *,
    N, L, P, P_RD,
    N_idx, L_idx, P_idx,
    p_at_n,
    p_rd_at_n,
    B_matrix,
    H_matrix,
    slack_node,
    cost_curt,
):
    """
    Differentiable D-0 redispatch layer.

    This reuses build_d0_redispatch_problem_components() so the optimization
    remains identical between:
      - differentiable layer usage
      - direct forward solve usage with a plain CVXPY problem
    """

    comps = build_d0_redispatch_problem_components(
        N=N,
        L=L,
        P=P,
        P_RD=P_RD,
        N_idx=N_idx,
        L_idx=L_idx,
        P_idx=P_idx,
        p_at_n=p_at_n,
        p_rd_at_n=p_rd_at_n,
        B_matrix=B_matrix,
        H_matrix=H_matrix,
        slack_node=slack_node,
        cost_curt=cost_curt,
    )

    problem = comps["problem"]
    parameters = comps["parameters"]
    variables = comps["variables"]

    layer = CvxpyLayer(
        problem,
        parameters=[
            parameters["dem"],
            parameters["renew"],
            parameters["gen_d1"],
            parameters["curt_d1"],
            parameters["mc_rd"],
            parameters["gmax_rd"],
            parameters["line_cap"]
        ],
        variables=[
            variables["CURT_RD"],
            variables["RD_POS"],
            variables["RD_NEG"],
            variables["DELTA"],
            variables["NOD_INJ"],
            variables["LINE_F"]
        ]
    )

    return layer


# Use in main
# mc_rd = np.array([get_mc(p) for p in P_RD], dtype=float)
# gmax_rd = np.array([get_gen_up(p) for p in P_RD], dtype=float)