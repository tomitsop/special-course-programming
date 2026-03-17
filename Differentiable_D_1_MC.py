import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from input_data_base_functions import (
    # sets
    P, N, Z, Z_FBMC, Z_not_in_FBMC,
    # index maps
    P_idx, N_idx, Z_idx, Z_fb_idx,
    # mappings
    p_in_z, n_in_z, z_to_z,
    # parameters / arrays
    cost_gen, gmax,
    PTDF_Z_CNEC, CNEC, CNEC_indices,
    frm,
)


def _build_export_structure():
    """
    Build export pair indexing exactly like the original model:
    for z in Z, zz in z_to_z[z].
    """
    export_pairs = [(z, zz) for z in Z for zz in z_to_z[z]]
    export_idx = {pair: i for i, pair in enumerate(export_pairs)}
    return export_pairs, export_idx


def build_d1_mc_problem_components(
    cost_curt_mc=0.0,
    max_ntc=1000.0,
    export_eps=1e-7,
):
    """
    Build plain CVXPY problem components for the D-1 Market Coupling model.

    This preserves the optimization exactly as in the original layer version,
    but exposes:
        - problem
        - parameters
        - variables
        - metadata

    so that the forward pass can be solved directly with Gurobi, e.g.:

        comps = build_d1_mc_problem_components(...)
        params = comps["parameters"]

        params["dem"].value = ...
        params["renew"].value = ...
        params["np_d2_fb"].value = ...
        params["ram_pos"].value = ...
        params["ram_neg"].value = ...

        comps["problem"].solve(solver=cp.GUROBI, verbose=False)

    Parameters (fixed):
        cost_curt_mc : curtailment cost
        max_ntc      : NTC bound on EXPORT
        export_eps   : tiny export cost

    Parameters (per solve):
        dem      : (|N|,) nodal demand
        renew    : (|N|,) nodal renewable availability
        np_d2_fb : (|Z_FBMC|,) D-2 net positions for FB zones
        ram_pos  : (|CNEC|,) RAM+ for each CNEC
        ram_neg  : (|CNEC|,) RAM- for each CNEC

    Variables:
        GEN     : (|P|,)
        CURT    : (|N|,)
        NP      : (|Z_FBMC|,)
        EXPORT  : (|E|,)
    """

    nP = len(P)
    nN = len(N)
    nZ = len(Z)
    nZ_fb = len(Z_FBMC)
    nCNE = len(CNEC)

    export_pairs, export_idx = _build_export_structure()
    nE = len(export_pairs)

    # -------- Parameters --------
    dem = cp.Parameter(nN, name="dem")
    renew = cp.Parameter(nN, name="renew")
    np_d2_fb = cp.Parameter(nZ_fb, name="np_d2_fb")
    ram_pos = cp.Parameter(nCNE, name="ram_pos")
    ram_neg = cp.Parameter(nCNE, name="ram_neg")

    # -------- Variables --------
    GEN = cp.Variable(nP, name="GEN")
    CURT = cp.Variable(nN, name="CURT")
    NP = cp.Variable(nZ_fb, name="NP")
    EXPORT = cp.Variable(nE, name="EXPORT")

    constraints = []

    # Bounds
    constraints += [GEN >= 0, GEN <= gmax]
    constraints += [CURT >= 0, CURT <= renew]
    constraints += [EXPORT >= 0, EXPORT <= max_ntc]

    # ---------- Zonal balance constraints (FBMC zones) ----------
    for z in Z_FBMC:
        zfi = Z_fb_idx[z]
        nodes = [N_idx[n] for n in n_in_z[z]]
        plants = [P_idx[p] for p in p_in_z[z]]

        # neighbors outside FBMC only
        zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]

        exports_out = [EXPORT[export_idx[(z, zz)]] for zz in zz_list]
        exports_in = [
            EXPORT[export_idx[(zz, z)]]
            for zz in zz_list if (zz, z) in export_idx
        ]

        gen_sum = cp.sum(GEN[plants]) if plants else 0
        ren_sum = cp.sum(renew[nodes]) if nodes else 0
        curt_sum = cp.sum(CURT[nodes]) if nodes else 0
        dem_sum = cp.sum(dem[nodes]) if nodes else 0

        constraints += [
            gen_sum + ren_sum - curt_sum
            - cp.sum(exports_out) + cp.sum(exports_in)
            - NP[zfi]
            == dem_sum
        ]

    # ---------- Zonal balance constraints (outside FBMC) ----------
    for z in Z_not_in_FBMC:
        nodes = [N_idx[n] for n in n_in_z[z]]
        plants = [P_idx[p] for p in p_in_z[z]]

        zz_list = z_to_z[z]

        exports_out = [EXPORT[export_idx[(z, zz)]] for zz in zz_list]
        exports_in = [
            EXPORT[export_idx[(zz, z)]]
            for zz in zz_list if (zz, z) in export_idx
        ]

        gen_sum = cp.sum(GEN[plants]) if plants else 0
        ren_sum = cp.sum(renew[nodes]) if nodes else 0
        curt_sum = cp.sum(CURT[nodes]) if nodes else 0
        dem_sum = cp.sum(dem[nodes]) if nodes else 0

        constraints += [
            gen_sum + ren_sum - curt_sum
            - cp.sum(exports_out) + cp.sum(exports_in)
            == dem_sum
        ]

    # Net position sum-zero inside FBMC
    constraints += [cp.sum(NP) == 0]

    # ---------- Flow-based constraints on CNECs ----------
    for j_idx in range(nCNE):
        ptdf_row = PTDF_Z_CNEC[j_idx, :]
        flow_expr = ptdf_row @ (NP - np_d2_fb)

        constraints += [
            flow_expr <= ram_pos[j_idx],
            flow_expr >= ram_neg[j_idx],
        ]

    # Objective
    objective = cp.Minimize(
        cost_gen @ GEN
        + cost_curt_mc * cp.sum(CURT)
        + export_eps * cp.sum(EXPORT)
    )

    problem = cp.Problem(objective, constraints)

    return {
        "problem": problem,
        "parameters": {
            "dem": dem,
            "renew": renew,
            "np_d2_fb": np_d2_fb,
            "ram_pos": ram_pos,
            "ram_neg": ram_neg,
        },
        "variables": {
            "GEN": GEN,
            "CURT": CURT,
            "NP": NP,
            "EXPORT": EXPORT,
        },
        "metadata": {
            "export_pairs": export_pairs,
            "export_idx": export_idx,
            "nP": nP,
            "nN": nN,
            "nZ": nZ,
            "nZ_fb": nZ_fb,
            "nCNE": nCNE,
            "nE": nE,
        },
    }


def build_d1_mc_layer(
    cost_curt_mc=0.0,
    max_ntc=1000.0,
    export_eps=1e-7,
):
    """
    Differentiable D-1 MC layer matching the original implementation.

    This now reuses build_d1_mc_problem_components() so the optimization
    remains identical between:
      - differentiable layer usage
      - direct forward solve usage with a plain CVXPY problem
    """

    comps = build_d1_mc_problem_components(
        cost_curt_mc=cost_curt_mc,
        max_ntc=max_ntc,
        export_eps=export_eps,
    )

    problem = comps["problem"]
    parameters = comps["parameters"]
    variables = comps["variables"]

    layer = CvxpyLayer(
        problem,
        parameters=[
            parameters["dem"],
            parameters["renew"],
            parameters["np_d2_fb"],
            parameters["ram_pos"],
            parameters["ram_neg"],
        ],
        variables=[
            variables["GEN"],
            variables["CURT"],
            variables["NP"],
            variables["EXPORT"],
        ],
    )

    return layer