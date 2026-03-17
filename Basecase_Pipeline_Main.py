import os
import json
import math
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import cvxpy as cp

from input_data_base_functions import *

# D-2
from Differentiable_D_2_SCOPF_CGM import build_d2_cgm_problem_components,compute_lodf_from_ptdf
# D-1 CGM
from Differentiable_D_1_CGM import build_d1_cgm_problem_components

# D-0 CM
from D_0_Congestion_Management import build_d0_redispatch_problem_components

# GSK utilities (refactored module)
from Basecase_Dynamic_GSK_Definition_pipeline import (
    GSKStrategyManager,
    build_dynamic_headroom_gsk,
    build_dynamic_gen_gsk,
    compute_cnec_from_gsk,
)


###############################################################################
# CONFIG
###############################################################################
#scopf runs for 85% of line capacities
#no scopf runs at 60%
RUN_NAME = "results/pipeline_run_gurobi_SCOPF_dynamic_headroom"
RESULTS_ROOT = Path("results") / RUN_NAME

N_WORKERS = 10
GUROBI_THREADS_PER_WORKER = 1

# Choose one:
# "flat", "flat_unit", "pmax", "pmax_sub", "dynamic_headroom", "dynamic_gen"
GSK_STRATEGY = "dynamic_headroom"

INCLUDE_CB_LINES = True
CNE_ALPHA = cne_alpha

# Optional time selection
TIME_START = None   # e.g. 1
TIME_END = None     # e.g. 200

# Optimization constants
COST_CURT = math.ceil(find_maximum_mc())
COST_CURT_MC = 0.0
MAX_NTC = 1000.0
EXPORT_EPS = 1e-7

# Robust run behavior
CONTINUE_ON_ERROR = True


LODF = None
bad_k = None

###############################################################################
# HELPERS
###############################################################################

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_time_index():
    """
    Uses the same logic you previously had: time points from X / predictions,
    but only to define the MTU universe. If you have a cleaner source of MTUs,
    replace this function.
    """
    X = pd.read_csv("data/X.csv", index_col=0)
    if X.index.min() == 0:
        X.index = X.index + 1

    time_index = X.index.values

    if TIME_START is not None:
        time_index = time_index[time_index >= TIME_START]
    if TIME_END is not None:
        time_index = time_index[time_index <= TIME_END]

    return list(time_index)


def build_dem_renew_np(t: int):
    dem_np = np.array([get_dem(t, n) for n in N], dtype=np.float64)
    renew_np = np.array([get_renew(t, n) for n in N], dtype=np.float64)
    return dem_np, renew_np


def build_line_cap_np(t: int):
    """
    If line capacities are static in your data, this returns the imported line_cap.
    If you later make them time-varying, replace this helper accordingly.
    """
    return np.array(line_cap, dtype=np.float64)


def build_mc_rd_np():
    return np.array([get_mc(p) for p in P_RD], dtype=np.float64)


def build_gmax_rd_np():
    return np.array([get_gen_up(p) for p in P_RD], dtype=np.float64)


def compute_ram_from_d2_numpy(line_f_d2_cnec: np.ndarray, line_cap_cnec: np.ndarray):
    """
    Same formula you showed before:
        base    = line_cap_cnec * (1 - frm)
        RAM_pos = base - line_f_d2_cnec
        RAM_neg = -base - line_f_d2_cnec
    """
    base = line_cap_cnec * (1.0 - float(frm))
    ram_pos = base - line_f_d2_cnec
    ram_neg = -base - line_f_d2_cnec
    return ram_pos.astype(np.float64), ram_neg.astype(np.float64)


def export_pairs_builder():
    return [(z, zz) for z in Z for zz in z_to_z[z]]


###############################################################################
# RUNTIME D-1 MC BUILDER
# Same equations as before, but PTDF_Z_CNEC is passed per MTU
###############################################################################

def build_d1_mc_problem_components_runtime(
    *,
    ptdf_z_cnec_t: np.ndarray,
    cnec_t,
    cost_curt_mc=0.0,
    max_ntc=1000.0,
    export_eps=1e-7,
):
    nP = len(P)
    nN = len(N)
    nZ_fb = len(Z_FBMC)
    nCNE = len(cnec_t)

    export_pairs = export_pairs_builder()
    export_idx = {pair: i for i, pair in enumerate(export_pairs)}
    nE = len(export_pairs)

    dem = cp.Parameter(nN, name="dem")
    renew = cp.Parameter(nN, name="renew")
    np_d2_fb = cp.Parameter(nZ_fb, name="np_d2_fb")
    ram_pos = cp.Parameter(nCNE, name="ram_pos")
    ram_neg = cp.Parameter(nCNE, name="ram_neg")

    GEN = cp.Variable(nP, name="GEN")
    CURT = cp.Variable(nN, name="CURT")
    NP = cp.Variable(nZ_fb, name="NP")
    EXPORT = cp.Variable(nE, name="EXPORT")

    constraints = []
    fbmc_balance_constraints = {}

    constraints += [GEN >= 0, GEN <= gmax]
    constraints += [CURT >= 0, CURT <= renew]
    constraints += [EXPORT >= 0, EXPORT <= max_ntc]

    # FBMC zones
    for z in Z_FBMC:
        zfi = Z_fb_idx[z]
        nodes = [N_idx[n] for n in n_in_z[z]]
        plants = [P_idx[p] for p in p_in_z[z]]

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

        fbmc_balance_constraints[z] = (
            gen_sum + ren_sum - curt_sum
            - cp.sum(exports_out) + cp.sum(exports_in)
            - NP[zfi]
            == dem_sum
        )
        constraints += [fbmc_balance_constraints[z]]

    # non-FBMC zones
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

    constraints += [cp.sum(NP) == 0]

    for j_idx in range(nCNE):
        ptdf_row = ptdf_z_cnec_t[j_idx, :]
        flow_expr = ptdf_row @ (NP - np_d2_fb)
        constraints += [
            flow_expr <= ram_pos[j_idx],
            flow_expr >= ram_neg[j_idx],
        ]

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
        },
        "duals": {
            "power_balance": fbmc_balance_constraints,
        },
    }


###############################################################################
# GSK / CNEC PER MTU
###############################################################################

def build_gsk_payload_for_t(strategy: str, gsk_manager: GSKStrategyManager, gen_d2_np: np.ndarray):
    """
    Static strategies: use manager cache
    Dynamic strategies: build from this MTU's D-2 generation
    """
    if strategy in {"flat", "flat_unit", "pmax", "pmax_sub"}:
        return gsk_manager.build_for_t(strategy=strategy)

    gen_d2_series = pd.Series(gen_d2_np, index=P)

    if strategy == "dynamic_headroom":
        gsk = build_dynamic_headroom_gsk(gen_d2_series)
    elif strategy == "dynamic_gen":
        gsk = build_dynamic_gen_gsk(gen_d2_series)
    else:
        raise ValueError(f"Unknown GSK strategy: {strategy}")

    cnec, cnec_idx, ptdf_z, ptdf_z_cnec = compute_cnec_from_gsk(
        gsk=gsk,
        cne_alpha=CNE_ALPHA,
        include_cb_lines=INCLUDE_CB_LINES,
    )

    return {
        "gsk": gsk,
        "cnec": cnec,
        "cnec_idx": cnec_idx,
        "ptdf_z": ptdf_z,
        "ptdf_z_cnec": ptdf_z_cnec,
    }


###############################################################################
# SINGLE MTU SOLVE
###############################################################################

def solve_single_mtu(t: int, LODF: np.ndarray, bad_k: np.ndarray):
    try:
        # ---------------------------------------------------------------------
        # Local solver options
        # ---------------------------------------------------------------------
        gurobi_opts = {
            "Threads": GUROBI_THREADS_PER_WORKER,
            "OutputFlag": 0,
        }

        # ---------------------------------------------------------------------
        # Build inputs
        # ---------------------------------------------------------------------
        dem_np, renew_np = build_dem_renew_np(t)
        line_cap_np = build_line_cap_np(t)
        mc_rd_np = build_mc_rd_np()
        gmax_rd_np = build_gmax_rd_np()

        gsk_manager = GSKStrategyManager(
            cne_alpha=CNE_ALPHA,
            include_cb_lines=INCLUDE_CB_LINES,
        )

        # ---------------------------------------------------------------------
        # 1) D-2 CGM
        # ---------------------------------------------------------------------

        objective, constraints, params_list, vars_list, _, params, vars_ = \
            build_d2_cgm_problem_components(
                cost_curt=COST_CURT,
                frm=float(frm),
                max_ntc=MAX_NTC,
                preventive=True,   # REMEMBER TO CHANGE FOR SCOPF
                LODF=LODF,
                bad_k=bad_k,
            )

        problem = cp.Problem(objective, constraints)

        # set parameters
        params["dem"].value = dem_np
        params["renew"].value = renew_np
        params["line_cap"].value = line_cap_np

        # solve
        problem.solve(solver=cp.GUROBI, **gurobi_opts)
        obj_d2 = problem.value

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-2 infeasible/status={problem.status}")

        # extract variables
        gen_d2 = np.array(vars_["GEN"].value).reshape(-1)
        curt_d2 = np.array(vars_["CURT"].value).reshape(-1)
        np_d2_fb = np.array(vars_["NP"].value).reshape(-1)
        line_f_d2 = np.array(vars_["LINE_F"].value).reshape(-1)
        delta_d2 = np.array(vars_["DELTA"].value).reshape(-1)
        nod_inj_d2 = np.array(vars_["NOD_INJ"].value).reshape(-1)
        export_d2 = np.array(vars_["EXPORT"].value).reshape(-1)
        
        # ---------------------------------------------------------------------
        # 2) GSK / CNEC / PTDF_Z_CNEC
        # ---------------------------------------------------------------------
        gsk_payload = build_gsk_payload_for_t(
            strategy=GSK_STRATEGY,
            gsk_manager=gsk_manager,
            gen_d2_np=gen_d2,
        )

        gsk_t = gsk_payload["gsk"]
        cnec_t = gsk_payload["cnec"]
        cnec_idx_t = gsk_payload["cnec_idx"]
        ptdf_z_cnec_t = np.array(gsk_payload["ptdf_z_cnec"], dtype=np.float64)

        # ---------------------------------------------------------------------
        # 3) RAM from D-2 flows on CNEC
        # ---------------------------------------------------------------------
        line_f_d2_cnec = line_f_d2[cnec_idx_t]
        line_cap_cnec = line_cap_np[cnec_idx_t]

        ram_pos, ram_neg = compute_ram_from_d2_numpy(
            line_f_d2_cnec=line_f_d2_cnec,
            line_cap_cnec=line_cap_cnec,
        )

        # ---------------------------------------------------------------------
        # 4) D-1 MC
        # ---------------------------------------------------------------------
        d1_mc = build_d1_mc_problem_components_runtime(
            ptdf_z_cnec_t=ptdf_z_cnec_t,
            cnec_t=cnec_t,
            cost_curt_mc=COST_CURT_MC,
            max_ntc=MAX_NTC,
            export_eps=EXPORT_EPS,
        )

        d1_mc["parameters"]["dem"].value = dem_np
        d1_mc["parameters"]["renew"].value = renew_np
        d1_mc["parameters"]["np_d2_fb"].value = np_d2_fb
        d1_mc["parameters"]["ram_pos"].value = ram_pos
        d1_mc["parameters"]["ram_neg"].value = ram_neg

        d1_mc["problem"].solve(solver=cp.GUROBI, **gurobi_opts)
        obj_d1_mc = d1_mc["problem"].value

        if d1_mc["problem"].status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-1 MC infeasible/status={d1_mc['problem'].status}")

        gen_d1 = np.array(d1_mc["variables"]["GEN"].value).reshape(-1)
        curt_d1 = np.array(d1_mc["variables"]["CURT"].value).reshape(-1)
        np_d1 = np.array(d1_mc["variables"]["NP"].value).reshape(-1)
        export_d1_mc = np.array(d1_mc["variables"]["EXPORT"].value).reshape(-1)
        dual_power_balance_d1_mc = np.array(
        [d1_mc["duals"]["power_balance"][z].dual_value for z in Z_FBMC],
        dtype=np.float64
        )

        # ---------------------------------------------------------------------
        # 5) D-1 CGM
        # ---------------------------------------------------------------------
        d1_cgm = build_d1_cgm_problem_components(
            cost_curt=COST_CURT,
            max_ntc=MAX_NTC,
        )

        d1_cgm["parameters"]["dem"].value = dem_np
        d1_cgm["parameters"]["renew"].value = renew_np
        d1_cgm["parameters"]["gen_sched"].value = gen_d1
        d1_cgm["parameters"]["curt_sched"].value = curt_d1

        d1_cgm["problem"].solve(solver=cp.GUROBI, **gurobi_opts)
        obj_d1_cgm = d1_cgm["problem"].value

        if d1_cgm["problem"].status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-1 CGM infeasible/status={d1_cgm['problem'].status}")

        delta_d1 = np.array(d1_cgm["variables"]["DELTA"].value).reshape(-1)
        nod_inj_d1 = np.array(d1_cgm["variables"]["NOD_INJ"].value).reshape(-1)
        line_f_d1 = np.array(d1_cgm["variables"]["LINE_F"].value).reshape(-1)
        np_d1_cgm = np.array(d1_cgm["variables"]["NP"].value).reshape(-1)
        export_d1_cgm = np.array(d1_cgm["variables"]["EXPORT"].value).reshape(-1)

        # ---------------------------------------------------------------------
        # 6) D-0 Congestion Management
        # ---------------------------------------------------------------------
        d0 = build_d0_redispatch_problem_components(
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
            cost_curt=COST_CURT,
        )

        d0["parameters"]["dem"].value = dem_np
        d0["parameters"]["renew"].value = renew_np
        d0["parameters"]["gen_d1"].value = gen_d1
        d0["parameters"]["curt_d1"].value = curt_d1
        d0["parameters"]["mc_rd"].value = mc_rd_np
        d0["parameters"]["gmax_rd"].value = gmax_rd_np
        d0["parameters"]["line_cap"].value = line_cap_np

        d0["problem"].solve(solver=cp.GUROBI, **gurobi_opts)
        obj_d0 = d0["problem"].value

        if d0["problem"].status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-0 CM infeasible/status={d0['problem'].status}")

        curt_rd = np.array(d0["variables"]["CURT_RD"].value).reshape(-1)
        rd_pos = np.array(d0["variables"]["RD_POS"].value).reshape(-1)
        rd_neg = np.array(d0["variables"]["RD_NEG"].value).reshape(-1)
        delta_d0 = np.array(d0["variables"]["DELTA"].value).reshape(-1)
        nod_inj_d0 = np.array(d0["variables"]["NOD_INJ"].value).reshape(-1)
        line_f_d0 = np.array(d0["variables"]["LINE_F"].value).reshape(-1)

        export_pairs = d1_mc["metadata"]["export_pairs"]

        return {
            "t": t,
            "status": "ok",
            
            "objectives": {
                "d2": obj_d2,
                "d1_mc": obj_d1_mc,
                "d1_cgm": obj_d1_cgm,
                "d0": obj_d0,
            },

            "d2": {
                "GEN": gen_d2,
                "CURT": curt_d2,
                "NP": np_d2_fb,
                "LINE_F": line_f_d2,
                "DELTA": delta_d2,
                "NOD_INJ": nod_inj_d2,
                "EXPORT": export_d2,
            },

            "fb": {
                "GSK": gsk_t,
                "CNEC": cnec_t,
                "CNEC_IDX": np.array(cnec_idx_t, dtype=np.int64),
                "RAM_POS": ram_pos,
                "RAM_NEG": ram_neg,
                "PTDF_Z_CNEC": ptdf_z_cnec_t,
            },

            "d1_mc": {
                "GEN": gen_d1,
                "CURT": curt_d1,
                "NP": np_d1,
                "EXPORT": export_d1_mc,
                "DUAL_POWER_BALANCE": dual_power_balance_d1_mc,
            },

            "d1_cgm": {
                "DELTA": delta_d1,
                "NOD_INJ": nod_inj_d1,
                "LINE_F": line_f_d1,
                "NP": np_d1_cgm,
                "EXPORT": export_d1_cgm,
            },

            "d0": {
                "CURT_RD": curt_rd,
                "RD_POS": rd_pos,
                "RD_NEG": rd_neg,
                "DELTA": delta_d0,
                "NOD_INJ": nod_inj_d0,
                "LINE_F": line_f_d0,
            },

            "meta": {
                "export_pairs": export_pairs,
            },
        }

    except Exception as e:
        return {
            "t": t,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


###############################################################################
# SAVE HELPERS
###############################################################################

def series_name_list(prefix, items):
    return [f"{prefix}__{x}" for x in items]


def save_matrix_results(results_ok, stage_dir: Path):
    ensure_dir(stage_dir)

    time_sorted = sorted([r["t"] for r in results_ok])
    
    
    obj_rows = []

    for r in results_ok:
        obj_rows.append({
            "t": r["t"],
            "d2": r["objectives"]["d2"],
            "d1_mc": r["objectives"]["d1_mc"],
            "d1_cgm": r["objectives"]["d1_cgm"],
            "d0": r["objectives"]["d0"],
        })

    pd.DataFrame(obj_rows).set_index("t").to_parquet(stage_dir / "objectives.parquet")

    # ---- D-2 ----
    d2_dir = stage_dir / "d2"
    ensure_dir(d2_dir)

    pd.DataFrame(
        [r["d2"]["GEN"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=P
    ).to_parquet(d2_dir / "gen.parquet")

    pd.DataFrame(
        [r["d2"]["CURT"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d2_dir / "curt.parquet")

    pd.DataFrame(
        [r["d2"]["NP"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=Z_FBMC
    ).to_parquet(d2_dir / "np.parquet")

    pd.DataFrame(
        [r["d2"]["LINE_F"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=L
    ).to_parquet(d2_dir / "line_f.parquet")

    pd.DataFrame(
        [r["d2"]["DELTA"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d2_dir / "delta.parquet")

    pd.DataFrame(
        [r["d2"]["NOD_INJ"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d2_dir / "nod_inj.parquet")

    # D-2 export columns may depend on your implementation.
    # If export dimension matches Z_not_in_FBMC, keep these labels.
    if len(results_ok[0]["d2"]["EXPORT"]) == len(Z_not_in_FBMC):
        d2_export_cols = Z_not_in_FBMC
    else:
        d2_export_cols = [f"export_{i}" for i in range(len(results_ok[0]["d2"]["EXPORT"]))]

    pd.DataFrame(
        [r["d2"]["EXPORT"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=d2_export_cols
    ).to_parquet(d2_dir / "export.parquet")

    # ---- FB ----
    fb_dir = stage_dir / "fb"
    ensure_dir(fb_dir)

    line_cap_df = pd.DataFrame(
        {
            "line_cap": pd.Series(line_cap, index=L, dtype=float),
            "line_cap_margin": pd.Series(line_cap, index=L, dtype=float) * (1.0 - float(frm)),
        }
    )
    line_cap_df.index.name = "line"
    line_cap_df.to_parquet(fb_dir / "line_cap_margin.parquet")

    cnec_rows = []
    for r in sorted(results_ok, key=lambda x: x["t"]):
        cnec_rows.append({
            "t": r["t"],
            "n_cnec": len(r["fb"]["CNEC"]),
            "cnec": list(r["fb"]["CNEC"]),
            "cnec_idx": list(r["fb"]["CNEC_IDX"]),
        })
    pd.DataFrame(cnec_rows).set_index("t").to_parquet(fb_dir / "cnec_info.parquet")

    gsk_long = []
    for r in sorted(results_ok, key=lambda x: x["t"]):
        gsk = r["fb"]["GSK"]
        for ni, n in enumerate(N_FBMC):
            for zi, z in enumerate(Z_FBMC):
                gsk_long.append({
                    "t": r["t"],
                    "node": n,
                    "zone": z,
                    "weight": float(gsk[ni, zi]),
                })
    pd.DataFrame(gsk_long).to_parquet(fb_dir / "gsk_long.parquet")

    ram_rows = []
    for r in sorted(results_ok, key=lambda x: x["t"]):
        for k, line_name in enumerate(r["fb"]["CNEC"]):
            ram_rows.append({
                "t": r["t"],
                "cnec": line_name,
                "cnec_idx": int(r["fb"]["CNEC_IDX"][k]),
                "ram_pos": float(r["fb"]["RAM_POS"][k]),
                "ram_neg": float(r["fb"]["RAM_NEG"][k]),
            })
    pd.DataFrame(ram_rows).to_parquet(fb_dir / "ram_long.parquet")

    ptdf_rows = []
    for r in sorted(results_ok, key=lambda x: x["t"]):
        mat = r["fb"]["PTDF_Z_CNEC"]
        cnec_names = r["fb"]["CNEC"]
        cnec_idx = r["fb"]["CNEC_IDX"]

        for i, cnec_name in enumerate(cnec_names):
            for j, z in enumerate(Z_FBMC):
                ptdf_rows.append({
                    "t": r["t"],
                    "cnec": cnec_name,
                    "cnec_idx": int(cnec_idx[i]),
                    "zone": z,
                    "ptdf": float(mat[i, j]),
                })

    pd.DataFrame(ptdf_rows).to_parquet(fb_dir / "ptdf_z_cnec_long.parquet")

    # ---- D-1 MC ----
    d1_mc_dir = stage_dir / "d1_mc"
    ensure_dir(d1_mc_dir)

    export_pairs = results_ok[0]["meta"]["export_pairs"]
    export_cols = [f"{a}__to__{b}" for a, b in export_pairs]

    pd.DataFrame(
        [r["d1_mc"]["GEN"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=P
    ).to_parquet(d1_mc_dir / "gen.parquet")

    pd.DataFrame(
        [r["d1_mc"]["CURT"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d1_mc_dir / "curt.parquet")

    pd.DataFrame(
        [r["d1_mc"]["NP"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=Z_FBMC
    ).to_parquet(d1_mc_dir / "np.parquet")

    pd.DataFrame(
        [r["d1_mc"]["EXPORT"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=export_cols
    ).to_parquet(d1_mc_dir / "export.parquet")
    
    pd.DataFrame(
        [r["d1_mc"]["DUAL_POWER_BALANCE"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted,
        columns=Z_FBMC
    ).to_parquet(d1_mc_dir / "dual_power_balance.parquet")

    # ---- D-1 CGM ----
    d1_cgm_dir = stage_dir / "d1_cgm"
    ensure_dir(d1_cgm_dir)

    pd.DataFrame(
        [r["d1_cgm"]["DELTA"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d1_cgm_dir / "delta.parquet")

    pd.DataFrame(
        [r["d1_cgm"]["NOD_INJ"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d1_cgm_dir / "nod_inj.parquet")

    pd.DataFrame(
        [r["d1_cgm"]["LINE_F"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=L
    ).to_parquet(d1_cgm_dir / "line_f.parquet")

    pd.DataFrame(
        [r["d1_cgm"]["NP"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=Z_FBMC
    ).to_parquet(d1_cgm_dir / "np.parquet")

    if len(results_ok[0]["d1_cgm"]["EXPORT"]) == len(Z_not_in_FBMC):
        d1_cgm_export_cols = Z_not_in_FBMC
    else:
        d1_cgm_export_cols = [f"export_{i}" for i in range(len(results_ok[0]["d1_cgm"]["EXPORT"]))]

    pd.DataFrame(
        [r["d1_cgm"]["EXPORT"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=d1_cgm_export_cols
    ).to_parquet(d1_cgm_dir / "export.parquet")

    # ---- D-0 ----
    d0_dir = stage_dir / "d0"
    ensure_dir(d0_dir)

    pd.DataFrame(
        [r["d0"]["CURT_RD"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d0_dir / "curt_rd.parquet")

    pd.DataFrame(
        [r["d0"]["RD_POS"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=P_RD
    ).to_parquet(d0_dir / "rd_pos.parquet")

    pd.DataFrame(
        [r["d0"]["RD_NEG"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=P_RD
    ).to_parquet(d0_dir / "rd_neg.parquet")

    pd.DataFrame(
        [r["d0"]["DELTA"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d0_dir / "delta.parquet")

    pd.DataFrame(
        [r["d0"]["NOD_INJ"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=N
    ).to_parquet(d0_dir / "nod_inj.parquet")

    pd.DataFrame(
        [r["d0"]["LINE_F"] for r in sorted(results_ok, key=lambda x: x["t"])],
        index=time_sorted, columns=L
    ).to_parquet(d0_dir / "line_f.parquet")


###############################################################################
# MAIN
###############################################################################

def main():
    ensure_dir(RESULTS_ROOT)

    config = {
        "run_name": RUN_NAME,
        "n_workers": N_WORKERS,
        "gurobi_threads_per_worker": GUROBI_THREADS_PER_WORKER,
        "gsk_strategy": GSK_STRATEGY,
        "include_cb_lines": INCLUDE_CB_LINES,
        "cne_alpha": CNE_ALPHA,
        "cost_curt": COST_CURT,
        "cost_curt_mc": COST_CURT_MC,
        "max_ntc": MAX_NTC,
        "export_eps": EXPORT_EPS,
        "time_start": TIME_START,
        "time_end": TIME_END,
        "continue_on_error": CONTINUE_ON_ERROR,
        "frm": float(frm),
    }

    with open(RESULTS_ROOT / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    time_index = build_time_index()
    print(f"Running {len(time_index)} MTUs with {N_WORKERS} workers")
    
    global LODF, bad_k
    
    LODF, bad_k = compute_lodf_from_ptdf(
    df_branch=df_branch,
    PTDF_full=PTDF_full,
    N_idx=N_idx,
    L_idx=L_idx,
    L=L,
    )

    results = []
    failures = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(solve_single_mtu, t, LODF, bad_k): t for t in time_index}

        for fut in as_completed(futures):
            t = futures[fut]
            res = fut.result()

            if res["status"] == "ok":
                results.append(res)
                print(f"[OK] t={t}")
            else:
                failures.append(res)
                print(f"[FAIL] t={t} :: {res['error']}")
                if not CONTINUE_ON_ERROR:
                    raise RuntimeError(f"Stopping on failed MTU t={t}")

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_df.to_csv(RESULTS_ROOT / "failures.csv", index=False)

    if results:
        save_matrix_results(results, RESULTS_ROOT)
        print(f"Saved results to: {RESULTS_ROOT}")

    print(f"Finished. Success={len(results)}, Failed={len(failures)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
