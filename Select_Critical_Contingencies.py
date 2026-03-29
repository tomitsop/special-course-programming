import json
import math
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cvxpy as cp
import numpy as np
import pandas as pd

from input_data_base_functions import *
from Differentiable_D_2_SCOPF_CGM import (
    build_d2_cgm_problem_components,
    compute_lodf_from_ptdf,
)
from Basecase_Dynamic_GSK_Definition_pipeline import (
    GSKStrategyManager,
    compute_selected_post_contingency_line_flows,
)

###############################################################################
# CONFIG
###############################################################################

N_WORKERS = 12
GUROBI_THREADS_PER_WORKER = 1

# GSK strategy used for the offline contingency-selection study.
# The final reduced basecase / DFL pipelines can rebuild PTDF rows with the
# same selected contingency subset and any chosen GSK logic.
GSK_STRATEGY = "pmax_sub"

# Adjust line capacities for contingency selection to avod infeasibilities
df_branch["Pmax"] = 0.85 * df_branch["Pmax"]

# Must stay in N-1 mode for contingency selection.
FBMC_MODE = "n1"
INCLUDE_BASECASE_IN_N1 = True
INCLUDE_CB_LINES = True

FRM_VALUE = float(frm)
CNE_ALPHA = float(cne_alpha)

# Optional MTU slice
TIME_START = None
TIME_END = None

# Optimization constants
COST_CURT = math.ceil(find_maximum_mc())
COST_CURT_MC = 0.0
MAX_NTC = 1000.0
EXPORT_EPS = 1e-7

# Screening / row generation
SCREENING_MAX_ITERS = 15
SCREENING_VIOL_TOL = 1e-6
SCREENING_BIND_TOL = 1e-5
NEAR_BIND_TOL = SCREENING_BIND_TOL

# Failure handling
CONTINUE_ON_ERROR = True

# Final selection rule
# If TOP_K_CONTINGENCIES is None, threshold-based filtering is used.
TOP_K_CONTINGENCIES = 25
MIN_BINDING_FREQ = None      # e.g. 0.02
MIN_ACTIVE_FREQ = None       # e.g. 0.10

# Distinct output tree so it never mixes with the normal pipeline results.
RESULTS_ROOT = Path("results") / "critical_contingency_selection" / GSK_STRATEGY


###############################################################################
# GLOBALS FOR WORKERS
###############################################################################

GLOBAL_LODF = None
GLOBAL_BAD_K = None


###############################################################################
# HELPERS
###############################################################################

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_time_index():
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
    return np.array(line_cap, dtype=np.float64)


def compute_ram_from_d2_numpy(line_f_d2_cnec: np.ndarray, line_cap_cnec: np.ndarray):
    base = line_cap_cnec * (1.0 - FRM_VALUE)
    ram_pos = base - line_f_d2_cnec
    ram_neg = -base - line_f_d2_cnec
    return ram_pos.astype(np.float64), ram_neg.astype(np.float64)


def export_pairs_builder():
    return [(z, zz) for z in Z for zz in z_to_z[z]]


def build_line_f_d2_for_fb_rows(
    *,
    line_f_d2: np.ndarray,
    cnec_idx_t,
    contingency_idx_t,
    lodf: np.ndarray,
) -> np.ndarray:
    line_f_d2 = np.asarray(line_f_d2, dtype=np.float64).reshape(-1)
    cnec_idx_arr = np.asarray(cnec_idx_t, dtype=np.int64).reshape(-1)
    contingency_idx_arr = np.asarray(contingency_idx_t, dtype=np.int64).reshape(-1)

    if cnec_idx_arr.shape[0] != contingency_idx_arr.shape[0]:
        raise ValueError("cnec_idx_t and contingency_idx_t must have the same length.")

    out = np.zeros(cnec_idx_arr.shape[0], dtype=np.float64)
    unique_k = np.unique(contingency_idx_arr)

    for k in unique_k:
        mask = contingency_idx_arr == k
        monitored_idx = cnec_idx_arr[mask].tolist()

        if int(k) == -1:
            out[mask] = line_f_d2[cnec_idx_arr[mask]]
        else:
            out[mask] = compute_selected_post_contingency_line_flows(
                line_f_base=line_f_d2,
                lodf=lodf,
                monitored_idx=monitored_idx,
                contingency_idx=int(k),
            )

    return out


def subset_fb_rows(
    *,
    ptdf_z_cnec_all: np.ndarray,
    cnec_all,
    cnec_idx_all,
    contingency_all,
    contingency_idx_all,
    line_f_d2_cnec_all: np.ndarray,
    ram_pos_all: np.ndarray,
    ram_neg_all: np.ndarray,
    selected_idx,
):
    idx = np.asarray(selected_idx, dtype=np.int64)
    return {
        "PTDF_Z_CNEC": ptdf_z_cnec_all[idx, :],
        "CNEC": [cnec_all[i] for i in idx],
        "CNEC_IDX": np.asarray(cnec_idx_all, dtype=np.int64)[idx],
        "CONTINGENCY": [contingency_all[i] for i in idx],
        "CONTINGENCY_IDX": np.asarray(contingency_idx_all, dtype=np.int64)[idx],
        "LINE_F_D2_CNEC": np.asarray(line_f_d2_cnec_all, dtype=np.float64)[idx],
        "RAM_POS": np.asarray(ram_pos_all, dtype=np.float64)[idx],
        "RAM_NEG": np.asarray(ram_neg_all, dtype=np.float64)[idx],
        "ACTIVE_IDX_IN_FULL": idx,
    }


def compute_constraint_flows_all(
    ptdf_z_cnec_all: np.ndarray,
    np_solution: np.ndarray,
    np_d2_fb: np.ndarray,
) -> np.ndarray:
    delta_np = np.asarray(np_solution, dtype=np.float64) - np.asarray(np_d2_fb, dtype=np.float64)
    return np.asarray(ptdf_z_cnec_all, dtype=np.float64) @ delta_np


def find_new_violated_rows(
    *,
    flows_all: np.ndarray,
    ram_pos_all: np.ndarray,
    ram_neg_all: np.ndarray,
    active_mask: np.ndarray,
    viol_tol: float,
):
    upper_violation = flows_all - ram_pos_all
    lower_violation = ram_neg_all - flows_all
    violated_mask = (upper_violation > viol_tol) | (lower_violation > viol_tol)
    new_mask = violated_mask & (~active_mask)
    new_idx = np.where(new_mask)[0].tolist()
    return new_idx, upper_violation, lower_violation


def identify_binding_rows(
    *,
    flows_active: np.ndarray,
    ram_pos_active: np.ndarray,
    ram_neg_active: np.ndarray,
    bind_tol: float,
):
    upper_slack = ram_pos_active - flows_active
    lower_slack = flows_active - ram_neg_active
    binding_mask = (upper_slack <= bind_tol) | (lower_slack <= bind_tol)
    return binding_mask, upper_slack, lower_slack


def build_d1_mc_problem_components_runtime(
    *,
    ptdf_z_cnec_t: np.ndarray,
    n_constraints: int,
    cost_curt_mc=0.0,
    max_ntc=1000.0,
    export_eps=1e-7,
):
    nP = len(P)
    nN = len(N)
    nZ_fb = len(Z_FBMC)
    nCNE = int(n_constraints)

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

    constraints += [GEN >= 0, GEN <= gmax]
    constraints += [CURT >= 0, CURT <= renew]
    constraints += [EXPORT >= 0, EXPORT <= max_ntc]

    for z in Z_FBMC:
        zfi = Z_fb_idx[z]
        nodes = [N_idx[n] for n in n_in_z[z]]
        plants = [P_idx[p] for p in p_in_z[z]]
        zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]

        exports_out = [EXPORT[export_idx[(z, zz)]] for zz in zz_list]
        exports_in = [EXPORT[export_idx[(zz, z)]] for zz in zz_list if (zz, z) in export_idx]

        gen_sum = cp.sum(GEN[plants]) if plants else 0
        ren_sum = cp.sum(renew[nodes]) if nodes else 0
        curt_sum = cp.sum(CURT[nodes]) if nodes else 0
        dem_sum = cp.sum(dem[nodes]) if nodes else 0

        constraints += [
            gen_sum + ren_sum - curt_sum - cp.sum(exports_out) + cp.sum(exports_in) - NP[zfi] == dem_sum
        ]

    for z in Z_not_in_FBMC:
        nodes = [N_idx[n] for n in n_in_z[z]]
        plants = [P_idx[p] for p in p_in_z[z]]
        zz_list = z_to_z[z]

        exports_out = [EXPORT[export_idx[(z, zz)]] for zz in zz_list]
        exports_in = [EXPORT[export_idx[(zz, z)]] for zz in zz_list if (zz, z) in export_idx]

        gen_sum = cp.sum(GEN[plants]) if plants else 0
        ren_sum = cp.sum(renew[nodes]) if nodes else 0
        curt_sum = cp.sum(CURT[nodes]) if nodes else 0
        dem_sum = cp.sum(dem[nodes]) if nodes else 0

        constraints += [
            gen_sum + ren_sum - curt_sum - cp.sum(exports_out) + cp.sum(exports_in) == dem_sum
        ]

    constraints += [cp.sum(NP) == 0]

    for j_idx in range(nCNE):
        ptdf_row = ptdf_z_cnec_t[j_idx, :]
        flow_expr = ptdf_row @ (NP - np_d2_fb)
        constraints += [flow_expr <= ram_pos[j_idx], flow_expr >= ram_neg[j_idx]]

    objective = cp.Minimize(cost_gen @ GEN + cost_curt_mc * cp.sum(CURT) + export_eps * cp.sum(EXPORT))
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
    }


def build_gsk_payload_for_t(
    strategy: str,
    gsk_manager: GSKStrategyManager,
    gen_d2_np: np.ndarray,
    lodf: np.ndarray,
    bad_k: np.ndarray,
):
    gen_d2_series = pd.Series(gen_d2_np, index=P)
    return gsk_manager.build_for_t(
        strategy=strategy,
        df_d2_gen=gen_d2_series,
        lodf=lodf,
        bad_k=bad_k,
        fbmc_mode=FBMC_MODE,
        include_basecase=INCLUDE_BASECASE_IN_N1,
    )


def solve_d1_mc_with_screening(
    *,
    dem_np: np.ndarray,
    renew_np: np.ndarray,
    np_d2_fb: np.ndarray,
    ptdf_z_cnec_all: np.ndarray,
    cnec_all,
    cnec_idx_all,
    contingency_all,
    contingency_idx_all,
    line_f_d2_cnec_all: np.ndarray,
    ram_pos_all: np.ndarray,
    ram_neg_all: np.ndarray,
    gurobi_opts: dict,
):
    n_total = int(ptdf_z_cnec_all.shape[0])
    contingency_idx_arr = np.asarray(contingency_idx_all, dtype=np.int64)
    initial_active_idx = np.where(contingency_idx_arr == -1)[0].tolist()

    if len(initial_active_idx) == 0:
        raise RuntimeError(
            "Screening requested but no base-case rows found. "
            "Set INCLUDE_BASECASE_IN_N1=True or change initialization logic."
        )

    active_idx = sorted(set(initial_active_idx))
    iteration_log = []
    final_np = None
    final_obj = None

    for it in range(1, SCREENING_MAX_ITERS + 1):
        active = subset_fb_rows(
            ptdf_z_cnec_all=ptdf_z_cnec_all,
            cnec_all=cnec_all,
            cnec_idx_all=cnec_idx_all,
            contingency_all=contingency_all,
            contingency_idx_all=contingency_idx_all,
            line_f_d2_cnec_all=line_f_d2_cnec_all,
            ram_pos_all=ram_pos_all,
            ram_neg_all=ram_neg_all,
            selected_idx=active_idx,
        )

        d1_mc = build_d1_mc_problem_components_runtime(
            ptdf_z_cnec_t=active["PTDF_Z_CNEC"],
            n_constraints=len(active_idx),
            cost_curt_mc=COST_CURT_MC,
            max_ntc=MAX_NTC,
            export_eps=EXPORT_EPS,
        )
        d1_mc["parameters"]["dem"].value = dem_np
        d1_mc["parameters"]["renew"].value = renew_np
        d1_mc["parameters"]["np_d2_fb"].value = np_d2_fb
        d1_mc["parameters"]["ram_pos"].value = active["RAM_POS"]
        d1_mc["parameters"]["ram_neg"].value = active["RAM_NEG"]

        d1_mc["problem"].solve(solver=cp.GUROBI, ignore_dpp=True, **gurobi_opts)
        if d1_mc["problem"].status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-1 MC infeasible/status={d1_mc['problem'].status}")

        np_sol = np.array(d1_mc["variables"]["NP"].value).reshape(-1)
        flows_all = compute_constraint_flows_all(
            ptdf_z_cnec_all=ptdf_z_cnec_all,
            np_solution=np_sol,
            np_d2_fb=np_d2_fb,
        )

        active_mask = np.zeros(n_total, dtype=bool)
        active_mask[np.asarray(active_idx, dtype=np.int64)] = True

        new_idx, upper_violation, lower_violation = find_new_violated_rows(
            flows_all=flows_all,
            ram_pos_all=ram_pos_all,
            ram_neg_all=ram_neg_all,
            active_mask=active_mask,
            viol_tol=SCREENING_VIOL_TOL,
        )

        iteration_log.append(
            {
                "iteration": it,
                "active_before_add": int(len(active_idx)),
                "new_added": int(len(new_idx)),
                "active_after_add": int(len(active_idx) + len(new_idx)),
                "max_upper_violation": float(np.max(upper_violation)) if upper_violation.size else 0.0,
                "max_lower_violation": float(np.max(lower_violation)) if lower_violation.size else 0.0,
            }
        )

        final_np = np_sol
        final_obj = float(d1_mc["problem"].value)

        if len(new_idx) == 0:
            break
        active_idx = sorted(set(active_idx).union(new_idx))

    if final_np is None:
        raise RuntimeError("D-1 MC screening loop did not produce a solution.")

    final_active = subset_fb_rows(
        ptdf_z_cnec_all=ptdf_z_cnec_all,
        cnec_all=cnec_all,
        cnec_idx_all=cnec_idx_all,
        contingency_all=contingency_all,
        contingency_idx_all=contingency_idx_all,
        line_f_d2_cnec_all=line_f_d2_cnec_all,
        ram_pos_all=ram_pos_all,
        ram_neg_all=ram_neg_all,
        selected_idx=active_idx,
    )

    final_flows_active = compute_constraint_flows_all(
        ptdf_z_cnec_all=final_active["PTDF_Z_CNEC"],
        np_solution=final_np,
        np_d2_fb=np_d2_fb,
    )

    binding_mask, upper_slack, lower_slack = identify_binding_rows(
        flows_active=final_flows_active,
        ram_pos_active=final_active["RAM_POS"],
        ram_neg_active=final_active["RAM_NEG"],
        bind_tol=SCREENING_BIND_TOL,
    )

    return {
        "objective": final_obj,
        "active": final_active,
        "screening": {
            "n_total_candidates": int(n_total),
            "n_initial_active": int(len(initial_active_idx)),
            "n_final_active": int(len(active_idx)),
            "n_binding_final": int(np.sum(binding_mask)),
            "iterations": iteration_log,
            "binding_mask_final_active": binding_mask.astype(bool),
            "upper_slack_final_active": upper_slack.astype(np.float64),
            "lower_slack_final_active": lower_slack.astype(np.float64),
            "flows_final_active": final_flows_active.astype(np.float64),
        },
    }


def build_contingency_metrics_for_mtu(
    *,
    t: int,
    active_contingency_idx: np.ndarray,
    binding_mask: np.ndarray,
    upper_slack: np.ndarray,
    lower_slack: np.ndarray,
):
    df = pd.DataFrame(
        {
            "contingency_idx": np.asarray(active_contingency_idx, dtype=np.int64),
            "binding": np.asarray(binding_mask, dtype=bool),
            "upper_slack": np.asarray(upper_slack, dtype=np.float64),
            "lower_slack": np.asarray(lower_slack, dtype=np.float64),
        }
    )
    df = df[df["contingency_idx"] >= 0].copy()  # exclude basecase rows from contingency ranking
    if df.empty:
        return pd.DataFrame(
            columns=[
                "t", "contingency_idx", "active_rows", "binding_rows",
                "has_binding", "has_near_binding", "min_upper_slack",
                "min_lower_slack", "min_abs_slack",
            ]
        )

    df["min_side_slack"] = np.minimum(df["upper_slack"], df["lower_slack"])
    df["near_binding"] = df["min_side_slack"] <= NEAR_BIND_TOL

    grouped = (
        df.groupby("contingency_idx", sort=False)
        .agg(
            active_rows=("contingency_idx", "size"),
            binding_rows=("binding", "sum"),
            near_binding_rows=("near_binding", "sum"),
            min_upper_slack=("upper_slack", "min"),
            min_lower_slack=("lower_slack", "min"),
            min_abs_slack=("min_side_slack", "min"),
        )
        .reset_index()
    )
    grouped["t"] = int(t)
    grouped["has_binding"] = grouped["binding_rows"] > 0
    grouped["has_near_binding"] = grouped["near_binding_rows"] > 0
    return grouped[
        [
            "t", "contingency_idx", "active_rows", "binding_rows",
            "has_binding", "has_near_binding", "min_upper_slack",
            "min_lower_slack", "min_abs_slack",
        ]
    ]


def solve_single_mtu(t: int):
    try:
        if GLOBAL_LODF is None or GLOBAL_BAD_K is None:
            raise RuntimeError("GLOBAL_LODF / GLOBAL_BAD_K not initialized in worker.")

        gurobi_opts = {"Threads": GUROBI_THREADS_PER_WORKER, "OutputFlag": 0}
        dem_np, renew_np = build_dem_renew_np(t)
        line_cap_np = build_line_cap_np(t)

        gsk_manager = GSKStrategyManager(cne_alpha=CNE_ALPHA, include_cb_lines=INCLUDE_CB_LINES)

        # 1) D-2 CGM
        objective, constraints, params_list, vars_list, _, params, vars_ = build_d2_cgm_problem_components(
            cost_curt=COST_CURT,
            frm=FRM_VALUE,
            max_ntc=MAX_NTC,
            preventive=False,
            LODF=None,
            bad_k=None,
        )
        problem = cp.Problem(objective, constraints)
        params["dem"].value = dem_np
        params["renew"].value = renew_np
        params["line_cap"].value = line_cap_np
        problem.solve(solver=cp.GUROBI, **gurobi_opts)

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-2 infeasible/status={problem.status}")

        gen_d2 = np.array(vars_["GEN"].value).reshape(-1)
        np_d2_fb = np.array(vars_["NP"].value).reshape(-1)
        line_f_d2 = np.array(vars_["LINE_F"].value).reshape(-1)

        # 2) Full candidate CNEC pool
        gsk_payload = build_gsk_payload_for_t(
            strategy=GSK_STRATEGY,
            gsk_manager=gsk_manager,
            gen_d2_np=gen_d2,
            lodf=GLOBAL_LODF,
            bad_k=GLOBAL_BAD_K,
        )

        cnec_all = list(gsk_payload["cnec"])
        cnec_idx_all = np.array(gsk_payload["cnec_idx"], dtype=np.int64)
        ptdf_z_cnec_all = np.array(gsk_payload["ptdf_z_cnec"], dtype=np.float64)
        contingency_all = list(gsk_payload.get("contingency", ["basecase"] * len(cnec_all)))
        contingency_idx_all = np.array(gsk_payload.get("contingency_idx", [-1] * len(cnec_all)), dtype=np.int64)

        if ptdf_z_cnec_all.shape[0] != len(cnec_all):
            raise ValueError("Mismatch between PTDF_Z_CNEC rows and CNEC metadata length.")

        # 3) Full candidate RAM vectors
        line_f_d2_cnec_all = build_line_f_d2_for_fb_rows(
            line_f_d2=line_f_d2,
            cnec_idx_t=cnec_idx_all,
            contingency_idx_t=contingency_idx_all,
            lodf=GLOBAL_LODF,
        )
        line_cap_cnec_all = line_cap_np[cnec_idx_all]
        ram_pos_all, ram_neg_all = compute_ram_from_d2_numpy(
            line_f_d2_cnec=line_f_d2_cnec_all,
            line_cap_cnec=line_cap_cnec_all,
        )

        # 4) D-1 MC with screening
        d1_mc_screened = solve_d1_mc_with_screening(
            dem_np=dem_np,
            renew_np=renew_np,
            np_d2_fb=np_d2_fb,
            ptdf_z_cnec_all=ptdf_z_cnec_all,
            cnec_all=cnec_all,
            cnec_idx_all=cnec_idx_all,
            contingency_all=contingency_all,
            contingency_idx_all=contingency_idx_all,
            line_f_d2_cnec_all=line_f_d2_cnec_all,
            ram_pos_all=ram_pos_all,
            ram_neg_all=ram_neg_all,
            gurobi_opts=gurobi_opts,
        )

        active_fb = d1_mc_screened["active"]
        screening_info = d1_mc_screened["screening"]

        contingency_metrics = build_contingency_metrics_for_mtu(
            t=t,
            active_contingency_idx=active_fb["CONTINGENCY_IDX"],
            binding_mask=screening_info["binding_mask_final_active"],
            upper_slack=screening_info["upper_slack_final_active"],
            lower_slack=screening_info["lower_slack_final_active"],
        )

        return {
            "status": "ok",
            "t": int(t),
            "d2_objective": float(problem.value),
            "d1_mc_objective": float(d1_mc_screened["objective"]),
            "fb": {
                "SCREENING": screening_info,
            },
            "contingency_metrics": contingency_metrics,
        }

    except Exception as e:
        return {
            "status": "fail",
            "t": int(t),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


###############################################################################
# AGGREGATION / SAVING
###############################################################################

def aggregate_contingency_metrics(per_mtu_frames, n_successful_mtus: int) -> pd.DataFrame:
    if not per_mtu_frames:
        return pd.DataFrame(
            columns=[
                "contingency_idx", "mtu_active_count", "mtu_binding_count",
                "mtu_near_binding_count", "active_frequency", "binding_frequency",
                "near_binding_frequency", "avg_active_rows_when_active",
                "avg_binding_rows_when_active", "best_min_abs_slack",
                "mean_min_abs_slack_when_active", "selection_score",
            ]
        )

    all_df = pd.concat(per_mtu_frames, ignore_index=True)

    agg = (
        all_df.groupby("contingency_idx", sort=False)
        .agg(
            mtu_active_count=("t", "nunique"),
            mtu_binding_count=("has_binding", "sum"),
            mtu_near_binding_count=("has_near_binding", "sum"),
            avg_active_rows_when_active=("active_rows", "mean"),
            avg_binding_rows_when_active=("binding_rows", "mean"),
            best_min_abs_slack=("min_abs_slack", "min"),
            mean_min_abs_slack_when_active=("min_abs_slack", "mean"),
        )
        .reset_index()
    )

    denom = max(int(n_successful_mtus), 1)
    agg["active_frequency"] = agg["mtu_active_count"] / denom
    agg["binding_frequency"] = agg["mtu_binding_count"] / denom
    agg["near_binding_frequency"] = agg["mtu_near_binding_count"] / denom

    # Main ranking metric: binding frequency first, active frequency second.
    agg["selection_score"] = agg["binding_frequency"] 

    agg = agg.sort_values(
        ["selection_score", "binding_frequency", "active_frequency", "best_min_abs_slack"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return agg


def select_final_contingencies(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    selected = summary_df.copy()

    if MIN_BINDING_FREQ is not None:
        selected = selected[selected["binding_frequency"] >= float(MIN_BINDING_FREQ)]

    if MIN_ACTIVE_FREQ is not None:
        selected = selected[selected["active_frequency"] >= float(MIN_ACTIVE_FREQ)]

    if TOP_K_CONTINGENCIES is not None:
        selected = selected.head(int(TOP_K_CONTINGENCIES))

    return selected.reset_index(drop=True)


def save_outputs(
    *,
    results_root: Path,
    config: dict,
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    per_mtu_summary_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    failures_df: pd.DataFrame | None,
):
    ensure_dir(results_root)

    with open(results_root / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary_df.to_csv(results_root / "contingency_metrics_summary.csv", index=False)
    selected_df.to_csv(results_root / "selected_contingencies.csv", index=False)
    per_mtu_summary_df.to_csv(results_root / "per_mtu_contingency_metrics.csv", index=False)
    run_summary_df.to_csv(results_root / "mtu_run_summary.csv", index=False)

    with open(results_root / "selected_contingency_idx.json", "w") as f:
        json.dump(selected_df["contingency_idx"].astype(int).tolist(), f, indent=2)

    if failures_df is not None and not failures_df.empty:
        failures_df.to_csv(results_root / "failures.csv", index=False)


###############################################################################
# WORKER INIT
###############################################################################

def init_worker(lodf, bad_k):
    global GLOBAL_LODF, GLOBAL_BAD_K
    GLOBAL_LODF = np.asarray(lodf, dtype=np.float64)
    GLOBAL_BAD_K = np.asarray(bad_k, dtype=bool)


###############################################################################
# MAIN
###############################################################################

def main():
    ensure_dir(RESULTS_ROOT)

    config = {
        "n_workers": N_WORKERS,
        "gurobi_threads_per_worker": GUROBI_THREADS_PER_WORKER,
        "gsk_strategy": GSK_STRATEGY,
        "fbmc_mode": FBMC_MODE,
        "include_basecase_in_n1": INCLUDE_BASECASE_IN_N1,
        "include_cb_lines": INCLUDE_CB_LINES,
        "cne_alpha": CNE_ALPHA,
        "frm": FRM_VALUE,
        "cost_curt": COST_CURT,
        "cost_curt_mc": COST_CURT_MC,
        "max_ntc": MAX_NTC,
        "export_eps": EXPORT_EPS,
        "time_start": TIME_START,
        "time_end": TIME_END,
        "continue_on_error": CONTINUE_ON_ERROR,
        "screening_max_iters": SCREENING_MAX_ITERS,
        "screening_viol_tol": SCREENING_VIOL_TOL,
        "screening_bind_tol": SCREENING_BIND_TOL,
        "near_bind_tol": NEAR_BIND_TOL,
        "top_k_contingencies": TOP_K_CONTINGENCIES,
        "min_binding_freq": MIN_BINDING_FREQ,
        "min_active_freq": MIN_ACTIVE_FREQ,
        "results_root": str(RESULTS_ROOT),
    }

    time_index = build_time_index()
    print(f"Running critical contingency selection on {len(time_index)} MTUs with {N_WORKERS} workers")

    lodf, bad_k = compute_lodf_from_ptdf(
        df_branch=df_branch,
        PTDF_full=PTDF_full,
        N_idx=N_idx,
        L_idx=L_idx,
        L=L,
    )

    results = []
    failures = []

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=init_worker,
        initargs=(lodf, bad_k),
    ) as executor:
        futures = {executor.submit(solve_single_mtu, t): t for t in time_index}

        for fut in as_completed(futures):
            t = futures[fut]
            res = fut.result()

            if res["status"] == "ok":
                results.append(res)
                s = res["fb"]["SCREENING"]
                print(
                    f"[OK] t={t} | total={s['n_total_candidates']} "
                    f"| init={s['n_initial_active']} "
                    f"| final={s['n_final_active']} "
                    f"| binding={s['n_binding_final']} "
                    f"| iters={len(s['iterations'])}"
                )
                for it_row in s["iterations"]:
                    print(
                        f"    iter={it_row['iteration']} "
                        f"active_before={it_row['active_before_add']} "
                        f"added={it_row['new_added']} "
                        f"active_after={it_row['active_after_add']} "
                        f"max_up_viol={it_row['max_upper_violation']:.6g} "
                        f"max_low_viol={it_row['max_lower_violation']:.6g}"
                    )
            else:
                failures.append(res)
                print(f"[FAIL] t={t} :: {res['error']}")
                if not CONTINUE_ON_ERROR:
                    raise RuntimeError(f"Stopping on failed MTU t={t}")

    n_success = len(results)
    if n_success == 0:
        raise RuntimeError("No successful MTUs were solved. No contingency ranking can be produced.")

    per_mtu_frames = [res["contingency_metrics"] for res in results if not res["contingency_metrics"].empty]
    per_mtu_summary_df = (
        pd.concat(per_mtu_frames, ignore_index=True)
        if per_mtu_frames else
        pd.DataFrame(columns=[
            "t", "contingency_idx", "active_rows", "binding_rows", "has_binding",
            "has_near_binding", "min_upper_slack", "min_lower_slack", "min_abs_slack",
        ])
    )

    summary_df = aggregate_contingency_metrics(per_mtu_frames, n_successful_mtus=n_success)
    selected_df = select_final_contingencies(summary_df)

    run_summary_rows = []
    for res in results:
        s = res["fb"]["SCREENING"]
        run_summary_rows.append(
            {
                "t": res["t"],
                "d2_objective": res["d2_objective"],
                "d1_mc_objective": res["d1_mc_objective"],
                "n_total_candidates": s["n_total_candidates"],
                "n_initial_active": s["n_initial_active"],
                "n_final_active": s["n_final_active"],
                "n_binding_final": s["n_binding_final"],
                "n_screening_iterations": len(s["iterations"]),
            }
        )
    run_summary_df = pd.DataFrame(run_summary_rows).sort_values("t").reset_index(drop=True)

    failures_df = pd.DataFrame(failures) if failures else pd.DataFrame()

    save_outputs(
        results_root=RESULTS_ROOT,
        config=config,
        summary_df=summary_df,
        selected_df=selected_df,
        per_mtu_summary_df=per_mtu_summary_df,
        run_summary_df=run_summary_df,
        failures_df=failures_df,
    )

    print(f"Saved contingency-selection outputs to: {RESULTS_ROOT}")
    print(f"Successful MTUs: {n_success} / {len(time_index)}")
    print(f"Selected contingencies: {len(selected_df)}")
    if not selected_df.empty:
        print("Top selected contingency indices:", selected_df["contingency_idx"].astype(int).tolist()[:10])


if __name__ == "__main__":
    main()
