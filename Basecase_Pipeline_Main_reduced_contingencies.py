import json
import math
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cvxpy as cp
import numpy as np
import pandas as pd

from input_data_base_functions import *

# D-2
from Differentiable_D_2_SCOPF_CGM import (
    build_d2_cgm_problem_components,
    compute_lodf_from_ptdf,
)

# D-1 CGM
from Differentiable_D_1_CGM import build_d1_cgm_problem_components

# D-0 CM
from D_0_Congestion_Management import build_d0_redispatch_problem_components

# GSK utilities (updated contingency-aware module)
from Basecase_Dynamic_GSK_Definition_pipeline import (
    GSKStrategyManager,
    compute_selected_post_contingency_line_flows,
)

###############################################################################
# CONFIG
###############################################################################

N_WORKERS = 10
GUROBI_THREADS_PER_WORKER = 1

# Choose one:
# "flat", "flat_unit", "pmax", "pmax_sub", "dynamic_headroom", "dynamic_gen"
GSK_STRATEGY = "pmax_sub"

df_branch["Pmax"] = 0.85 * df_branch["Pmax"]

# Keep N-1 mode but with a reduced contingency subset loaded from file.
FBMC_MODE = "n1"
INCLUDE_BASECASE_IN_N1 = True
INCLUDE_CB_LINES = True

FRM_VALUE = float(frm)
CNE_ALPHA = float(cne_alpha)

# Sensitivity-study organization:
#   "GSK"  -> vary GSK strategy, keep FRM and CNE alpha at reference values
#   "FRM"  -> vary FRM, keep GSK and CNE alpha at reference values
#   "CNE"  -> vary CNE alpha, keep GSK and FRM at reference values
#   "MISC" -> fallback custom organization
STUDY_DIMENSION = "GSK"

# Reference values used in parent-folder naming
REFERENCE_GSK = "pmax_sub"
REFERENCE_FRM = 0.05
REFERENCE_CNE_ALPHA = 0.05

# Optional time selection
TIME_START = None
TIME_END = None

# Optimization constants
COST_CURT = math.ceil(find_maximum_mc())
COST_CURT_MC = 0.0
MAX_NTC = 1000.0
EXPORT_EPS = 1e-7

# Robust run behavior
CONTINUE_ON_ERROR = True

# Reduced contingency selection artifact
SELECTED_CONTINGENCIES_ROOT = Path("results") / "critical_contingency_selection" / GSK_STRATEGY
SELECTED_CONTINGENCIES_FILE = SELECTED_CONTINGENCIES_ROOT / "selected_contingency_idx.json"
KEEP_BASECASE_ROWS = True


def fmt_float(x):
    return str(float(x)).replace(".", "p")


if STUDY_DIMENSION == "GSK":
    RUN_NAME = f"pipeline_GSK_frm-{fmt_float(REFERENCE_FRM)}_alpha-{fmt_float(REFERENCE_CNE_ALPHA)}"
    RESULTS_ROOT = Path("results") / RUN_NAME / f"{GSK_STRATEGY}_reduced_contingencies"

elif STUDY_DIMENSION == "FRM":
    RUN_NAME = f"pipeline_FRM_gsk-{REFERENCE_GSK}_alpha-{fmt_float(REFERENCE_CNE_ALPHA)}"
    RESULTS_ROOT = Path("results") / RUN_NAME / f"frm-{fmt_float(FRM_VALUE)}_reduced_contingencies"

elif STUDY_DIMENSION == "CNE":
    RUN_NAME = f"pipeline_CNE_gsk-{REFERENCE_GSK}_frm-{fmt_float(REFERENCE_FRM)}"
    RESULTS_ROOT = Path("results") / RUN_NAME / f"alpha-{fmt_float(CNE_ALPHA)}_reduced_contingencies"

else:
    RUN_NAME = "pipeline_misc"
    RESULTS_ROOT = Path("results") / RUN_NAME / (
        f"gsk-{GSK_STRATEGY}_frm-{fmt_float(FRM_VALUE)}_alpha-{fmt_float(CNE_ALPHA)}_reduced_contingencies"
    )


###############################################################################
# GLOBALS FOR WORKERS
###############################################################################

GLOBAL_LODF = None
GLOBAL_BAD_K = None
GLOBAL_SELECTED_CONTINGENCIES = None


###############################################################################
# HELPERS
###############################################################################


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)

    if isinstance(df.index, pd.MultiIndex):
        new_levels = []
        for level in df.index.levels:
            if getattr(level, "dtype", None) == object:
                new_levels.append(level.astype(str))
            else:
                new_levels.append(level)
        df.index = df.index.set_levels(new_levels)
    elif getattr(df.index, "dtype", None) == object:
        df.index = df.index.astype(str)

    if isinstance(df.columns, pd.MultiIndex):
        new_levels = []
        for level in df.columns.levels:
            if getattr(level, "dtype", None) == object:
                new_levels.append(level.astype(str))
            else:
                new_levels.append(level)
        df.columns = df.columns.set_levels(new_levels)
    elif getattr(df.columns, "dtype", None) == object:
        df.columns = df.columns.astype(str)

    return df



def save_parquet_safe(df: pd.DataFrame, path: Path, index: bool = True):
    make_parquet_safe(df).to_parquet(path, index=index)



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



def build_mc_rd_np():
    return np.array([get_mc(p) for p in P_RD], dtype=np.float64)



def build_gmax_rd_np():
    return np.array([get_gen_up(p) for p in P_RD], dtype=np.float64)



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

    for k in np.unique(contingency_idx_arr):
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



def filter_fb_rows_to_selected_contingencies(
    *,
    cnec_all,
    cnec_idx_all: np.ndarray,
    contingency_all,
    contingency_idx_all: np.ndarray,
    ptdf_z_cnec_all: np.ndarray,
    selected_contingencies: np.ndarray,
    keep_basecase_rows: bool,
):
    contingency_idx_all = np.asarray(contingency_idx_all, dtype=np.int64)
    cnec_idx_all = np.asarray(cnec_idx_all, dtype=np.int64)
    selected_contingencies = np.asarray(selected_contingencies, dtype=np.int64)

    keep_mask = np.isin(contingency_idx_all, selected_contingencies)
    if keep_basecase_rows:
        keep_mask |= contingency_idx_all == -1

    selected_idx = np.where(keep_mask)[0]

    if selected_idx.size == 0:
        raise RuntimeError(
            "No FB rows remain after contingency filtering. Check selected_contingency_idx.json and GSK settings."
        )

    return {
        "CNEC": [cnec_all[i] for i in selected_idx],
        "CNEC_IDX": cnec_idx_all[selected_idx],
        "CONTINGENCY": [contingency_all[i] for i in selected_idx],
        "CONTINGENCY_IDX": contingency_idx_all[selected_idx],
        "PTDF_Z_CNEC": np.asarray(ptdf_z_cnec_all, dtype=np.float64)[selected_idx, :],
        "ACTIVE_IDX_IN_FULL": selected_idx.astype(np.int64),
    }



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
    fbmc_balance_constraints = {}

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

        fbmc_balance_constraints[z] = (
            gen_sum + ren_sum - curt_sum
            - cp.sum(exports_out) + cp.sum(exports_in)
            - NP[zfi]
            == dem_sum
        )
        constraints += [fbmc_balance_constraints[z]]

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
        cost_gen @ GEN + cost_curt_mc * cp.sum(CURT) + export_eps * cp.sum(EXPORT)
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



def build_gsk_payload_for_t(
    strategy: str,
    gsk_manager: GSKStrategyManager,
    gen_d2_np: np.ndarray,
    lodf: np.ndarray,
    bad_k: np.ndarray,
    selected_contingencies=None,
):
    gen_d2_series = pd.Series(gen_d2_np, index=P)
    return gsk_manager.build_for_t(
        strategy=strategy,
        df_d2_gen=gen_d2_series,
        lodf=lodf,
        bad_k=bad_k,
        fbmc_mode=FBMC_MODE,
        include_basecase=INCLUDE_BASECASE_IN_N1,
        selected_contingencies=selected_contingencies,
    )


###############################################################################
# SINGLE MTU SOLVE
###############################################################################


def solve_single_mtu(t: int):
    try:
        if GLOBAL_LODF is None or GLOBAL_BAD_K is None or GLOBAL_SELECTED_CONTINGENCIES is None:
            raise RuntimeError("Worker globals not initialized.")

        gurobi_opts = {
            "Threads": GUROBI_THREADS_PER_WORKER,
            "OutputFlag": 0,
        }

        dem_np, renew_np = build_dem_renew_np(t)
        line_cap_np = build_line_cap_np(t)
        mc_rd_np = build_mc_rd_np()
        gmax_rd_np = build_gmax_rd_np()

        gsk_manager = GSKStrategyManager(
            cne_alpha=CNE_ALPHA,
            include_cb_lines=INCLUDE_CB_LINES,
        )

        # ------------------------------------------------------------------
        # 1) D-2 CGM
        # ------------------------------------------------------------------
        objective, constraints, _, _, _, params, vars_ = build_d2_cgm_problem_components(
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
        obj_d2 = problem.value

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-2 infeasible/status={problem.status}")

        gen_d2 = np.array(vars_["GEN"].value).reshape(-1)
        curt_d2 = np.array(vars_["CURT"].value).reshape(-1)
        np_d2_fb = np.array(vars_["NP"].value).reshape(-1)
        line_f_d2 = np.array(vars_["LINE_F"].value).reshape(-1)
        delta_d2 = np.array(vars_["DELTA"].value).reshape(-1)
        nod_inj_d2 = np.array(vars_["NOD_INJ"].value).reshape(-1)
        export_d2 = np.array(vars_["EXPORT"].value).reshape(-1)

        # ------------------------------------------------------------------
        # 2) Build full candidate pool then filter to selected contingencies
        # ------------------------------------------------------------------
        gsk_payload = build_gsk_payload_for_t(
            strategy=GSK_STRATEGY,
            gsk_manager=gsk_manager,
            gen_d2_np=gen_d2,
            lodf=GLOBAL_LODF,
            bad_k=GLOBAL_BAD_K,
            selected_contingencies=GLOBAL_SELECTED_CONTINGENCIES,
        )

        gsk_t = np.array(gsk_payload["gsk"], dtype=np.float64)

        cnec_all = list(gsk_payload["cnec"])
        cnec_idx_all = np.array(gsk_payload["cnec_idx"], dtype=np.int64)
        ptdf_z_cnec_all = np.array(gsk_payload["ptdf_z_cnec"], dtype=np.float64)

        contingency_all = (
            list(gsk_payload["contingency"])
            if "contingency" in gsk_payload
            else ["basecase"] * len(cnec_all)
        )
        contingency_idx_all = np.array(
            gsk_payload["contingency_idx"]
            if "contingency_idx" in gsk_payload
            else [-1] * len(cnec_all),
            dtype=np.int64,
        )

        if ptdf_z_cnec_all.shape[0] != len(cnec_all):
            raise ValueError("Mismatch between PTDF_Z_CNEC rows and CNEC metadata length.")

        filtered_fb = filter_fb_rows_to_selected_contingencies(
            cnec_all=cnec_all,
            cnec_idx_all=cnec_idx_all,
            contingency_all=contingency_all,
            contingency_idx_all=contingency_idx_all,
            ptdf_z_cnec_all=ptdf_z_cnec_all,
            selected_contingencies=GLOBAL_SELECTED_CONTINGENCIES,
            keep_basecase_rows=KEEP_BASECASE_ROWS,
        )

        cnec_t = filtered_fb["CNEC"]
        cnec_idx_t = filtered_fb["CNEC_IDX"]
        contingency_t = filtered_fb["CONTINGENCY"]
        contingency_idx_t = filtered_fb["CONTINGENCY_IDX"]
        ptdf_z_cnec_t = filtered_fb["PTDF_Z_CNEC"]

        # ------------------------------------------------------------------
        # 3) RAM vectors aligned with reduced FB rows
        # ------------------------------------------------------------------
        line_f_d2_cnec = build_line_f_d2_for_fb_rows(
            line_f_d2=line_f_d2,
            cnec_idx_t=cnec_idx_t,
            contingency_idx_t=contingency_idx_t,
            lodf=GLOBAL_LODF,
        )
        line_cap_cnec = line_cap_np[cnec_idx_t]
        ram_pos, ram_neg = compute_ram_from_d2_numpy(
            line_f_d2_cnec=line_f_d2_cnec,
            line_cap_cnec=line_cap_cnec,
        )

        # ------------------------------------------------------------------
        # 4) D-1 MC with reduced fixed FB row set
        # ------------------------------------------------------------------
        d1_mc = build_d1_mc_problem_components_runtime(
            ptdf_z_cnec_t=ptdf_z_cnec_t,
            n_constraints=len(cnec_t),
            cost_curt_mc=COST_CURT_MC,
            max_ntc=MAX_NTC,
            export_eps=EXPORT_EPS,
        )

        d1_mc["parameters"]["dem"].value = dem_np
        d1_mc["parameters"]["renew"].value = renew_np
        d1_mc["parameters"]["np_d2_fb"].value = np_d2_fb
        d1_mc["parameters"]["ram_pos"].value = ram_pos
        d1_mc["parameters"]["ram_neg"].value = ram_neg

        d1_mc["problem"].solve(solver=cp.GUROBI, ignore_dpp=True, **gurobi_opts)
        obj_d1_mc = d1_mc["problem"].value

        if d1_mc["problem"].status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"D-1 MC infeasible/status={d1_mc['problem'].status}")

        gen_d1 = np.array(d1_mc["variables"]["GEN"].value).reshape(-1)
        curt_d1 = np.array(d1_mc["variables"]["CURT"].value).reshape(-1)
        np_d1 = np.array(d1_mc["variables"]["NP"].value).reshape(-1)
        export_d1_mc = np.array(d1_mc["variables"]["EXPORT"].value).reshape(-1)
        dual_power_balance_d1_mc = np.array(
            [d1_mc["duals"]["power_balance"][z].dual_value for z in Z_FBMC],
            dtype=np.float64,
        )
        export_pairs = d1_mc["metadata"]["export_pairs"]

        # ------------------------------------------------------------------
        # 5) D-1 CGM
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 6) D-0 Congestion Management
        # ------------------------------------------------------------------
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
                "MODE": str(FBMC_MODE),
                "GSK": gsk_t,
                "CNEC": [str(x) for x in cnec_t],
                "CNEC_IDX": np.asarray(cnec_idx_t, dtype=np.int64),
                "CONTINGENCY": [str(x) for x in contingency_t],
                "CONTINGENCY_IDX": np.asarray(contingency_idx_t, dtype=np.int64),
                "RAM_POS": np.asarray(ram_pos, dtype=np.float64),
                "RAM_NEG": np.asarray(ram_neg, dtype=np.float64),
                "LINE_F_D2_CNEC": np.asarray(line_f_d2_cnec, dtype=np.float64),
                "PTDF_Z_CNEC": np.asarray(ptdf_z_cnec_t, dtype=np.float64),
                "N_TOTAL_FULL_ROWS": int(len(cnec_all)),
                "N_REDUCED_ROWS": int(len(cnec_t)),
                "N_SELECTED_CONTINGENCIES": int(len(np.asarray(GLOBAL_SELECTED_CONTINGENCIES, dtype=np.int64))),
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


def save_matrix_results(results_ok, stage_dir: Path):
    ensure_dir(stage_dir)

    results_sorted = sorted(results_ok, key=lambda x: x["t"])
    time_sorted = [r["t"] for r in results_sorted]

    obj_rows = [
        {
            "t": r["t"],
            "d2": r["objectives"]["d2"],
            "d1_mc": r["objectives"]["d1_mc"],
            "d1_cgm": r["objectives"]["d1_cgm"],
            "d0": r["objectives"]["d0"],
        }
        for r in results_sorted
    ]
    save_parquet_safe(pd.DataFrame(obj_rows).set_index("t"), stage_dir / "objectives.parquet")

    # ---- D-2 ----
    d2_dir = stage_dir / "d2"
    ensure_dir(d2_dir)
    save_parquet_safe(pd.DataFrame([r["d2"]["GEN"] for r in results_sorted], index=time_sorted, columns=P), d2_dir / "gen.parquet")
    save_parquet_safe(pd.DataFrame([r["d2"]["CURT"] for r in results_sorted], index=time_sorted, columns=N), d2_dir / "curt.parquet")
    save_parquet_safe(pd.DataFrame([r["d2"]["NP"] for r in results_sorted], index=time_sorted, columns=Z_FBMC), d2_dir / "np.parquet")
    save_parquet_safe(pd.DataFrame([r["d2"]["LINE_F"] for r in results_sorted], index=time_sorted, columns=L), d2_dir / "line_f.parquet")
    save_parquet_safe(pd.DataFrame([r["d2"]["DELTA"] for r in results_sorted], index=time_sorted, columns=N), d2_dir / "delta.parquet")
    save_parquet_safe(pd.DataFrame([r["d2"]["NOD_INJ"] for r in results_sorted], index=time_sorted, columns=N), d2_dir / "nod_inj.parquet")
    n_export = len(results_sorted[0]["d2"]["EXPORT"])
    export_cols = [f"export_{i}" for i in range(n_export)]
    save_parquet_safe(pd.DataFrame([r["d2"]["EXPORT"] for r in results_sorted], index=time_sorted, columns=export_cols), d2_dir / "export.parquet")

    # ---- Flow-Based ----
    fb_dir = stage_dir / "fb"
    ensure_dir(fb_dir)

    gsk_rows = []
    for r in results_sorted:
        gsk_mat = np.asarray(r["fb"]["GSK"], dtype=float)
        for i, n in enumerate(N_FBMC):
            for j, z in enumerate(Z_FBMC):
                gsk_rows.append({
                    "t": int(r["t"]),
                    "node": str(n),
                    "zone": str(z),
                    "gsk": float(gsk_mat[i, j]),
                })

    save_parquet_safe(pd.DataFrame(gsk_rows), fb_dir / "gsk_long.parquet", index=False)

    fb_summary_rows = []
    ram_rows = []
    ptdf_rows = []
    for r in results_sorted:
        fb_summary_rows.append({
            "t": r["t"],
            "n_total_full_rows": int(r["fb"]["N_TOTAL_FULL_ROWS"]),
            "n_reduced_rows": int(r["fb"]["N_REDUCED_ROWS"]),
            "n_selected_contingencies": int(r["fb"]["N_SELECTED_CONTINGENCIES"]),
        })
        mat = r["fb"]["PTDF_Z_CNEC"]
        cnec_names = r["fb"]["CNEC"]
        cnec_idx = r["fb"]["CNEC_IDX"]
        contingency_names = r["fb"]["CONTINGENCY"]
        contingency_idx = r["fb"]["CONTINGENCY_IDX"]
        ram_pos = r["fb"]["RAM_POS"]
        ram_neg = r["fb"]["RAM_NEG"]
        line_f_d2_cnec = r["fb"]["LINE_F_D2_CNEC"]

        for i in range(mat.shape[0]):
            ram_rows.append({
                "t": int(r["t"]),
                "cnec": str(cnec_names[i]),
                "cnec_idx": int(cnec_idx[i]),
                "contingency": str(contingency_names[i]),
                "contingency_idx": int(contingency_idx[i]),
                "line_f_d2_cnec": float(line_f_d2_cnec[i]),
                "ram_pos": float(ram_pos[i]),
                "ram_neg": float(ram_neg[i]),
            })
            for j, z in enumerate(Z_FBMC):
                ptdf_rows.append({
                    "t": int(r["t"]),
                    "cnec": str(cnec_names[i]),
                    "cnec_idx": int(cnec_idx[i]),
                    "contingency": str(contingency_names[i]),
                    "contingency_idx": int(contingency_idx[i]),
                    "zone": str(z),
                    "ptdf": float(mat[i, j]),
                })

    save_parquet_safe(pd.DataFrame(fb_summary_rows).set_index("t"), fb_dir / "fb_summary.parquet")
    save_parquet_safe(pd.DataFrame(ram_rows), fb_dir / "ram_long.parquet", index=False)
    save_parquet_safe(pd.DataFrame(ptdf_rows), fb_dir / "ptdf_z_cnec_long.parquet", index=False)

    # ---- D-1 MC ----
    d1_mc_dir = stage_dir / "d1_mc"
    ensure_dir(d1_mc_dir)
    save_parquet_safe(pd.DataFrame([r["d1_mc"]["GEN"] for r in results_sorted], index=time_sorted, columns=P), d1_mc_dir / "gen.parquet")
    save_parquet_safe(pd.DataFrame([r["d1_mc"]["CURT"] for r in results_sorted], index=time_sorted, columns=N), d1_mc_dir / "curt.parquet")
    save_parquet_safe(pd.DataFrame([r["d1_mc"]["NP"] for r in results_sorted], index=time_sorted, columns=Z_FBMC), d1_mc_dir / "np.parquet")
    d1mc_export_cols = [f"export_{i}" for i in range(len(results_sorted[0]["d1_mc"]["EXPORT"]))]
    save_parquet_safe(pd.DataFrame([r["d1_mc"]["EXPORT"] for r in results_sorted], index=time_sorted, columns=d1mc_export_cols), d1_mc_dir / "export.parquet")
    save_parquet_safe(pd.DataFrame([r["d1_mc"]["DUAL_POWER_BALANCE"] for r in results_sorted], index=time_sorted, columns=Z_FBMC), d1_mc_dir / "dual_power_balance.parquet")

    # ---- D-1 CGM ----
    d1_cgm_dir = stage_dir / "d1_cgm"
    ensure_dir(d1_cgm_dir)
    save_parquet_safe(pd.DataFrame([r["d1_cgm"]["DELTA"] for r in results_sorted], index=time_sorted, columns=N), d1_cgm_dir / "delta.parquet")
    save_parquet_safe(pd.DataFrame([r["d1_cgm"]["NOD_INJ"] for r in results_sorted], index=time_sorted, columns=N), d1_cgm_dir / "nod_inj.parquet")
    save_parquet_safe(pd.DataFrame([r["d1_cgm"]["LINE_F"] for r in results_sorted], index=time_sorted, columns=L), d1_cgm_dir / "line_f.parquet")
    save_parquet_safe(pd.DataFrame([r["d1_cgm"]["NP"] for r in results_sorted], index=time_sorted, columns=Z_FBMC), d1_cgm_dir / "np.parquet")
    d1cgm_export_cols = [f"export_{i}" for i in range(len(results_sorted[0]["d1_cgm"]["EXPORT"]))]
    save_parquet_safe(pd.DataFrame([r["d1_cgm"]["EXPORT"] for r in results_sorted], index=time_sorted, columns=d1cgm_export_cols), d1_cgm_dir / "export.parquet")

    # ---- D-0 ----
    d0_dir = stage_dir / "d0"
    ensure_dir(d0_dir)
    save_parquet_safe(pd.DataFrame([r["d0"]["CURT_RD"] for r in results_sorted], index=time_sorted, columns=N), d0_dir / "curt_rd.parquet")
    save_parquet_safe(pd.DataFrame([r["d0"]["RD_POS"] for r in results_sorted], index=time_sorted, columns=P_RD), d0_dir / "rd_pos.parquet")
    save_parquet_safe(pd.DataFrame([r["d0"]["RD_NEG"] for r in results_sorted], index=time_sorted, columns=P_RD), d0_dir / "rd_neg.parquet")
    save_parquet_safe(pd.DataFrame([r["d0"]["DELTA"] for r in results_sorted], index=time_sorted, columns=N), d0_dir / "delta.parquet")
    save_parquet_safe(pd.DataFrame([r["d0"]["NOD_INJ"] for r in results_sorted], index=time_sorted, columns=N), d0_dir / "nod_inj.parquet")
    save_parquet_safe(pd.DataFrame([r["d0"]["LINE_F"] for r in results_sorted], index=time_sorted, columns=L), d0_dir / "line_f.parquet")


###############################################################################
# WORKER INIT
###############################################################################


def init_worker(lodf, bad_k, selected_contingencies):
    global GLOBAL_LODF, GLOBAL_BAD_K, GLOBAL_SELECTED_CONTINGENCIES
    GLOBAL_LODF = np.asarray(lodf, dtype=np.float64)
    GLOBAL_BAD_K = np.asarray(bad_k, dtype=bool)
    GLOBAL_SELECTED_CONTINGENCIES = np.asarray(selected_contingencies, dtype=np.int64)


###############################################################################
# MAIN
###############################################################################


def load_selected_contingencies(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Selected contingencies file not found: {path}. Run Select_Critical_Contingencies.py first."
        )

    with open(path, "r") as f:
        data = json.load(f)

    selected = np.asarray(data, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise RuntimeError(f"No selected contingencies found in {path}.")

    if np.any(selected == -1):
        selected = selected[selected != -1]

    return np.unique(selected)



def main():
    ensure_dir(RESULTS_ROOT)

    selected_contingencies = load_selected_contingencies(SELECTED_CONTINGENCIES_FILE)

    config = {
        "run_name": RUN_NAME,
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
        "selected_contingencies_file": str(SELECTED_CONTINGENCIES_FILE),
        "selected_contingencies_root": str(SELECTED_CONTINGENCIES_ROOT),
        "keep_basecase_rows": KEEP_BASECASE_ROWS,
        "n_selected_contingencies": int(selected_contingencies.size),
        "selected_contingencies": selected_contingencies.astype(int).tolist(),
    }

    with open(RESULTS_ROOT / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    time_index = build_time_index()
    print(
        f"Running {len(time_index)} MTUs with {N_WORKERS} workers | "
        f"selected contingencies={selected_contingencies.size}"
    )

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
        initargs=(lodf, bad_k, selected_contingencies),
    ) as executor:
        futures = {executor.submit(solve_single_mtu, t): t for t in time_index}

        for fut in as_completed(futures):
            t = futures[fut]
            res = fut.result()

            if res["status"] == "ok":
                results.append(res)
                fb = res["fb"]
                print(
                    f"[OK] t={t} | full_rows={fb['N_TOTAL_FULL_ROWS']} | "
                    f"reduced_rows={fb['N_REDUCED_ROWS']} | "
                    f"selected_cont={fb['N_SELECTED_CONTINGENCIES']}"
                )
            else:
                failures.append(res)
                print(f"[FAIL] t={t} :: {res['error']}")
                if not CONTINUE_ON_ERROR:
                    raise RuntimeError(f"Stopping on failed MTU t={t}")

    if failures:
        pd.DataFrame(failures).to_csv(RESULTS_ROOT / "failures.csv", index=False)

    if results:
        save_matrix_results(results, RESULTS_ROOT)
        print(f"Saved results to: {RESULTS_ROOT}")

    print(f"Finished. Success={len(results)}, Failures={len(failures)}")


if __name__ == "__main__":
    main()
