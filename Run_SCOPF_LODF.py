"""
Run D-2 CGM as a preventive N-1 secure nodal DC-OPF using LODF (NO pred_np / NO slack vars),
and save base-case results to Parquet (same style as your old script).

What this script does:
1) Precomputes LODF from PTDF once.
2) Builds ONE CVXPY problem for the preventive D-2 model:
      - base nodal DC-OPF (D-2 CGM)
      - + preventive N-1 constraints (LODF-only, no corrective redispatch)
3) Loops over time t, sets demand/renew parameters, solves, and exports:
      GEN, CURT, LINE_F, NP, EXPORT, NOD_INJ, DELTA + summary diagnostics.
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp

from input_data_base_functions import *  # must provide: df_branch, PTDF_full, sets, indices, get_dem_vec, get_renew_vec, line_cap, etc.
from Differentiable_D_2_SCOPF_CGM import build_d2_cgm_problem_components


# =============================================================================
# LODF computation from PTDF (base topology)
# =============================================================================
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

    # ptdf_txn[:,k] = PTDF(:,from_k) - PTDF(:,to_k)
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


# =============================================================================
# Diagnostics for preventive SCOPF (LODF-only)
# =============================================================================
def preventive_diagnostics_from_solution(flow_base, cap_margin, LODF, bad_k):
    """
    Computes:
      base_max_load:   max_l |f_l| / cap_l
      worst_post_load: max_k max_{l!=k} |f_post_l(k)| / cap_l
      worst_k:         contingency index k giving worst_post_load
    where f_post(k) = f + LODF[:,k]*f[k]
    """
    f = np.asarray(flow_base, dtype=float).reshape(-1)
    cap = np.asarray(cap_margin, dtype=float).reshape(-1)

    base_max_load = float(np.max(np.abs(f) / cap))

    nL = len(f)
    worst_post_load = -np.inf
    worst_k = None

    for k in range(nL):
        if bad_k[k]:
            continue

        f_post = f + (LODF[:, k] * f[k])

        mask = np.ones(nL, dtype=bool)
        mask[k] = False

        loading_k = float(np.max(np.abs(f_post[mask]) / cap[mask]))
        if loading_k > worst_post_load:
            worst_post_load = loading_k
            worst_k = int(k)

    return base_max_load, worst_post_load, worst_k


# =============================================================================
# Build the preventive D-2 model once (solve many times by changing parameters)
# =============================================================================
def build_preventive_d2_model(
    frm=0.05,
    lodf_eps=1e-8,
    contingencies_idx=None,
    monitored_idx=None,
):
    """
    Builds the preventive D-2 problem:
      - Uses build_d2_cgm_problem_components (NO pred_np)
      - Adds preventive constraints inside that builder via (preventive=True, LODF, bad_k)

    Returns a dict containing:
      problem, parameter handles (dem, renew, line_cap), variable handles, and constants.
    """

    # LODF is constant over time -> compute once
    LODF, bad_k = compute_lodf_from_ptdf(df_branch, PTDF_full, N_idx, L_idx, L, eps=lodf_eps)

    # Build D-2 preventive components (note: NO use_pred_np here)
    objective, constraints, params_list, vars_list, _cost_curt, _params_dict, _vars_dict = build_d2_cgm_problem_components(
        frm=frm,
        preventive=True,
        LODF=LODF,
        bad_k=bad_k,
        contingencies_idx=contingencies_idx,
        monitored_idx=monitored_idx,
    )

    # Params now are [dem, renew, line_cap]
    dem_param, renew_param, line_cap_param = params_list

    # Vars are [GEN, CURT, NOD_INJ, LINE_F, NP, EXPORT, DELTA]
    GEN, CURT, NOD_INJ, LINE_F, NP, EXPORT, DELTA = vars_list

    problem = cp.Problem(objective, constraints)

    # For diagnostics and reporting
    cap_margin = line_cap * (1.0 - frm)

    return {
        "problem": problem,
        "dem": dem_param,
        "renew": renew_param,
        "line_cap": line_cap_param,
        "GEN": GEN,
        "CURT": CURT,
        "NOD_INJ": NOD_INJ,
        "LINE_F": LINE_F,
        "NP": NP,
        "EXPORT": EXPORT,
        "DELTA": DELTA,
        "cap_margin": cap_margin,
        "LODF": LODF,
        "bad_k": bad_k,
        "frm": frm,
    }


# =============================================================================
# Run and export Parquet
# =============================================================================
def run_preventive_d2_and_save_parquet(
    out_dir: str,
    t_start: int = 1,
    t_end: int = 4000,
    frm: float = 0.05,
    solver=cp.GUROBI,
    warm_start: bool = True,
    verbose: bool = False,
    write_every: int = 0,
    include_diagnostics_in_summary: bool = True,
    contingencies_idx=None,
    monitored_idx=None,
):
    """
    Solves preventive D-2 for t in [t_start, t_end] and writes Parquet outputs:
      - d_2_summary.parquet (status, obj, diagnostics)
      - d_2_gen.parquet
      - d_2_curt.parquet
      - d_2_flows.parquet
      - d_2_nps.parquet
      - d_2_export.parquet
      - d_2_nod_inj.parquet
      - d_2_delta.parquet

    If infeasible/non-optimal: store NaNs for that row.
    """

    os.makedirs(out_dir, exist_ok=True)

    model = build_preventive_d2_model(
        frm=frm,
        contingencies_idx=contingencies_idx,
        monitored_idx=monitored_idx,
    )
    problem = model["problem"]

    # Column labels (IDs, not indices)
    gen_cols = [str(p) for p in P]
    curt_cols = [str(n) for n in N]
    flow_cols = [str(l) for l in L]
    np_cols = [str(z) for z in Z_FBMC]
    export_cols = [str(z) for z in Z_not_in_FBMC]
    nodinj_cols = [str(n) for n in N]
    delta_cols = [str(n) for n in N]

    summary_rows, gen_rows, curt_rows, flow_rows = [], [], [], []
    np_rows, export_rows, nodinj_rows, delta_rows = [], [], [], []

    def flush(tag: str = ""):
        pd.DataFrame(summary_rows).to_parquet(os.path.join(out_dir, f"d_2_summary{tag}.parquet"), index=False)
        pd.DataFrame(gen_rows,    columns=["t"] + gen_cols).to_parquet(os.path.join(out_dir, f"d_2_gen{tag}.parquet"), index=False)
        pd.DataFrame(curt_rows,   columns=["t"] + curt_cols).to_parquet(os.path.join(out_dir, f"d_2_curt{tag}.parquet"), index=False)
        pd.DataFrame(flow_rows,   columns=["t"] + flow_cols).to_parquet(os.path.join(out_dir, f"d_2_flows{tag}.parquet"), index=False)
        pd.DataFrame(np_rows,     columns=["t"] + np_cols).to_parquet(os.path.join(out_dir, f"d_2_nps{tag}.parquet"), index=False)
        pd.DataFrame(export_rows, columns=["t"] + export_cols).to_parquet(os.path.join(out_dir, f"d_2_export{tag}.parquet"), index=False)
        pd.DataFrame(nodinj_rows, columns=["t"] + nodinj_cols).to_parquet(os.path.join(out_dir, f"d_2_nod_inj{tag}.parquet"), index=False)
        pd.DataFrame(delta_rows,  columns=["t"] + delta_cols).to_parquet(os.path.join(out_dir, f"d_2_delta{tag}.parquet"), index=False)

    for i, t in enumerate(range(t_start, t_end + 1), start=1):
        # Set time-varying parameters
        model["dem"].value = get_dem_vec(t)
        model["renew"].value = get_renew_vec(t)

        # line capacities are constant in your data -> set once or set each time (cheap)
        model["line_cap"].value = line_cap

        # Solve
        problem.solve(
            solver=solver,
            verbose=verbose,
            warm_start=warm_start,
            reoptimize=True,
            Method=2,
            Crossover=0,
            NumericFocus=1,
        )

        status = problem.status
        obj = float(problem.value) if problem.value is not None else np.nan

        # Optional diagnostics (only if optimal)
        base_max = worst_post = np.nan
        worst_k = None
        if status == "optimal" and include_diagnostics_in_summary:
            base_max, worst_post, worst_k = preventive_diagnostics_from_solution(
                flow_base=model["LINE_F"].value,
                cap_margin=model["cap_margin"],
                LODF=model["LODF"],
                bad_k=model["bad_k"],
            )

        summary_rows.append({
            "t": int(t),
            "status": status,
            "obj": obj,
            **({
                "base_max_load": base_max,
                "worst_post_load": worst_post,
                "worst_k": worst_k,
            } if include_diagnostics_in_summary else {})
        })

        obj_str = f"{obj:,.1f}" if np.isfinite(obj) else "None"
        print(f"t={t} | status={status} | obj={obj_str}", end="")

        if status != "optimal":
            print("  <-- NaNs stored")
            gen_rows.append([int(t)] + [np.nan] * len(P))
            curt_rows.append([int(t)] + [np.nan] * len(N))
            flow_rows.append([int(t)] + [np.nan] * len(L))
            np_rows.append([int(t)] + [np.nan] * len(Z_FBMC))
            export_rows.append([int(t)] + [np.nan] * len(Z_not_in_FBMC))
            nodinj_rows.append([int(t)] + [np.nan] * len(N))
            delta_rows.append([int(t)] + [np.nan] * len(N))
        else:
            print("")
            gen_rows.append([int(t)] + np.asarray(model["GEN"].value, dtype=float).reshape(-1).tolist())
            curt_rows.append([int(t)] + np.asarray(model["CURT"].value, dtype=float).reshape(-1).tolist())
            flow_rows.append([int(t)] + np.asarray(model["LINE_F"].value, dtype=float).reshape(-1).tolist())
            np_rows.append([int(t)] + np.asarray(model["NP"].value, dtype=float).reshape(-1).tolist())
            export_rows.append([int(t)] + np.asarray(model["EXPORT"].value, dtype=float).reshape(-1).tolist())
            nodinj_rows.append([int(t)] + np.asarray(model["NOD_INJ"].value, dtype=float).reshape(-1).tolist())
            delta_rows.append([int(t)] + np.asarray(model["DELTA"].value, dtype=float).reshape(-1).tolist())

        # Periodic flush (optional)
        if write_every and (i % write_every == 0):
            tag = f"_upto_{t}"
            flush(tag=tag)
            print(f"  wrote partial Parquet files up to t={t} into {out_dir}")

    flush(tag="")
    print(f"\nSaved Parquet results to: {out_dir}")
    return out_dir


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    frm = 0.05
    out_dir = os.path.join(os.getcwd(), "BaseCase_D_2_CGM_Results")

    run_preventive_d2_and_save_parquet(
        out_dir=out_dir,
        t_start=1,
        t_end=len(T),
        frm=frm,
        solver=cp.GUROBI,
        warm_start=True,
        verbose=False,
        write_every=0,
        include_diagnostics_in_summary=True,
        # To reduce size later:
        # contingencies_idx=CNEC_indices,
        # monitored_idx=CNEC_indices,
    )