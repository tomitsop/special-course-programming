import os
import math
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
import cvxpy as cp
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import contextlib, io, importlib
from typing import Any, Dict, List, Tuple


# ---- keep your exact solver settings ----
solver_args = {
    "solve_method": "SCS",
    "eps": 1e-8,
    "max_iters": 200000,
    "verbose": False,
}

# Global cache per worker process
G: Dict[str, Any] = {}


def _set_thread_env():
    # Prevent each process from spawning its own BLAS thread pool
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)


def init_worker(ml_results_path: str, suppress_diffcp_warnings: bool = True):
    _set_thread_env()

    if suppress_diffcp_warnings:
        warnings.filterwarnings("ignore", message="Solved/Inaccurate.", category=UserWarning)

    # Import base quietly (avoids clutter from printouts during worker start)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        base = importlib.import_module("input_data_base_functions")

    device = torch.device("cpu")
    G["base"] = base
    G["device"] = device

    # predictions
    pred_path = os.path.join(ml_results_path, "predictions_NP.csv")
    pred_df = pd.read_csv(pred_path, index_col=0)
    pred_df.index = pred_df.index.astype(int)
    pred_df.columns = pred_df.columns.astype(str)
    G["predictions_np"] = pred_df

    # PTDF alignment -> PTDF^T (Z x J)
    if isinstance(base.PTDF_Z_CNEC, pd.DataFrame):
        PTDF_df = base.PTDF_Z_CNEC.copy()
    else:
        PTDF_df = pd.DataFrame(base.PTDF_Z_CNEC, index=base.CNEC, columns=base.Z_FBMC)

    cnec_order = [str(x) for x in base.CNEC]
    z_order = [str(x) for x in base.Z_FBMC]
    PTDF_df.index = PTDF_df.index.astype(str)
    PTDF_df.columns = PTDF_df.columns.astype(str)

    missing_rows = set(cnec_order) - set(PTDF_df.index)
    missing_cols = set(z_order) - set(PTDF_df.columns)
    if missing_rows or missing_cols:
        raise ValueError(
            f"[init_worker] PTDF label mismatch.\n"
            f"Missing rows: {list(sorted(missing_rows))[:10]}\n"
            f"Missing cols: {list(sorted(missing_cols))[:10]}"
        )

    PTDF_mat = PTDF_df.loc[cnec_order, z_order].values  # (J, Z)
    G["PTDF_T"] = torch.tensor(PTDF_mat.T, dtype=torch.double, device=device)  # (Z, J)

    # CNEC indices
    CNEC_indices = [base.L_idx[l] for l in base.CNEC]
    G["CNEC_idx_t"] = torch.tensor(CNEC_indices, dtype=torch.long, device=device)

    # RAM constants
    G["line_cap_cnec"] = torch.tensor(
        [base.line_cap[i] for i in CNEC_indices],
        dtype=torch.double,
        device=device,
    )
    G["frm_t"] = torch.tensor(float(base.frm), dtype=torch.double, device=device)

    # layers
    cost_curt = math.ceil(base.find_maximum_mc())
    cost_curt_mc = 0
    max_ntc = 1000.0
    export_eps = 1e-7

    from Differentiable_D_2_CGM import build_d2_cgm_layer
    from Differentiable_D_1_MC import build_d1_mc_layer
    from Differentiable_D_1_CGM import build_d1_cgm_layer

    G["d2_layer"] = build_d2_cgm_layer(cost_curt=cost_curt)
    G["d1_mc_layer"] = build_d1_mc_layer(cost_curt_mc=cost_curt_mc, max_ntc=max_ntc, export_eps=export_eps)
    G["d1_cgm_layer"] = build_d1_cgm_layer(cost_curt=cost_curt, max_ntc=max_ntc)


def compute_ram_from_d2(line_f_d2_cnec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    base_cap = G["line_cap_cnec"] * (1.0 - G["frm_t"])  # (J,)
    RAM_pos = base_cap - line_f_d2_cnec
    RAM_neg = -base_cap - line_f_d2_cnec
    return RAM_pos, RAM_neg


def build_dem_renew_t(t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    base = G["base"]
    dem_np = np.array([base.get_dem(t, n) for n in base.N], dtype=np.float64)
    renew_np = np.array([base.get_renew(t, n) for n in base.N], dtype=np.float64)
    dem_t = torch.from_numpy(dem_np).to(device=G["device"], dtype=torch.double)
    renew_t = torch.from_numpy(renew_np).to(device=G["device"], dtype=torch.double)
    return dem_t, renew_t


def fbme_slack_np1mc_from_np(
    np_pred_t: torch.Tensor,
    dem_t: torch.Tensor,
    renew_t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      fbme_vec:  (J,)   FBME per CNEC (signed)
      slack_vec: (Z,)   slack_pos_d2 - slack_neg_d2  (used for slack≈0 filtering like the paper code)
      NP_d1:     (Z,)   D-1 MC net positions (used to choose the paper-style direction)

    FBME definition matches the paper:
      FBME = F_D1_CGM - F_D2_CGM - (NP_D1_CGM - NP_D2_CGM) * PTDF
    """
    d2_layer = G["d2_layer"]
    d1_mc_layer = G["d1_mc_layer"]
    d1_cgm_layer = G["d1_cgm_layer"]

    # ---- D-2 (differentiable) ----
    GEN_d2, CURT_d2, NOD_INJ_d2, LINE_F_d2, NP_d2, EXPORT_d2, \
    DELTA_d2, slack_pos_d2, slack_neg_d2 = d2_layer(np_pred_t, dem_t, renew_t, solver_args=solver_args)

    # Slack vector (Z,)
    slack_pos_d2 = slack_pos_d2.to(torch.double).reshape(-1)
    slack_neg_d2 = slack_neg_d2.to(torch.double).reshape(-1)
    slack_vec = slack_pos_d2 - slack_neg_d2

    LINE_F_d2_CNEC = LINE_F_d2[G["CNEC_idx_t"]].to(torch.double)  # (J,)
    RAM_pos, RAM_neg = compute_ram_from_d2(LINE_F_d2_CNEC)

    # ---- D-1 MC (differentiable) ----
    GEN_d1, CURT_d1, NP_d1, EXPORT_d1 = d1_mc_layer(
        dem_t, renew_t, NP_d2.to(torch.double), RAM_pos, RAM_neg, solver_args=solver_args
    )
    NP_d1 = NP_d1.to(torch.double).reshape(-1)  # ensure (Z,)

    # ---- D-1 CGM (differentiable) ----
    GEN_cgm, CURT_cgm, DELTA_cgm, NOD_INJ_cgm, LINE_F_cgm, \
    NP_cgm, EXPORT_cgm = d1_cgm_layer(
        dem_t, renew_t, GEN_d1.to(torch.double), CURT_d1.to(torch.double), solver_args=solver_args
    )

    LINE_F_cgm_CNEC = LINE_F_cgm[G["CNEC_idx_t"]].to(torch.double)  # (J,)

    # ---- FBME per CNEC (J,) ----
    np_diff = (NP_cgm - NP_d2).to(torch.double).reshape(-1)  # (Z,)
    adjust = np_diff @ G["PTDF_T"]                           # (J,)
    fbme_vec = (LINE_F_cgm_CNEC - LINE_F_d2_CNEC) - adjust   # (J,)

    return fbme_vec, slack_vec, NP_d1


def jacobian_vjp_vector_output(target_vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    target_vec: (J,)
    x: (Z,) requires_grad=True
    returns: (J, Z)
    """
    J = target_vec.shape[0]
    Z = x.shape[0]
    JAC = torch.zeros((J, Z), dtype=target_vec.dtype, device=target_vec.device)

    # Row-by-row via VJP with basis vector in output space
    for j in range(J):
        v = torch.zeros_like(target_vec)
        v[j] = 1.0
        grad = torch.autograd.grad(
            outputs=target_vec,
            inputs=x,
            grad_outputs=v,
            retain_graph=(j < J - 1),
            create_graph=False,
            allow_unused=False,
        )[0]  # (Z,)
        JAC[j, :] = grad

    return JAC


def solve_one_mtu(t: int, retry_serial: int = 1) -> Tuple[int, Any]:
    """
    Returns:
      (t, (JAC_DIR, slack_vec)) where:
        JAC_DIR: (J, Z) directional derivative of |FBME| wrt np_pred along paper-style "worsen error" direction
        slack_vec: (Z,) = slack_pos_d2 - slack_neg_d2
    Or:
      (t, {"error": ..., "trace": ...})
    """
    base = G["base"]
    pred_df = G["predictions_np"]
    cols = [str(z) for z in base.Z_FBMC]
    device = G["device"]

    last_err = None
    last_trace = None

    for _attempt in range(retry_serial + 1):
        try:
            dem_t, renew_t = build_dem_renew_t(int(t))

            # input to D-2: np_pred (CGMA prediction)
            np_pred_np = pred_df.loc[int(t), cols].values.astype(np.float64)
            np_pred_t = torch.from_numpy(np_pred_np).to(device=device, dtype=torch.double)
            np_pred_t.requires_grad_(True)

            fbme_vec, slack_vec, NP_d1 = fbme_slack_np1mc_from_np(np_pred_t, dem_t, renew_t)

            # We match the paper's sensitivity metric by differentiating |FBME|
            abs_fbme = torch.abs(fbme_vec)  # (J,)

            # Full Jacobian: ∂|FBME|/∂np_pred  -> (J,Z)
            JAC = jacobian_vjp_vector_output(abs_fbme, np_pred_t)

            # -------- NEW: paper-style direction (worsen prediction error) --------
            # paper rule: eps is +0.001 if np_pred > NP_d1mc else -0.001
            # so direction d_z(t) = sign(np_pred_z - NP_d1mc_z)
            dir_z = torch.sign(np_pred_t.detach() - NP_d1.detach()).to(torch.double).reshape(-1)  # (Z,)
            # tie-break: if exactly equal, pick +1 (or set to 0 if you want to ignore ties)
            dir_z[dir_z == 0] = 1.0

            # Directional derivative per CNEC and zone: JAC_dir[j,z] = JAC[j,z] * dir_z[z]
            JAC_DIR = JAC * dir_z.unsqueeze(0)  # broadcast over rows

            return int(t), (JAC_DIR.detach().cpu().numpy(), slack_vec.detach().cpu().numpy())

        except Exception as e:
            last_err = e
            last_trace = traceback.format_exc()
            time.sleep(0.05)

    return int(t), {"error": repr(last_err), "trace": last_trace}


def main():
    print("Torch:", torch.__version__)
    print("CVXPY solvers:", cp.installed_solvers())

    number_of_days = 30 * 6
    number_of_hours = 24 * number_of_days
    ml_results_path = f"ML_results/{number_of_hours}_hours"

    # Use X just to build the list of MTUs
    X = pd.read_csv("data/X.csv", index_col=0)
    if X.index.min() == 0:
        X.index = X.index + 1
    X.index = X.index.astype(int)

    pred_df = pd.read_csv(f"{ml_results_path}/predictions_NP.csv", index_col=0)
    pred_df.index = pred_df.index.astype(int)

    common_index = X.index.intersection(pred_df.index)
    time_index = common_index.values.astype(int).tolist()
    total = len(time_index)

    max_workers = 12
    print(f"MTUs: {total} | tasks: {total} | max_workers: {max_workers}")

    # NEW output folder name to avoid overwriting old results
    out_dir = os.path.join(ml_results_path, "pipeline_results_absfbme_dirgrad_slackfiltered")
    os.makedirs(out_dir, exist_ok=True)

    # IMPORTANT: slack tolerance should be realistic for SCS
    slack_tol = 1e-5 # if too strict, try 1e-5

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    t_prog0 = time.perf_counter()

    jac_rows: List[Tuple[int, np.ndarray]] = []   # (t, JAC_DIR)
    slack_rows: List[Tuple[int, np.ndarray]] = [] # (t, slack_vec)
    failures: List[int] = []

    done = 0
    last_print = 0
    print_every = 25

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(ml_results_path, True),
    ) as ex:
        futures = {ex.submit(solve_one_mtu, t, 1): t for t in time_index}

        for fut in as_completed(futures):
            done += 1
            t_out, out = fut.result()

            if isinstance(out, dict):
                failures.append(t_out)
            else:
                JAC_dir_np, slack_np = out
                jac_rows.append((t_out, JAC_dir_np))
                slack_rows.append((t_out, slack_np))

            if done - last_print >= print_every or done == total:
                elapsed = time.perf_counter() - t_prog0
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"Progress: {done}/{total} ({100*done/total:.1f}%) | "
                    f"ok={len(jac_rows)} fail={len(failures)}",
                    flush=True
                )
                last_print = done

    if not jac_rows:
        print("\nNo successful MTUs.")
        return

    # Sort by time index
    jac_rows.sort(key=lambda x: x[0])
    slack_rows.sort(key=lambda x: x[0])

    J_stack = np.stack([J for _, J in jac_rows], axis=0)   # (T_ok, J, Z) directional derivatives
    S_stack = np.stack([s for _, s in slack_rows], axis=0) # (T_ok, Z)

    # Grab labels
    import input_data_base_functions as base_main
    cnec_names = [str(x) for x in base_main.CNEC]
    zone_names = [str(z) for z in base_main.Z_FBMC]

    Tn, Jn, Zn = J_stack.shape
    assert len(cnec_names) == Jn, "CNEC label length mismatch"
    assert len(zone_names) == Zn, "Zone label length mismatch"

    # PAPER-STYLE AGGREGATION (but now directional):
    #   For each zone z:
    #     - filter MTUs where slack_z ≈ 0
    #     - compute median over time of directional derivative per CNE
    #     - plot abs(median) like "absolute median PD"
    for z in range(Zn):
        slack_z = S_stack[:, z]
        mask = np.abs(slack_z) <= slack_tol
        n_keep = int(mask.sum())

        if n_keep == 0:
            print(f"[WARN] Zone {z+1}: no MTUs with slack≈0 under tol={slack_tol}. Try slack_tol=1e-5.")
            continue

        grads_keep = J_stack[mask, :, z]        # (n_keep, J) directional derivative for zone z
        med = np.median(grads_keep, axis=0)     # (J,) signed median directional derivative
        abs_med = np.abs(med)                   # paper-style plotted metric

        # Save CSV (include both signed and absolute)
        df = pd.DataFrame({
            "CNEC": cnec_names,
            "median_dir_dAbsFBME_dNPpred": med,
            "abs_median_dir_dAbsFBME_dNPpred": abs_med,
            "n_MTUs_used": n_keep
        })
        df.to_csv(os.path.join(out_dir, f"abs_median_dirpd_zone_{z+1}_slack0.csv"), index=False)

        # Plot absolute median PD sorted desc
        order = np.argsort(-abs_med)
        abs_sorted = abs_med[order]

        plt.figure(figsize=(12, 4))
        plt.bar(range(len(abs_sorted)), abs_sorted)
        plt.title(
            f"Absolute Median Directional PD per CNE: |median(d|FBME|/dNP_pred · dir)| | "
            f"Zone {z+1} | slack≈0 MTUs: {n_keep}"
        )
        plt.xlabel("CNE (sorted)")
        plt.ylabel("Absolute Median Directional PD [MW/MW]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"abs_median_dirpd_zone_{z+1}.png"), dpi=200)
        plt.show()

    t1 = time.perf_counter()
    print("\nDone.")
    print("Success:", len(jac_rows), "| Failures:", len(failures))
    print("Runtime:", f"{t1 - t0:.2f}s")
    if failures:
        print("Failure examples:", failures[:10])


if __name__ == "__main__":
    main()
