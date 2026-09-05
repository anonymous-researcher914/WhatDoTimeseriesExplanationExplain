from typing import Optional

import numpy as np
from tqdm import tqdm
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import (
    KernelEstimatorProtocol,
)


def _decode_history(h_idx: int, K: int, D: int, N: int) -> np.ndarray:
    """Flat history index -> (K, D) int array."""
    h_arr = np.zeros((K, D), dtype=int)
    idx = h_idx
    for k in reversed(range(K)):
        for d in reversed(range(D)):
            h_arr[k, d] = idx % N
            idx //= N
    return h_arr


def compute_edge_contribution(
    est: "KernelEstimatorProtocol",
    pi_star: dict,
    disc: Discretiser,
    src_var: int,
    src_lag: int,
    tgt_var: int,
    tgt_step: int = 0,
) -> float:
    """
    rho(e) for edge (X^src_var_{t-src_lag} -> X^tgt_var_{t+tgt_step}).

    When est.horizon > 1 the TV is measured over the joint horizon kernel
    T^{f,tgt_var,tau}_{K*} with tau = {0, ..., L-1} (all forecast steps),
    delegating to compute_edge_contribution_multistep.  tgt_step is ignored
    in that case.

    When est.horizon == 1 the TV is measured at the single forecast step
    tgt_step (default 0), as before.

    Parameters
    ----------
    est      : BStarKernelEstimator or SDEKernelEstimator
    pi_star  : dict h_idx -> float
    disc     : Discretiser
    src_var  : source variable index d' in [D]
    src_lag  : lag k in [1, K*], 1-indexed
    tgt_var  : target variable index d in [D]
    tgt_step : forecast step l in [0, T-1]; used only when horizon == 1
    """
    T = getattr(est, "horizon", 1)
    if T > 1:
        return compute_edge_contribution_multistep(
            est,
            pi_star,
            disc,
            src_var,
            src_lag,
            tgt_var,
            tau=list(range(T)),
        )

    K = est.K
    D = disc.D_
    N = disc.N
    rho = 0.0

    for h_idx, pi_h in pi_star.items():
        if pi_h == 0.0:
            continue
        p_full = est.predict_marginal(tgt_var, h_idx, step=tgt_step)
        h_arr = _decode_history(h_idx, K, D, N)
        lag_pos = K - src_lag

        if lag_pos < 0 or lag_pos >= K:
            return 0.0

        sum_abs_diff = 0.0
        for bin_val in range(N):
            h_mod = h_arr.copy()
            h_mod[lag_pos, src_var] = bin_val
            h_mod_idx = disc.history_to_index(h_mod)
            p_mod = est.predict_marginal(tgt_var, h_mod_idx, step=tgt_step)
            sum_abs_diff += np.sum(np.abs(p_full - p_mod))

        rho += pi_h * 0.5 * sum_abs_diff / N

    return rho


def average_causal_effect(
    est: "KernelEstimatorProtocol",
    pi_star: dict,
    disc: Discretiser,
    src_var: int,
    src_lag: int,
    tgt_var: int,
    adv_var: int,
    tgt_step: int = 0,
    tau: Optional[list] = None,
) -> float:
    """
    ACE for edge (X^src_var_{t-src_lag} -> X^tgt_var_tau).

    Similar to compute_edge_contribution but measures expected value shift
    instead of TV distance. Expectation is linear, so — unlike the TV-based
    rho, which needs the true joint path kernel to capture cross-step
    correlation (Eq. 13) — the multi-step ACE over a set of forecast steps
    tau is exactly the sum of the single-step ACE at each step in tau; no
    joint kernel is required.

    When est.horizon > 1 and tau is None, tau defaults to every forecast
    step {0, ..., L-1} (tgt_step is ignored), mirroring the auto-dispatch in
    compute_edge_contribution. When est.horizon == 1, ACE is measured at the
    single step tgt_step (default 0), as before.

    Parameters
    ----------
    est      : BStarKernelEstimator or SDEKernelEstimator
    pi_star  : dict h_idx -> float
    disc     : Discretiser
    src_var  : source variable index d' in [D]
    src_lag  : lag k in [1, K*], 1-indexed
    tgt_var  : target variable index d in [D]
    adv_var  : adversarial bin index for src_var
    tgt_step : forecast step l in [0, T-1]; used only when horizon == 1
    tau      : 0-indexed forecast steps to sum over; None auto-selects all
               T steps when horizon > 1, else [tgt_step]
    """
    K = est.K
    D = disc.D_
    N = disc.N

    if tau is None:
        T = getattr(est, "horizon", 1)
        steps = list(range(T)) if T > 1 else [tgt_step]
    else:
        steps = tau

    ace = 0.0
    values = disc.bin_to_value(np.arange(N))

    for h_idx, pi_h in pi_star.items():
        if pi_h == 0.0:
            continue

        h_arr = _decode_history(h_idx, K, D, N)
        lag_pos = K - src_lag

        if lag_pos < 0 or lag_pos >= K:
            return 0.0

        assert 0 <= adv_var < N, "adv_var must be a valid bin index"
        h_mod = h_arr.copy()
        h_mod[lag_pos, src_var] = adv_var
        h_mod_idx = disc.history_to_index(h_mod)

        for step in steps:
            p_full = est.predict_marginal(tgt_var, h_idx, step=step)
            p_marg = est.predict_marginal(tgt_var, h_mod_idx, step=step)
            ace += pi_h * np.sum(values * (p_full - p_marg))

    return ace


def compute_variable_importance(
    est,
    pi_star: dict,
    disc: Discretiser,
    lam: float = 0.01,
) -> dict:
    """
    Compute variable importance over all D*T future target nodes.

    For each source (src_var, lag) and each target (tgt_var, tgt_step),
    computes rho = pi*-weighted TV when src_var at lag is marginalised from
    history h and the shift is measured in tgt_var's distribution at
    forecast step tgt_step.  No joint kernel needed — uses factored marginals.

    Returns
    -------
    dict with:
        "rho_mat" : np.ndarray (D, T, D, K) — rho[tgt_d, tgt_step, src_d, lag-1]
        "phi"     : np.ndarray (D, K)        — phi[src_d, lag-1], total influence
                                               summed over all (tgt_d, tgt_step)
        "Phi"     : np.ndarray (D,)          — total influence per source variable
        "Phi_n"   : np.ndarray (D,)          — normalised (sums to 1)
        "edges"   : list of dicts with keys src, tgt, tgt_step, lag, rho
                    for all entries with rho >= lam
    """
    K = est.K
    D = disc.D_
    N = disc.N
    T = getattr(est, "horizon", 1)
    alpha = getattr(est, "alpha", 0.5)

    h_idx_list = [h for h, w in pi_star.items() if w > 0.0]
    pi_arr = np.array([pi_star[h] for h in h_idx_list])  # (H,)
    H = len(h_idx_list)

    phi = np.zeros((D, K))
    if H == 0:
        return {
            "rho_mat": np.zeros((D, T, D, K)),
            "phi": phi,
            "Phi": phi.sum(axis=1),
            "Phi_n": np.zeros(D),
            "edges": [],
        }

    # ── Stride for each (src_var, lag) modification ───────────────────────
    # Digit at position (k=K-lag, d=src_var) has weight N^pos in the encoding,
    # where pos = (lag-1)*D + (D-1-src_var).
    # h_mod(b) = h + (b - cur_val) * stride,  cur_val = (h // stride) % N
    # Use Python int arithmetic — h_idx can reach N^(K*D) = 3^70 > int64.
    def _stride(src_var: int, lag: int) -> int:
        return N ** ((lag - 1) * D + (D - 1 - src_var))

    # ── Build kernel lookup table of shape (U+1, T, D, N) ────────────────
    # Index 0 is a uniform sentinel for unobserved histories.
    # Index i+1 holds the (T, D, N) marginal kernels for the i-th known history.
    #
    # Two paths depending on estimator type:
    #
    #   FactoredKernelEstimator: read _nf_marginal directly — all T steps,
    #       unknown h_mod values map to uniform sentinel at index 0.
    #
    #   Generic (BStarKE, TreeKE, …): enumerate every unique h_mod value
    #       first, then call est.predict(h) → (T, D, N) for each.

    if hasattr(est, "_nf_marginal"):
        # ── FactoredKernelEstimator fast path ─────────────────────────────
        known_h: set = set(h_idx_list)
        for d in range(D):
            for h, _step, _s in est._nf_marginal[d]:
                known_h.add(h)

        known_h_sorted = sorted(known_h)
        h_to_ki = {h: i + 1 for i, h in enumerate(known_h_sorted)}

        U = len(known_h_sorted)
        kern_arr = np.empty((U + 1, T, D, N), dtype=np.float32)
        kern_arr[0] = 1.0 / N  # sentinel: uniform for unobserved histories
        for i, h in tqdm(
            enumerate(known_h_sorted),
            desc="VI build kernel lookup",
            total=U,
            leave=False,
        ):
            for step in range(T):
                for d in range(D):
                    counts = np.array(
                        [est._nf_marginal[d].get((h, step, s), 0) for s in range(N)],
                        float,
                    )
                    total = counts.sum()
                    kern_arr[i + 1, step, d] = (counts + alpha) / (total + N * alpha)

        base_ki = np.array([h_to_ki[h] for h in h_idx_list], dtype=np.int32)
        h_mod_ki = np.zeros((D * K, H, N), dtype=np.int32)
        dk_i = 0
        for src_var in range(D):
            for lag in range(1, K + 1):
                stride = _stride(src_var, lag)
                cur_vals = [(h // stride) % N for h in h_idx_list]
                for b in range(N):
                    h_mods = [
                        h + (b - cv) * stride for h, cv in zip(h_idx_list, cur_vals)
                    ]
                    h_mod_ki[dk_i, :, b] = [h_to_ki.get(hm, 0) for hm in h_mods]
                dk_i += 1

    else:
        # ── Generic path (BStarKE, TreeKE, …) ────────────────────────────
        # Single pass: compute cur_vals once per (src_var, lag), enumerate
        # all unique h_mods, and cache the index arrays for the TV loop.
        # cur_vals[dk][h] = current bin of src_var at lag in history h.
        # h_mod = h + (b - cur_val) * stride  (Python big-int arithmetic).

        strides = [_stride(sv, lg) for sv in range(D) for lg in range(1, K + 1)]
        cur_vals_per_dk: list[list[int]] = []
        all_h_mods: set = set(h_idx_list)

        for s in tqdm(strides, desc="VI collect history mods", leave=False):
            cvs = [(h // s) % N for h in h_idx_list]
            cur_vals_per_dk.append(cvs)
            for b in range(N):
                all_h_mods.update(h + (b - cv) * s for h, cv in zip(h_idx_list, cvs))

        all_h_sorted = sorted(all_h_mods)
        h_to_ki = {h: i + 1 for i, h in enumerate(all_h_sorted)}

        # Build kern_arr in one batch call — avoids per-h np.stack overhead
        U = len(all_h_sorted)
        kern_arr = np.empty((U + 1, T, D, N), dtype=np.float32)
        kern_arr[0] = 1.0 / N
        kern_arr[1:] = est.predict_batch(all_h_sorted)  # (U, T, D, N)

        # Build h_mod_ki using cached cur_vals — no redundant (h // stride) % N
        base_ki = np.array([h_to_ki[h] for h in h_idx_list], dtype=np.int32)
        h_mod_ki = np.empty((D * K, H, N), dtype=np.int32)
        for dk_i, (s, cvs) in enumerate(zip(strides, cur_vals_per_dk)):
            for b in range(N):
                h_mods = [h + (b - cv) * s for h, cv in zip(h_idx_list, cvs)]
                h_mod_ki[dk_i, :, b] = [h_to_ki[hm] for hm in h_mods]

    # ── Vectorised TV computation over all D*T target nodes ──────────────
    # kern_arr[base_ki]                     → p_full   (H, T, D, N)
    # kern_arr[ki_hn]                       → p_mods   (H, N_bin, T, D, N)
    # |p_full - p_mods|.sum(-1).mean(1)    → avg_diff (H, T, D)  [avg over x, sum over s^d]
    # 0.5 * avg_diff                        → tv       (H, T, D)
    # (tv * pi).sum(0)                      → (T, D) → rho_mat[:, :, src_var, lag-1]

    p_full = kern_arr[base_ki]  # (H, T, D, N)
    rho_mat = np.zeros((D, T, D, K))  # rho_mat[tgt_d, tgt_step, src_d, lag-1]

    dk_i = 0
    for src_var in tqdm(
        range(D),
        desc="VI compute target TV",
        leave=False,
    ):
        for lag in range(1, K + 1):
            ki_hn = h_mod_ki[dk_i]  # (H, N_bin)
            p_mods = kern_arr[ki_hn]  # (H, N_bin, T, D, N)
            avg_diff = (
                np.abs(p_full[:, None] - p_mods).sum(axis=-1).mean(axis=1)
            )  # (H, T, D)
            tv = 0.5 * avg_diff
            weighted = (tv * pi_arr[:, None, None]).sum(axis=0)  # (T, D)
            rho_mat[:, :, src_var, lag - 1] = weighted.T  # (D, T)
            dk_i += 1

    rho_thresh = np.where(rho_mat >= lam, rho_mat, 0.0)
    phi = rho_thresh.sum(axis=(0, 1))  # (D_src, K) — summed over all (tgt_d, tgt_step)

    tgt_d_i, tgt_step_i, src_i, k_i = np.where(rho_mat >= lam)
    edges = [
        {
            "src": int(src_i[i]),
            "tgt": int(tgt_d_i[i]),
            "tgt_step": int(tgt_step_i[i] + 1),  # 1-indexed
            "lag": int(k_i[i] + 1),
            "rho": float(rho_mat[tgt_d_i[i], tgt_step_i[i], src_i[i], k_i[i]]),
        }
        for i in range(len(tgt_d_i))
    ]

    Phi = phi.sum(axis=1)
    total = Phi.sum()
    Phi_n = Phi / total if total > 0 else Phi

    return {"rho_mat": rho_mat, "phi": phi, "Phi": Phi, "Phi_n": Phi_n, "edges": edges}


def compute_edge_contribution_multistep(
    est: "KernelEstimatorProtocol",
    pi_star: dict,
    disc: Discretiser,
    src_var: int,
    src_lag: int,
    tgt_var: int,
    tau: Optional[list] = None,
) -> float:
    """
    rho(X^{src_var}_{t-src_lag} -> X^{tgt_var}) via path kernel T^{f,M_d,tau}_{K*}.

    Estimates the true joint trajectory of tgt_var over tau steps from raw MC
    samples via est.predict_path_marginal, capturing cross-step correlations
    per Eq. (19).

    Parameters
    ----------
    est      : BStarKE / TreeKE — must implement predict_path_marginal
    pi_star  : dict h_idx -> float
    disc     : Discretiser
    src_var  : source variable d' in [D]
    src_lag  : lag k in [1, K*], 1-indexed
    tgt_var  : target variable d in [D]
    tau      : 0-indexed forecast step indices; None defaults to all T steps.
               State space is [N]^|tau| per target dim.
    """
    K = est.K
    D = disc.D_
    N = disc.N
    T = getattr(est, "horizon", 1)

    if tau is None:
        tau = list(range(T))
    tau_arr = np.array(tau, dtype=int)
    q = len(tau_arr)

    lag_pos = K - src_lag
    if lag_pos < 0 or lag_pos >= K:
        return 0.0

    tau_list = list(tau_arr)
    mask = [(tgt_var, i) for i in tau_list]  # supp(M_{tgt_var,tau}), Eq. (13)

    rho = 0.0
    for h_idx, pi_h in pi_star.items():
        if pi_h == 0.0:
            continue

        h_arr = _decode_history(h_idx, K, D, N)
        joint_full = est.predict_path_marginal(mask, h_idx)  # (N^q,)

        sum_abs_diff = 0.0
        for bin_val in range(N):
            h_mod = h_arr.copy()
            h_mod[lag_pos, src_var] = bin_val
            p_mod = est.predict_path_marginal(mask, disc.history_to_index(h_mod))
            sum_abs_diff += np.sum(np.abs(joint_full - p_mod))

        rho += pi_h * 0.5 * sum_abs_diff / N

    return rho


def compute_variable_importance_multistep(
    est,
    pi_star: dict,
    disc: Discretiser,
    tau: Optional[list] = None,
    lam: float = 0.01,
) -> dict:
    """
    Lag-resolved influence and variable importance via path kernels T^{f,M_d,tau}_{K*}.

    For each target variable d the path M_d selects dimension d at every step,
    giving the joint trajectory distribution T^{f,M_d,tau}_{K*} over [N]^|tau|,
    estimated from raw MC samples via est.predict_path_marginal per Eq. (19).

      rho(X^{d'}_{t-k} -> X^d_t) = sum_h pi*(h) · 0.5 · TV_h  (over [N]^|tau|)
      phi^{d'}_k  = sum_d rho(...) · 1{rho > lam}
      Phi^{d'}    = sum_k phi^{d'}_k, normalised

    Parameters
    ----------
    est     : BStarKE / TreeKE (path kernels) or FactoredKE (product fallback)
    pi_star : dict h_idx -> float
    disc    : Discretiser
    tau     : 0-indexed forecast step indices; None defaults to all T steps.
              State space per target dim is [N]^|tau|.
    lam     : threshold lambda for the indicator in phi

    Returns
    -------
    dict with:
        "rho_mat" : np.ndarray (D, D, K) — rho_mat[tgt_d, src_d, lag-1]
        "phi"     : np.ndarray (D, K)    — phi[src_d, lag-1]
        "Phi"     : np.ndarray (D,)      — variable importance per source
        "Phi_n"   : np.ndarray (D,)      — normalised (sums to 1)
        "edges"   : list of dicts with keys src, tgt, lag, rho for rho >= lam
    """
    K = est.K
    D = disc.D_
    N = disc.N
    T = getattr(est, "horizon", 1)
    alpha = getattr(est, "alpha", 0.5)

    if tau is None:
        tau = list(range(T))
    tau_arr = np.array(tau, dtype=int)
    q = len(tau_arr)
    Nq = N**q
    tau_list = list(tau_arr)

    h_idx_list = [h for h, w in pi_star.items() if w > 0.0]
    pi_arr = np.array([pi_star[h] for h in h_idx_list])
    H = len(h_idx_list)

    rho_mat = np.zeros((D, D, K))

    if H == 0:
        phi = np.zeros((D, K))
        return {
            "rho_mat": rho_mat,
            "phi": phi,
            "Phi": phi.sum(axis=1),
            "Phi_n": np.zeros(D),
            "edges": [],
        }

    def _stride(src_var: int, lag: int) -> int:
        return N ** ((lag - 1) * D + (D - 1 - src_var))

    strides = [_stride(sv, lg) for sv in range(D) for lg in range(1, K + 1)]
    cur_vals_per_dk: list = []
    all_h_mods: set = set(h_idx_list)
    for s in tqdm(strides, desc="VI multistep collect history mods", leave=False):
        cvs = [(h // s) % N for h in h_idx_list]
        cur_vals_per_dk.append(cvs)
        for b in range(N):
            all_h_mods.update(h + (b - cv) * s for h, cv in zip(h_idx_list, cvs))
    all_h_sorted = sorted(all_h_mods)
    h_to_ki = {h: i + 1 for i, h in enumerate(all_h_sorted)}
    U = len(all_h_sorted)

    base_ki = np.array([h_to_ki[h] for h in h_idx_list], dtype=np.int32)
    h_mod_ki = np.empty((D * K, H, N), dtype=np.int32)
    for dk_i, (s, cvs) in enumerate(zip(strides, cur_vals_per_dk)):
        for b in range(N):
            h_mods = [h + (b - cv) * s for h, cv in zip(h_idx_list, cvs)]
            h_mod_ki[dk_i, :, b] = [h_to_ki[hm] for hm in h_mods]

    # ── Build kern_arr (U+1, D, N^q) ──────────────────────────────────────
    kern_arr = np.empty((U + 1, D, Nq), dtype=np.float32)
    kern_arr[0] = 1.0 / Nq  # uniform sentinel for unobserved histories
    # if hasattr(est, "predict_path_marginal_all_dims_batch"):
    #     # One padded GPU/CPU batch call over every history at once, instead
    #     # of U separate predict_path_marginal_all_dims calls.
    #     kern_arr[1:] = est.predict_path_marginal_all_dims_batch(
    #         tau_list, all_h_sorted
    #     )
    #else:
    for hi, h in tqdm(
        enumerate(all_h_sorted),
        desc="VI multistep build path kernels",
        total=U,
        leave=False,
    ):
        # One call computes every target dim's M_{d,tau} kernel at once
        # (Eq. 13), instead of D separate predict_path_marginal calls
        # that would each re-route/re-concatenate h's samples from
        # scratch.
        kern_arr[hi + 1] = est.predict_path_marginal_all_dims(tau_list, h)

    # ── TV loop over (src_var, lag) → rho_mat[tgt_d, src_d, lag-1] ────────
    # kern_arr[base_ki]                    : (H, D, N^q)  — full path kernels
    # kern_arr[ki_hn]                      : (H, N_bin, D, N^q)
    # |p_full - p_mods|.sum(-1).mean(1)   : (H, D)  — avg over x, sum over joint states
    p_full = kern_arr[base_ki]  # (H, D, N^q)

    dk_i = 0
    for src_var in tqdm(
        range(D),
        desc="VI multistep compute target TV",
        leave=False,
    ):
        for lag in range(1, K + 1):
            ki_hn = h_mod_ki[dk_i]  # (H, N_bin)
            p_mods = kern_arr[ki_hn]  # (H, N_bin, D, N^q)
            avg_diff = (
                np.abs(p_full[:, None] - p_mods).sum(axis=-1).mean(axis=1)
            )  # (H, D)
            tv = 0.5 * avg_diff
            rho_mat[:, src_var, lag - 1] = (tv * pi_arr[:, None]).sum(axis=0)
            dk_i += 1

    rho_thresh = np.where(rho_mat >= lam, rho_mat, 0.0)
    phi = rho_thresh.sum(axis=0)  # (D_src, K)

    tgt_d_i, src_i, k_i = np.where(rho_mat >= lam)
    edges = [
        {
            "src": int(src_i[i]),
            "tgt": int(tgt_d_i[i]),
            "lag": int(k_i[i] + 1),
            "rho": float(rho_mat[tgt_d_i[i], src_i[i], k_i[i]]),
        }
        for i in range(len(tgt_d_i))
    ]

    Phi = phi.sum(axis=1)
    total = Phi.sum()
    Phi_n = Phi / total if total > 0 else Phi

    return {"rho_mat": rho_mat, "phi": phi, "Phi": Phi, "Phi_n": Phi_n, "edges": edges}
