import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple


from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import FactoredKernelEstimator


def compute_pi_star(X: np.ndarray, disc: Discretiser, K: int) -> dict:
    """
    Empirical stationary weights pi_hat*(h) from raw MVTS X.
    Unchanged from joint version — weights are over history space H_K
    regardless of whether the kernel is joint or factored.

    Returns
    -------
    pi_star : dict  h_idx -> float
    """
    T = X.shape[0]
    X_disc = disc.transform(X)
    counts = defaultdict(int)
    for t in range(K, T):
        h_idx = disc.history_to_index(X_disc[t - K : t])
        counts[h_idx] += 1
    total = sum(counts.values())
    return {h: cnt / total for h, cnt in counts.items()}


def compute_delta_pred(
    f: Callable,
    tilde_hs: np.ndarray,
    K: int,
    baseline: np.ndarray,
    loss: str = "classification",
) -> float:
    """Delta^pred(K, b) — average prediction change when prefix -> b."""
    n, W, D = tilde_hs.shape
    prefix_len = W - K
    assert baseline.shape == (prefix_len, D)
    total = 0.0
    for i in range(n):
        y_full = f(tilde_hs[i])
        padded = np.vstack([baseline, tilde_hs[i][-K:]])
        y_surr = f(padded)
        if loss == "classification":
            total += float(
                not np.array_equal(np.atleast_1d(y_full), np.atleast_1d(y_surr))
            )
        elif loss == "regression":
            total += float(
                np.mean(np.abs(np.atleast_1d(y_full) - np.atleast_1d(y_surr)))
            )
        else:
            raise ValueError(f"Unknown loss '{loss}'")
    return total / n


def compute_delta_f(
    est_K: FactoredKernelEstimator,
    est_K1: FactoredKernelEstimator,
    pi_star: dict,
    disc: Discretiser,
) -> float:
    """Factored TV stopping criterion Delta^f(K), averaged over D."""
    D = disc.D_
    delta = 0.0
    for h_idx, pi_h in pi_star.items():
        if pi_h == 0.0:
            continue
        tv_h = (
            sum(
                0.5
                * np.sum(
                    np.abs(
                        est_K.predict_marginal(d, h_idx)
                        - est_K1.predict_marginal(d, h_idx)
                    )
                )
                for d in range(D)
            )
            / D
        )
        delta += pi_h * tv_h
    return delta


def _cache_outputs(f: Callable, tilde_hs: np.ndarray) -> np.ndarray:
    """f(tilde_hs[i]) for every i, via one batched oracle call.

    Every oracle in this codebase accepts (B, W, D) and returns (B, ...)
    (see karma/utils/oracle.py, comparison_realdata.py's f_oracle, etc.), so
    dispatching all n windows together is both correct and far cheaper than
    n individual calls — the oracle call (often a torch forward pass) is the
    actual cost here, not the surrounding numpy bookkeeping.
    """
    return np.asarray(f(tilde_hs))


def _pad_prefix(b: np.ndarray, suffix: np.ndarray) -> np.ndarray:
    """Concatenate baseline b onto each row of a batched suffix.

    b       : (W-K, D)
    suffix  : (n, K, D)
    returns : (n, W, D), row i = vstack([b, suffix[i]])
    """
    n = suffix.shape[0]
    b_batch = np.broadcast_to(b, (n,) + b.shape)
    return np.concatenate([b_batch, suffix], axis=1)


def select_K_and_baseline(
    f: Callable,
    X_train: np.ndarray,
    X_val: np.ndarray,
    disc: Discretiser,
    W: int,
    eps: float = 0.05,
    q: float = 0.10,
    delta: float = 0.1,
    ell: Optional[Callable] = None,
    loss: str = "regression",
    K_max: Optional[int] = None,
    n_obs_candidates: int = 10,
    seed: int = 0,
    verbose: bool = True,
    cache_path: Optional[str] = None,
) -> dict:
    """
    Select Markov order K* and admissible baseline b* via A1–A3 admissibility.

    For each K = 1..K_max, builds a baseline family B(K) and returns the
    smallest K for which at least one member passes all of A1–A3. Prefers
    b_marg when multiple members pass; otherwise picks by largest combined margin.

    Stopping conditions (all three must hold)
    ------------------------------------------
    A1 — average insensitivity:  Δ̂_pred(K, b) ≤ eps
    A2 — worst-q diversity:      mean loss over top-q prefix-distant queries ≤ eps
    A3 — responsiveness:         mean response to suffix perturbation ≥ eps

    Parameters
    ----------
    f               : oracle callable (W, D) -> prediction
    X_train         : training series, shape (T_train, D)
    X_val           : held-out series, shape (T_val, D)
    disc            : fitted Discretiser
    W               : model input window size
    eps             : tolerance shared by A1, A2 (≤ eps) and A3 (≥ eps)
    q               : worst-matched fraction for A2 (e.g. 0.10 = hardest decile)
    delta           : suffix perturbation std for A3
    ell             : metric ℓ(y1, y2) -> R≥0; derived from loss if None
    loss            : "regression" or "classification" (used when ell is None)
    K_max           : maximum order to search; defaults to W
    n_obs_candidates: number of b_obs candidates sampled from X_train
    seed            : RNG seed for b_marg draws and A3 perturbations
    verbose         : print per-(K, b) diagnostics

    Returns
    -------
    dict with keys:
        certified         : bool   True iff an admissible (K*, b*) was found
        K_star            : int
        b_star            : ndarray (W-K*, D)
        b_star_type       : str in {obs, marg, argmin, zeros, mean, central_bin}
        b_star_name       : str
        Delta_A1          : float
        Delta_far_A2      : float
        R_A3              : float
        eps, q, delta     : float (echoed)
        certified_zero_bound : 2*eps  (E[phi_k] <= 2*eps for k > K*)
        family_agreement  : bool (b_obs and b_marg both passed independently)
        pi_star           : dict  h_idx -> float  stationary weights at K*
        best_attempt      : dict or None (best record when certified=False)
        seed              : int
        history           : list of per-(K, b) log records
    """
    import pickle, os as _os

    if cache_path is not None and _os.path.exists(cache_path):
        if verbose:
            print(f"  Loading cached result from {cache_path}")
        with open(cache_path, "rb") as _fh:
            return pickle.load(_fh)

    if ell is None:
        ell = _make_ell(loss)

    rng = np.random.default_rng(seed)
    K_max = K_max or W

    if X_val.ndim == 3:
        # Pre-windowed: (n, W, D)
        tilde_hs = X_val.astype(np.float32) if X_val.dtype != np.float32 else X_val
        _, W_data, D = tilde_hs.shape
        if W_data != W:
            raise ValueError(f"X_val window size {W_data} != W={W}")
    else:
        # Raw time series: (T_val, D)
        T_val, D = X_val.shape
        n_q = T_val - W
        if n_q <= 0:
            raise ValueError(f"X_val too short: need T_val > W={W}, got T_val={T_val}")
        tilde_hs = np.stack([X_val[t : t + W] for t in range(n_q)])

    cache = _cache_outputs(f, tilde_hs)

    n_windows = tilde_hs.shape[0]
    if verbose:
        print(
            f"select_K_and_baseline: n={n_windows}, W={W}, D={D}, K_max={K_max}, seed={seed}"
        )

    history_log: List[dict] = []
    best_overall: Optional[dict] = None
    best_overall_b: Optional[np.ndarray] = None
    pi_star_K: dict = {}

    def _max_violation(rec: dict) -> float:
        v = []
        if not rec["pass_A1"]:
            v.append(rec["Delta_A1"] - eps)
        if not rec["pass_A2"]:
            v.append(rec["Delta_far_A2"] - eps)
        if not rec["pass_A3"]:
            v.append(eps - rec["R_A3"])
        return max(v) if v else 0.0

    for K in tqdm(range(1, K_max + 1), desc="K search", disable=not verbose):
        pi_star_K = compute_pi_star(X_train, disc, K)
        family = _build_baseline_family(
            X_train,
            f,
            K,
            W,
            rng,
            disc=disc,
        )
        passing: List[dict] = []
        prefix_len = W - K
        n_windows = tilde_hs.shape[0]
        n_far = max(1, int(np.ceil(q * n_windows)))

        # ── Phase 1: A1 + A2 for all baselines (no extra oracle calls) ───────
        # _per_query_losses reuses `cache` for f(tilde_hs[i]); only f(b⊕h_i) is new.
        all_losses = np.stack(
            [
                _per_query_losses(f, tilde_hs, K, entry["b"], ell, cache=cache)
                for entry in family
            ]
        )  # (|family|, n)

        Delta_A1_all = all_losses.mean(axis=1)  # (|family|,)
        pass_A1_all = Delta_A1_all <= eps

        if prefix_len > 0:
            # Vectorised prefix distances: (n, |family|)
            prefixes = tilde_hs[:, :prefix_len]  # (n, prefix_len, D)
            B_stack = np.stack([e["b"] for e in family])  # (|family|, prefix_len, D)
            r_all = np.linalg.norm(
                prefixes[:, np.newaxis] - B_stack[np.newaxis], axis=(2, 3)
            )  # (n, |family|)
            far_mask = np.argsort(r_all, axis=0)[-n_far:]  # (n_far, |family|)
            Delta_far_A2_all = np.array(
                [all_losses[j, far_mask[:, j]].mean() for j in range(len(family))]
            )
        else:
            Delta_far_A2_all = np.zeros(len(family))
        pass_A2_all = Delta_far_A2_all <= eps

        pre_pass = pass_A1_all & pass_A2_all  # (|family|,)

        # ── Phase 2: A3 only for baselines surviving A1+A2 ───────────────────
        for j, entry in enumerate(family):
            b_name, b = entry["name"], entry["b"]

            if pre_pass[j]:
                noise = (
                    rng.normal(0.0, delta, (n_windows, K, D))
                    if K > 0
                    else np.empty((n_windows, 0, D))
                )
                suffix = tilde_hs[:, W - K :]
                y_hi = np.asarray(f(_pad_prefix(b, suffix)))
                y_hip = np.asarray(f(_pad_prefix(b, suffix + noise)))
                R_vals = np.array(
                    [ell(y_hi[i], y_hip[i]) for i in range(n_windows)]
                )
                R_A3 = float(R_vals.mean())
                pass_A3 = R_A3 >= eps
            else:
                R_A3, pass_A3 = 0.0, False

            rec = {
                "pass_A1": bool(pass_A1_all[j]),
                "pass_A2": bool(pass_A2_all[j]),
                "pass_A3": pass_A3,
                "Delta_A1": float(Delta_A1_all[j]),
                "Delta_far_A2": float(Delta_far_A2_all[j]),
                "R_A3": R_A3,
            }
            is_adm = bool(pre_pass[j]) and pass_A3

            log = {
                "K": K,
                "b_name": b_name,
                "is_admissible": is_adm,
                "memory_fraction": K / W,
                "n_observed_histories": len(pi_star_K),
                **rec,
            }
            history_log.append(log)

            if is_adm:
                passing.append({"b_name": b_name, "b": b, "rec": rec})

            if best_overall is None or _max_violation(log) < _max_violation(
                best_overall
            ):
                best_overall = log
                best_overall_b = b

        if passing:
            chosen = next((e for e in passing if e["b_name"] == "b_marg"), None)
            if chosen is None:

                def _margin(e: dict) -> float:
                    r = e["rec"]
                    return min(eps - r["Delta_far_A2"], r["R_A3"] - eps)

                chosen = max(passing, key=_margin)

            family_agreement = (
                sum(
                    1
                    for e in passing
                    if "b_obs" in e["b_name"] or e["b_name"] == "b_marg"
                )
                >= 2
            )

            rec = chosen["rec"]
            b_type = (
                "marg"
                if chosen["b_name"] == "b_marg"
                else (
                    "argmin"
                    if chosen["b_name"] == "b_argmin"
                    else (
                        "obs"
                        if "b_obs" in chosen["b_name"]
                        else chosen["b_name"].removeprefix("b_")
                    )
                )
            )

            if verbose:
                print(f"\n  => K* = {K}, b* = {chosen['b_name']}")
                print(f"     Compression: W/K* = {W}/{K} = {W/K:.1f}x")

            _result = {
                "certified": True,
                "K_star": K,
                "b_star": chosen["b"],
                "b_star_type": b_type,
                "b_star_name": chosen["b_name"],
                "Delta_A1": rec["Delta_A1"],
                "Delta_far_A2": rec["Delta_far_A2"],
                "R_A3": rec["R_A3"],
                "eps": eps,
                "q": q,
                "delta": delta,
                "certified_zero_bound": 2 * eps,
                "family_agreement": family_agreement,
                "pi_star": pi_star_K,
                "best_attempt": None,
                "seed": seed,
                "history": history_log,
            }
            if cache_path is not None:
                _os.makedirs(
                    _os.path.dirname(_os.path.abspath(cache_path)), exist_ok=True
                )
                with open(cache_path, "wb") as _fh:
                    pickle.dump(_result, _fh)
                if verbose:
                    print(f"  Result cached to {cache_path}")
            return _result

    if verbose:
        if best_overall:
            failed = [
                c
                for c, ok in [
                    ("A1", best_overall["pass_A1"]),
                    ("A2", best_overall["pass_A2"]),
                    ("A3", best_overall["pass_A3"]),
                ]
                if not ok
            ]
            print(
                f"\n  WARNING: compression achieved but zero-attribution not certifiable "
                f"for lags > K* ({', '.join(failed)} failed at best candidate "
                f"K={best_overall['K']}, b={best_overall['b_name']}). "
                f"Consider increasing K_max or eps."
            )
        else:
            print(f"\n  WARNING: no baseline evaluated up to K_max={K_max}.")

    _result = {
        "certified": False,
        "K_star": K_max,
        "b_star": best_overall_b,  # best uncertified baseline; never None
        "b_star_type": None,
        "b_star_name": best_overall["b_name"] if best_overall else None,
        "Delta_A1": best_overall["Delta_A1"] if best_overall else None,
        "Delta_far_A2": best_overall["Delta_far_A2"] if best_overall else None,
        "R_A3": best_overall["R_A3"] if best_overall else None,
        "eps": eps,
        "q": q,
        "delta": delta,
        "certified_zero_bound": None,
        "family_agreement": False,
        "pi_star": pi_star_K,
        "best_attempt": best_overall,
        "seed": seed,
        "history": history_log,
    }
    if cache_path is not None:
        _os.makedirs(_os.path.dirname(_os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as _fh:
            pickle.dump(_result, _fh)
        if verbose:
            print(f"  Result cached to {cache_path}")
    return _result


# ─── A1–A3 Admissibility: Principled Order & Baseline Selection ───────────────


def _make_ell(loss: str) -> Callable:
    """Convert a loss string to a metric ell(y1, y2) -> float."""
    if loss == "classification":
        return lambda y1, y2: float(
            not np.array_equal(np.atleast_1d(y1), np.atleast_1d(y2))
        )
    elif loss == "regression":
        return lambda y1, y2: float(
            np.mean(np.abs(np.atleast_1d(y1) - np.atleast_1d(y2)))
        )
    else:
        raise ValueError(f"Unknown loss '{loss}'")


def _per_query_losses(
    f: Callable,
    tilde_hs: np.ndarray,
    K: int,
    b: np.ndarray,
    ell: Callable,
    cache: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-query ℓ(f(h̃_i), f(b ⊕ h_i)), shape (n,).

    The n "full" and n "surrogate" (b-padded) queries are each dispatched to
    f in a single batched call — not one call per query — per this codebase's
    batched-oracle convention (see _cache_outputs). ell is still applied one
    query at a time since it's user-supplied and not assumed to be
    batch-aware; that loop is cheap relative to the oracle calls it replaces.
    """
    n, W, _ = tilde_hs.shape
    y_full = cache if cache is not None else np.asarray(f(tilde_hs))
    y_surr = np.asarray(f(_pad_prefix(b, tilde_hs[:, W - K :])))
    return np.array([ell(y_full[i], y_surr[i]) for i in range(n)])


def _build_baseline_family(
    X_train: np.ndarray,
    f: Callable,
    K: int,
    W: int,
    rng: np.random.Generator,
    disc: Optional["Discretiser"] = None,
) -> List[dict]:
    """
    Build baseline family B(K).

    b_obs        : observed prefixes sampled from X_train
    b_marg       : each prefix slot drawn independently from X_train marginal
    b_argmin     : b_obs candidate with lowest Δ̂_pred (discrepancy-optimal)
    b_zeros      : all-zeros prefix
    b_mean       : empirical mean prefix from X_train
    b_central_bin: central-bin prefix from disc (only when disc is provided)
    """
    T, D = X_train.shape
    prefix_len = W - K
    family: List[dict] = []

    if prefix_len == 0:
        return [{"name": "none (K=W)", "b": np.zeros((0, D))}]

    # 2. b_marg: each slot drawn independently from X_train marginal
    slot_indices = rng.integers(0, T, size=prefix_len)
    b_marg = X_train[slot_indices].copy()
    family.append({"name": "b_marg", "b": b_marg})

    # 4. b_zeros: constant zero prefix
    family.append({"name": "b_zeros", "b": np.zeros((prefix_len, D))})

    # 5. b_mean: empirical mean of all observed prefixes in X_train
    if T >= prefix_len:
        prefixes = np.stack(
            [X_train[t : t + prefix_len] for t in range(T - prefix_len + 1)]
        )
        family.append({"name": "b_mean", "b": prefixes.mean(axis=0)})

    # 6. b_central_bin: central-bin prefix from discretiser (when available)
    if disc is not None:
        N = disc.N
        mid = np.zeros(D)
        for d in range(D):
            bounds = disc.quantiles_[d]
            c = N // 2
            lo = bounds[c - 1] if c > 0 else bounds[0] - 1.0
            hi = bounds[c] if c < N - 1 else bounds[-1] + 1.0
            mid[d] = 0.5 * (lo + hi)
        family.append({"name": "b_central_bin", "b": np.tile(mid, (prefix_len, 1))})

    return family


def admissible(
    f: Callable,
    tilde_hs: np.ndarray,
    K: int,
    b: np.ndarray,
    eps: float,
    gamma: float,
    q: float,
    delta: float,
    ell: Callable,
    rng: Optional[np.random.Generator] = None,
    cache: Optional[np.ndarray] = None,
) -> Tuple[bool, dict]:
    """
    Check admissibility conditions A1–A3 for a candidate (K, b).

    A1 — Interventional insensitivity (average):
        Δ̂_pred(K, b) ≤ eps

    A2 — Prefix diversity (worst-matched q-fraction):
        Mean loss over top-q queries by prefix distance to b is ≤ eps.
        Guards against a small average carried only by prefixes near b.

    A3 — Responsiveness (dead-zone guard):
        Mean response to suffix perturbation of size delta is ≥ gamma.
        Confirms f is not flat at the substituted points.

    Parameters
    ----------
    f         : oracle callable
    tilde_hs  : (n, W, D) held-out query windows
    K         : candidate Markov order (suffix length)
    b         : candidate baseline, shape (W-K, D)
    eps       : insensitivity tolerance
    gamma     : responsiveness floor (must be > eps at caller)
    q         : worst-matched fraction for A2
    delta     : suffix perturbation std for A3
    ell       : metric ℓ(y1, y2) -> float
    rng       : numpy Generator for A3 perturbation (defaults to seed 42)
    cache     : precomputed f(tilde_hs[i]) to avoid redundant calls

    Returns
    -------
    (is_admissible, record)
    record keys: pass_A1, pass_A2, pass_A3, Delta_A1, Delta_far_A2, R_A3
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n, W, D = tilde_hs.shape
    prefix_len = W - K

    # A1
    per_loss = _per_query_losses(f, tilde_hs, K, b, ell, cache=cache)
    Delta_A1 = float(per_loss.mean())
    pass_A1 = Delta_A1 <= eps

    # A2: worst-q fraction by L2 prefix distance
    if prefix_len > 0:
        r_i = np.array([np.linalg.norm(tilde_hs[i, :prefix_len] - b) for i in range(n)])
        n_far = max(1, int(np.ceil(q * n)))
        far_idx = np.argsort(r_i)[-n_far:]
        Delta_far_A2 = float(per_loss[far_idx].mean())
    else:
        Delta_far_A2 = 0.0
    pass_A2 = Delta_far_A2 <= eps

    # A3: responsiveness to suffix perturbation
    noise = rng.normal(0.0, delta, size=(n, K, D)) if K > 0 else np.empty((n, 0, D))
    R_vals = np.zeros(n)
    for i in range(n):
        suffix_orig = tilde_hs[i, W - K :]
        b_hi = np.vstack([b, suffix_orig])
        b_hip = np.vstack([b, suffix_orig + noise[i]])
        R_vals[i] = ell(np.atleast_1d(f(b_hi)), np.atleast_1d(f(b_hip)))
    R_A3 = float(R_vals.mean())
    pass_A3 = R_A3 >= gamma

    record = {
        "pass_A1": pass_A1,
        "pass_A2": pass_A2,
        "pass_A3": pass_A3,
        "Delta_A1": Delta_A1,
        "Delta_far_A2": Delta_far_A2,
        "R_A3": R_A3,
    }
    return pass_A1 and pass_A2 and pass_A3, record
