"""Beijing multi-task evaluation: REAL-WORLD
GENERALITY, with N-fold cross-validation over the evaluation set.

One dataset (Beijing PM2.5 air quality), three task framings, one representative method each --
to show OMIC applies ACROSS TASK TYPES, not to run a method leaderboard (every cell below is a
different method on a different task; do not read this as "method X beats Y").

  - Forecasting / KARMA:        multivariate next-step forecasting (the SAME task/model
                                 experiments/comparison_realdata.py already trains for Beijing);
                                 KARMA's own edge-level transition structure -> claimed edges.
                                 The claim is training-derived, so it is FOLD-INVARIANT (fit
                                 once); only its OMIC/mediation evaluation is folded.
  - Regression / TimeSHAP:      TimeSHAP already reduces a forecaster's D-dimensional output to
                                 a single scalar via _multistep_target(...).sum(dim=-1) before
                                 explaining it (see run_timeshap) -- exactly what a scalar
                                 regression target needs, so this task reuses the SAME trained
                                 forecaster, with no separate scalar-target model. TimeSHAP is an
                                 INSTANCE-level explainer, so its own attribution (not just its
                                 OMIC evaluation) is recomputed fresh on EACH fold's windows.
  - Classification / TIMING:    Beijing has no natural classification target, so this task
                                 defines one: a binary "next-step PM2.5 exceeds its train-set
                                 75th percentile" event, a minimal GRU-encoder + single-logit
                                 head classifier (BeijingClassifier, below), and a dedicated
                                 TIMING wrapper (run_timing_classification) that attributes
                                 ONE logit directly via teig_telrp's _timing_segment_ig core --
                                 NOT run_timing itself, which loops target_channel over
                                 range(D) assuming a D-dimensional regression output (a
                                 pre-existing mismatch for a single-logit classifier, not
                                 something this script's new code inherits). Also
                                 INSTANCE-level -> recomputed per fold, like TimeSHAP.

CV convention (matches experiments/comparison_realdata_cv.py exactly): the evaluation pool is
split into `n_folds` non-overlapping contiguous chunks (`_fold_slices`); models, KARMA, and the
classifier are fit ONCE, never retrained per fold; every reported number is a fold mean +/-std
(`_aggregate_task_folds`). `n_folds: 1` in configs/default.yaml reproduces the original
single-eval-set behavior exactly.

Beijing has no analytic oracle (unlike experiments/run_var_omic.py's Block B1 VAR), so Delta
uses dagfaith.real_data.empirical_cond_sampler -- an ESTIMATED conditional (linear-Gaussian fit
off the observed windows), fit once and shared across all three tasks. This adds a real
estimation-error source Block B1's exact analytic conditional doesn't have; report results with
that caveat, not as directly comparable numbers to B1.

Outputs per task: fig_beijing_{clf,fc,reg}_{omic,auomic,edges,med}.pdf; tab_beijing.tex.

EXPLANATION CACHING (across multiple runs of this script, and within one run): KARMA's fit,
TimeSHAP's per-fold attribution and TIMING's per-fold attribution are the expensive, mostly-
deterministic-given-(seed, fold, nsamples) parts of every run -- re-running this script (e.g.
to iterate on the downstream OMIC/E1-E4 analysis) otherwise re-fits/re-explains from scratch
every time. Cached under `<ckpt_dir>/explanations/*.pkl`, keyed by a run tag (quick/full) plus
the seed/fold/nsamples that actually determine the result; `--no_cache` disables both reading
and writing it (forces a fresh compute, e.g. after changing an explainer's own code). E3 reuses
the SAME cache keys as the main per-fold TimeSHAP/TIMING calls, so it loads fold 0's own
attribution instead of recomputing it a second time even within a single run.

Usage:
    python experiments/run_beijing_multitask.py [--config configs/default.yaml] [--quick] [--no_cache]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dagfaith.config import results_dir as dagfaith_results_dir
from dagfaith.config import load_config
from dagfaith.intervention import delta_dict
from dagfaith.intervention_tot import delta_tot_dict
from dagfaith.metrics_tot import mediation, evaluate_dir_tot
from dagfaith.omic import auomic
from dagfaith.real_data import empirical_cond_sampler

from experiments.comparison_realdata import (
    DATASETS,
    _multistep_target,
    _scalar_numpy_forward,
    _timing_segment_ig,
    get_or_train_gru,
    karma_edges_to_attr,
    load_dataset,
    run_ig,
    run_karma,
    run_timeshap,
    set_global_seed,
)
from dagfaith.omic import omic_ranking_curve

PM25_INDEX = 0  

def _aggregate_folds(fold_results: list[dict]) -> dict:
    """Given a list of per-fold metric dicts, return {key_mean, key_std, key_folds}."""
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for fd in fold_results:
        for k, v in fd.items():
            if v is not None:
                buckets[k].append(float(v))

    agg: dict = {}
    for k, vals in buckets.items():
        arr = np.array(vals)
        agg[f"{k}_mean"] = float(arr.mean())
        agg[f"{k}_std"] = float(arr.std(ddof=0))
        agg[f"{k}_folds"] = vals
    return agg

def _explanation_cache_dir(ckpt_dir: Path) -> Path:
    d = ckpt_dir / "explanations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_explanation(ckpt_dir: Path, key: str, use_cache: bool):
    """Returns the cached object for `key`, or None if not cached / caching is disabled."""
    if not use_cache:
        return None
    path = _explanation_cache_dir(ckpt_dir) / f"{key}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_explanation(ckpt_dir: Path, key: str, obj, use_cache: bool) -> None:
    if not use_cache:
        return
    path = _explanation_cache_dir(ckpt_dir) / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class BeijingClassifier(nn.Module):
    """Minimal GRU encoder + a single linear head -> ONE scalar logit (binary classification of
    a next-step PM2.5-exceedance event) for Block B2's classification/TIMING task.

    `forward(x, return_all=False)` accepts (and ignores) the same `return_all` kwarg
    GRUForecastModel/comparison_realdata.py's other models expose, purely so this class is a
    drop-in for `_multistep_target`'s fallback branch (`model(x, return_all=False)`) and hence
    for `run_attribution_omic`/`_scalar_numpy_forward` unchanged -- there is no "return_all"
    variant for a single classification target, both branches return the same (B, 1) logit.
    """

    def __init__(self, D: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(D, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)
        self.forecast_steps = 1  # so _multistep_target's `forecast_steps > 1` branch is skipped

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """x: (B, D, T) -> (B, 1) logit."""
        x_td = x.permute(0, 2, 1)
        out, _ = self.gru(x_td)
        return self.head(out[:, -1, :])


def train_classifier(
    X_train: np.ndarray, y_train_bin: np.ndarray, X_val: np.ndarray, y_val_bin: np.ndarray,
    cfg, device, epochs: int,
) -> tuple[BeijingClassifier, float]:
    """Train BeijingClassifier with BCEWithLogitsLoss; returns (model, val_accuracy) -- report
    accuracy so the reader can judge whether TIMING's attribution reflects a genuine finding or
    a merely-incompetent classifier (the same honesty guard run_trained.py's val_error carries)."""
    X_tr = torch.from_numpy(X_train.transpose(0, 2, 1).astype(np.float32))
    y_tr = torch.from_numpy(y_train_bin.astype(np.float32))
    X_v = torch.from_numpy(X_val.transpose(0, 2, 1).astype(np.float32))
    y_v = torch.from_numpy(y_val_bin.astype(np.float32))

    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=cfg.batch_size)

    model = BeijingClassifier(D=X_train.shape[2], hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val, best_state, best_acc = float("inf"), None, 0.0
    for _epoch in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
                val_losses.append(criterion(logits, yb).item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += int((preds == yb).sum().item())
                total += yb.numel()
        vl = float(np.mean(val_losses))
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_acc = correct / max(total, 1)

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def run_timing_classification(x_test: np.ndarray, classifier: BeijingClassifier, device, n_samples: int = 50) -> np.ndarray:
    """TIMING attribution for a SINGLE-LOGIT classifier -> (D, T). Unlike run_timing (which loops
    target_channel over range(D) for a D-dimensional regression forecaster), a binary classifier
    has exactly one output channel: one _timing_segment_ig call, not a loop+stack -- calling
    run_timing here would be wrong (target_channel > 0 has nothing to index)."""
    classifier.eval()
    x = torch.from_numpy(x_test.astype(np.float32)).to(device)
    forward_func = lambda xin: classifier(xin)  # noqa: E731  -- (B, D, T) -> (B, 1)
    attr = _timing_segment_ig(forward_func, x, target_channel=0, n_samples=n_samples, num_segments=3, min_seg_len=1, max_seg_len=1)
    return attr.abs().mean(dim=0).detach().cpu().numpy()  # (D, T)


def _grad_x_input(model, x_fold_dt: np.ndarray, device, tau=None) -> np.ndarray:
    """Gradient x input -> (D, T), |.|, averaged over the batch -- the standard weak reference
    (E1), distinct from IG (a path integral) and from KARMA/TimeSHAP/TIMING (this repo's real
    explainers)."""
    model.eval()
    x = torch.from_numpy(x_fold_dt.astype(np.float32)).to(device).requires_grad_(True)
    out = _multistep_target(model, x, tau).sum(dim=-1)  # (B,)
    (grad,) = torch.autograd.grad(out.sum(), x)
    return (grad * x).abs().mean(dim=0).detach().cpu().numpy()  # (D, T)


def _ig_zero(model, x_fold_dt: np.ndarray, device, tau=None) -> np.ndarray:
    """IG with a ZERO baseline -> (D, T), |.|, averaged over the batch -- reuses
    experiments.comparison_realdata.run_ig directly (it already IS zero-baseline IG); doubles
    as E1's "off-manifold contrast the framework exists to criticise" reference."""
    attr = run_ig(x_fold_dt, model, device, tau=tau)  # (B, D, T), already abs()
    return attr.mean(axis=0)


def run_e1_baselines(
    model, x_fold_dt: np.ndarray, device, D: int, T: int,
    candidate_edges: list, delta_dir_all: dict, delta_tot_all: dict, n_plus: int,
    rho_max: float, seed: int, tau=None,
) -> dict:
    """RANDOM / grad-x-input / IG-zero / conditional-occlusion, each claiming `n_plus` edges
    (the SAME size as the real method's own claim) and scored via `evaluate_dir_tot` against
    the SAME delta_dir_all/delta_tot_all/candidate_edges. Conditional occlusion (attribution =
    |Delta_dir| itself) shares machinery with the reference and therefore UPPER-BOUNDS what any
    explainer can score on Delta_dir -- without it the real method's row can't be interpreted."""
    rng = np.random.default_rng(seed)
    results = {}

    random_claimed = [candidate_edges[k] for k in rng.choice(len(candidate_edges), size=n_plus, replace=False)]
    random_attr = {e: float(rng.random()) for e in random_claimed}
    results["RANDOM"] = evaluate_dir_tot(random_claimed, random_attr, delta_dir_all, delta_tot_all, candidate_edges, rho_max=rho_max)

    gxi = _grad_x_input(model, x_fold_dt, device, tau)
    gxi_attr = {(i, t, 0): float(gxi[i, t]) for i in range(D) for t in range(T)}
    gxi_claimed = sorted(candidate_edges, key=lambda e: gxi_attr[e], reverse=True)[:n_plus]
    results["GRAD_X_INPUT"] = evaluate_dir_tot(gxi_claimed, gxi_attr, delta_dir_all, delta_tot_all, candidate_edges, rho_max=rho_max)

    ig = _ig_zero(model, x_fold_dt, device, tau)
    ig_attr = {(i, t, 0): float(ig[i, t]) for i in range(D) for t in range(T)}
    ig_claimed = sorted(candidate_edges, key=lambda e: ig_attr[e], reverse=True)[:n_plus]
    results["IG_ZERO"] = evaluate_dir_tot(ig_claimed, ig_attr, delta_dir_all, delta_tot_all, candidate_edges, rho_max=rho_max)

    occ_attr = {e: float(abs(delta_dir_all[e])) for e in candidate_edges}
    occ_claimed = sorted(candidate_edges, key=lambda e: occ_attr[e], reverse=True)[:n_plus]
    results["COND_OCCLUSION"] = evaluate_dir_tot(occ_claimed, occ_attr, delta_dir_all, delta_tot_all, candidate_edges, rho_max=rho_max)

    return results


def oracle_ceiling_curve(
    claimed_edges: list, delta_dir_all: dict, delta_dir_oracle_all: dict, candidate_edges: list, rho_max: float,
):
    """Rank `claimed_edges` by a FRESH, independent Delta_dir draw (delta_dir_oracle_all), then
    score that ranking against the ORIGINAL delta_dir_all -- both are unbiased estimates of the
    SAME true Delta_dir, so the gap between this curve and 1.0 is pure Monte-Carlo/estimation
    noise, not a real ranking failure. This is the curve any real explainer's own OMIC_dir
    curve should be read against, not against a theoretical 1.0."""
    oracle_attribution = {e: abs(delta_dir_oracle_all[e]) for e in claimed_edges}
    return omic_ranking_curve(claimed_edges, oracle_attribution, delta_dir_all, candidate_edges, rho_max=rho_max)


def _fold_slices(N: int, n_folds: int) -> list[tuple[int, int]]:
    """Non-overlapping contiguous fold boundaries over a length-N pool -- the SAME convention
    experiments/comparison_realdata_cv.py uses for its own test-set folds (equal-sized chunks,
    last fold absorbs the remainder), so CV numbers here are read the same way."""
    fold_size = N // n_folds
    return [
        (f * fold_size, (f + 1) * fold_size if f < n_folds - 1 else N)
        for f in range(n_folds)
    ]


def _load_beijing(seed: int, n_eval: int, n_folds: int, karma_overrides: dict | None = None):
    """Returns an EVAL POOL of up to n_eval*n_folds windows (capped at the test set size),
    later split into n_folds folds by `_fold_slices` -- models/KARMA/classifier are fit ONCE
    (outside any fold loop, see each task runner), only the OMIC evaluation set is folded,
    matching comparison_realdata_cv.py's "models not retrained per fold" convention."""
    cfg = DATASETS["beijing_pm25"]
    if karma_overrides:
        from dataclasses import replace

        cfg = replace(cfg, **karma_overrides)
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(cfg)
    rng = np.random.default_rng(seed)
    pool_size = min(n_eval * n_folds, len(X_test))
    X_eval_pool = X_test[rng.choice(len(X_test), size=pool_size, replace=False)]
    cond_sampler = empirical_cond_sampler(X_train, seed=seed)
    return cfg, X_train, X_val, X_test, y_train, y_val, X_eval_pool, cond_sampler

def _fold_dir_tot(
    f_np, x_fold: np.ndarray, cond_sampler, D: int, T: int,
    attribution: dict, claimed: set[tuple[int, int]], delta_B: int, rho_max: float, seed: int,
) -> dict:
    candidate_edges = [(i, t, 0) for i in range(D) for t in range(T)]
    claimed_edges = [(i, t, 0) for (i, t) in claimed]

    rng_dir = np.random.default_rng(seed)
    rng_tot = np.random.default_rng(seed + 500)
    delta_dir_all = delta_dict(f_np, x_fold, candidate_edges, cond_sampler, B=delta_B, rng=rng_dir)
    delta_tot_all = delta_tot_dict(f_np, x_fold, candidate_edges, cond_sampler, D, T, B=delta_B, rng=rng_tot)

    dir_tot = evaluate_dir_tot(claimed_edges, attribution, delta_dir_all, delta_tot_all, candidate_edges, rho_max=rho_max)
    conf = mediation(delta_dir_all, delta_tot_all, claimed_edges=claimed_edges)

    non_boundary_claimed = [e for e in claimed_edges if e[1] != T - 1]
    conf_excl = mediation(delta_dir_all, delta_tot_all, claimed_edges=non_boundary_claimed) if non_boundary_claimed else conf

    rng_dir_oracle = np.random.default_rng(seed + 999)
    delta_dir_oracle_all = delta_dict(f_np, x_fold, candidate_edges, cond_sampler, B=delta_B, rng=rng_dir_oracle)
    _rho_oracle, omic_curve_oracle = oracle_ceiling_curve(claimed_edges, delta_dir_all, delta_dir_oracle_all, candidate_edges, rho_max)

    return {
        "omic_support": dir_tot["omic_support_dir"], "auomic": dir_tot["auomic_dir"],
        "rho": dir_tot["rho_dir"], "omic_curve": dir_tot["omic_curve_dir"], "omic_curve_tot": dir_tot["omic_curve_tot"],
        "omic_support_dir": dir_tot["omic_support_dir"], "omic_support_tot": dir_tot["omic_support_tot"],
        "auomic_dir": dir_tot["auomic_dir"], "auomic_tot": dir_tot["auomic_tot"],
        "gap": dir_tot["gap"], "mediation_fraction": conf["mediation_fraction"],
        "mediation_fraction_excl_boundary": conf_excl["mediation_fraction"],
        "med": conf["med"], "n_plus": len(claimed_edges),
        "omic_curve_oracle": omic_curve_oracle,
        "delta_dir_all": delta_dir_all, "delta_tot_all": delta_tot_all, "candidate_edges": candidate_edges,
        "claimed_edges": claimed_edges, "x_fold": x_fold,
    }


_RAW_FOLD_KEYS = (
    "rho", "omic_curve", "omic_curve_tot", "omic_curve_oracle", "med",
    "delta_dir_all", "delta_tot_all", "candidate_edges", "claimed_edges", "x_fold",
)


def _aggregate_task_folds(fold_results: list[dict]) -> dict:
    """{key}_mean/{key}_std across folds for scalar metrics (comparison_realdata_cv.py's own
    `_aggregate_folds` convention, reused directly) + separately-averaged rho/omic_curve*/med
    (arrays/dicts, not scalars -- `_aggregate_folds` would choke on `float(array)`). The raw
    per-fold delta dicts/candidate grid/x_fold are NOT aggregated here (no meaningful "average
    Delta dict") -- callers needing them read
    `fold_results` directly, typically from just the first fold (matching run_var_omic.py's own
    "first seed" convention for its own per-edge snapshot plots)."""
    scalar_keys = {k for k in fold_results[0] if k not in _RAW_FOLD_KEYS}
    scalar_folds = [{k: fd[k] for k in scalar_keys} for fd in fold_results]
    agg = _aggregate_folds(scalar_folds)

    agg["rho"] = fold_results[0]["rho"]  # same length/values every fold (n_plus is fold-invariant)
    for key in ("omic_curve", "omic_curve_tot", "omic_curve_oracle"):
        curves = np.array([fd[key] for fd in fold_results])
        agg[f"{key}_mean"] = np.nanmean(curves, axis=0)
        agg[f"{key}_std"] = np.nanstd(curves, axis=0)
        agg[f"{key}_folds"] = curves

    all_edges = set().union(*(fd["med"].keys() for fd in fold_results))
    agg["med_mean"] = {
        e: float(np.nanmean([fd["med"].get(e, np.nan) for fd in fold_results])) for e in all_edges
    }
    return agg


def run_forecasting_task(seed: int, n_eval: int, n_folds: int, ckpt_dir: Path, device, omic_kwargs: dict, delta_B: int, karma_overrides: dict | None = None, run_tag: str = "full", use_cache: bool = True) -> dict:
    cfg, X_train, X_val, _X_test, y_train, y_val, X_eval_pool, cond_sampler = _load_beijing(seed, n_eval, n_folds, karma_overrides)
    model = get_or_train_gru(cfg, X_train, y_train, X_val, y_val, device, ckpt_dir)

    karma_key = f"karma_{run_tag}_seed{seed}"
    cached = load_explanation(ckpt_dir, karma_key, use_cache)
    if cached is not None:
        edges, K_star = cached
        print(f"    [forecasting/KARMA] loaded cached fit: K*={K_star}  n_edges={len(edges)}")
    else:
        edges, K_star, _b_star = run_karma(X_train, X_val, model, cfg, device, verbose=False, seed=seed, tau=None)
        save_explanation(ckpt_dir, karma_key, (edges, K_star), use_cache)
        print(f"    [forecasting/KARMA] K*={K_star}  n_edges={len(edges)}")
    attr = karma_edges_to_attr(edges, cfg.D, cfg.T)
    claimed = set()
    for e in edges:
        t_idx = cfg.T - e["lag"]
        if 0 <= t_idx < cfg.T:
            claimed.add((e["src"], t_idx))
    attribution = {(i, t, 0): float(abs(attr[i, t])) for i in range(cfg.D) for t in range(cfg.T)}

    f_np = _scalar_numpy_forward(model, device, tau=None)
    fold_results = []
    for fs, (start, end) in enumerate(_fold_slices(len(X_eval_pool), n_folds)):
        x_fold = X_eval_pool[start:end]
        fd = _fold_dir_tot(f_np, x_fold, cond_sampler, cfg.D, cfg.T, attribution, claimed, delta_B, omic_kwargs["rho_max"], seed + fs)
        fold_results.append(fd)
        print(f"    [forecasting/KARMA] fold {fs + 1}/{n_folds}: omic_support={fd['omic_support']:.3f} auomic={fd['auomic']:.3f} mediation_fraction={fd['mediation_fraction']:.3f}")

    return {
        "attrs": [attr] * n_folds, "claimed_sets": [claimed] * n_folds,
        "fold_agg": _aggregate_task_folds(fold_results), "fold_results": fold_results, "D": cfg.D, "T": cfg.T,
        "X_train": X_train, "model": model,
    }


def run_regression_task(seed: int, n_eval: int, n_folds: int, ckpt_dir: Path, device, ts_nsamples: int, omic_kwargs: dict, delta_B: int, karma_overrides: dict | None = None, run_tag: str = "full", use_cache: bool = True) -> dict:
    cfg, X_train, X_val, _X_test, y_train, y_val, X_eval_pool, cond_sampler = _load_beijing(seed, n_eval, n_folds, karma_overrides)
    model = get_or_train_gru(cfg, X_train, y_train, X_val, y_val, device, ckpt_dir)
    f_np = _scalar_numpy_forward(model, device, tau=None)
    D, T = cfg.D, cfg.T

    fold_results, attrs, claimed_sets = [], [], []
    for fs, (start, end) in enumerate(_fold_slices(len(X_eval_pool), n_folds)):
        x_fold = X_eval_pool[start:end]
        x_fold_dt = x_fold.transpose(0, 2, 1).astype(np.float32)  # (B, D, T) -- run_timeshap's own convention
        ts_key = f"timeshap_{run_tag}_seed{seed}_fold{fs}_ns{ts_nsamples}"
        attr = load_explanation(ckpt_dir, ts_key, use_cache)
        if attr is None:
            attr = run_timeshap(x_fold_dt, model, device, nsamples=ts_nsamples, tau=None)
            save_explanation(ckpt_dir, ts_key, attr, use_cache)
        attr_agg = np.abs(attr).mean(axis=0)  # (D, T), population-level over this fold
        ranked = sorted(((i, t) for i in range(D) for t in range(T)), key=lambda c: attr_agg[c], reverse=True)
        n_plus = max(1, int(round(len(ranked) * omic_kwargs.get("top_frac", 0.25))))
        claimed = set(ranked[:n_plus])
        attribution = {(i, t, 0): float(attr_agg[i, t]) for i in range(D) for t in range(T)}

        fd = _fold_dir_tot(f_np, x_fold, cond_sampler, D, T, attribution, claimed, delta_B, omic_kwargs["rho_max"], seed + fs)
        fold_results.append(fd)
        attrs.append(attr_agg)
        claimed_sets.append(claimed)
        print(f"    [regression/TimeSHAP] fold {fs + 1}/{n_folds}: mean|attr|={attr_agg.mean():.4f} omic_support={fd['omic_support']:.3f} auomic={fd['auomic']:.3f} mediation_fraction={fd['mediation_fraction']:.3f}")

    return {
        "attrs": attrs, "claimed_sets": claimed_sets,
        "fold_agg": _aggregate_task_folds(fold_results), "fold_results": fold_results, "D": D, "T": T,
        "X_train": X_train, "model": model,
    }


def run_classification_task(seed: int, n_eval: int, n_folds: int, ckpt_dir: Path, device, epochs: int, timing_nsamples: int, omic_kwargs: dict, delta_B: int, karma_overrides: dict | None = None, run_tag: str = "full", use_cache: bool = True) -> dict:
    cfg, X_train, X_val, _X_test, y_train, y_val, X_eval_pool, cond_sampler = _load_beijing(seed, n_eval, n_folds, karma_overrides)

    # Binary target: does next-step PM2.5 exceed the TRAIN set's own 75th percentile? (fit the
    # threshold on train only, apply to val/test -- avoids leaking test-set distribution info
    # into the label definition itself.)
    threshold = float(np.percentile(y_train[:, 0, PM25_INDEX], 75))
    y_train_bin = (y_train[:, 0, PM25_INDEX] > threshold).astype(np.float32)
    y_val_bin = (y_val[:, 0, PM25_INDEX] > threshold).astype(np.float32)
    print(f"    [classification/TIMING] PM2.5 exceedance threshold={threshold:.4f} (train p75), "
          f"positive rate: train={y_train_bin.mean():.3f} val={y_val_bin.mean():.3f}")

    ckpt_path = ckpt_dir / "beijing_classifier.pt"
    if ckpt_path.exists():
        classifier = BeijingClassifier(D=cfg.D, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)
        classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
        classifier.eval()
        val_acc = float("nan")
        print(f"    [classification/TIMING] loaded classifier from {ckpt_path}")
    else:
        classifier, val_acc = train_classifier(X_train, y_train_bin, X_val, y_val_bin, cfg, device, epochs=epochs)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), ckpt_path)
        print(f"    [classification/TIMING] trained classifier, val_acc={val_acc:.4f} -> {ckpt_path}")

    f_np = _scalar_numpy_forward(classifier, device, tau=None)
    D, T = cfg.D, cfg.T

    fold_results, attrs, claimed_sets = [], [], []
    for fs, (start, end) in enumerate(_fold_slices(len(X_eval_pool), n_folds)):
        x_fold = X_eval_pool[start:end]
        x_fold_dt = x_fold.transpose(0, 2, 1).astype(np.float32)
        timing_key = f"timing_clf_{run_tag}_seed{seed}_fold{fs}_ns{timing_nsamples}"
        attr = load_explanation(ckpt_dir, timing_key, use_cache)
        if attr is None:
            attr = run_timing_classification(x_fold_dt, classifier, device, n_samples=timing_nsamples)
            save_explanation(ckpt_dir, timing_key, attr, use_cache)
        ranked = sorted(((i, t) for i in range(D) for t in range(T)), key=lambda c: abs(attr[c]), reverse=True)
        n_plus = max(1, int(round(len(ranked) * omic_kwargs.get("top_frac", 0.25))))
        claimed = set(ranked[:n_plus])
        attribution = {(i, t, 0): float(abs(attr[i, t])) for i in range(D) for t in range(T)}

        fd = _fold_dir_tot(f_np, x_fold, cond_sampler, D, T, attribution, claimed, delta_B, omic_kwargs["rho_max"], seed + fs)
        fold_results.append(fd)
        attrs.append(attr)
        claimed_sets.append(claimed)
        print(f"    [classification/TIMING] fold {fs + 1}/{n_folds}: omic_support={fd['omic_support']:.3f} auomic={fd['auomic']:.3f} mediation_fraction={fd['mediation_fraction']:.3f}")

    return {
        "attrs": attrs, "claimed_sets": claimed_sets, "val_acc": val_acc,
        "fold_agg": _aggregate_task_folds(fold_results), "fold_results": fold_results, "D": D, "T": T,
        "X_train": X_train, "model": classifier,
    }


def plot_omic_curve(fold_agg: dict, title: str, out_path: str) -> None:
    """ (Eq. 9-10), each mean +/-1 std across CV folds, on the SAME axes (run_var_
    mediation.py's own dir-vs-tot overlay convention/colors) -- see run_var_omic.py's
    plot_omic_curves for the full pointwise-vs-cumulative distinction (plot_cumulative_auomic
    below is the running INTEGRAL of this panel, not a repeat of it). Titled "Known-Method
    Evaluation" to contrast explicitly with run_var_omic.py's synthetic "OMIC Metric
    Validation" figures.
."""
    rho = fold_agg["rho"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label, color in (("omic_curve", "OMIC_dir", "#e76f51"), ("omic_curve_tot", "OMIC_tot", "#2a9d8f")):
        mean, std = fold_agg[f"{key}_mean"], fold_agg[f"{key}_std"]
        ax.plot(rho, mean, label=label, color=color, linewidth=2)
        ax.fill_between(rho, mean - std, mean + std, color=color, alpha=0.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel(r"$\rho$ (fraction of claimed edges retained, POINTWISE at level $k$)")
    ax.set_ylabel(r"$OMIC_k$")
    ax.set_title(f"(Beijing, real data)\n{title}\nPointwise ranking curve (mean +/-1 std, CV folds)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cumulative_auomic(fold_agg: dict, title: str, out_path: str) -> None:
    """Cumulative AUOMIC_dir-up-to-k AND AUOMIC_tot-up-to-k,
    each mean +/-1 std across CV folds, on the SAME axes -- the running integral of
    plot_omic_curve's pointwise panel over rho, DISTINCT from it."""
    rho = fold_agg["rho"]
    cum_rho = rho[1:]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label, color in (("omic_curve_folds", "AUOMIC_dir", "#e76f51"), ("omic_curve_tot_folds", "AUOMIC_tot", "#2a9d8f")):
        cum_curves = np.array([
            [auomic(rho[:m], curve[:m]) for m in range(2, len(rho) + 1)]
            for curve in fold_agg[key]
        ])
        cum_mean, cum_std = np.nanmean(cum_curves, axis=0), np.nanstd(cum_curves, axis=0)
        ax.plot(cum_rho, cum_mean, label=label, color=color, linewidth=2)
        ax.fill_between(cum_rho, cum_mean - cum_std, cum_mean + cum_std, color=color, alpha=0.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel(r"$\rho$ (CUMULATIVE fraction retained, integral up to $\rho$)")
    ax.set_ylabel("Cumulative AUOMIC")
    ax.set_title(f"(Beijing, real data)\n{title}\nCumulative AUOMIC curve (mean +/-1 std, CV folds)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_edges(task_result: dict, title: str, out_path: str) -> None:
    """Fold-averaged (D, T) |attribution| grid, cells claimed in ANY fold outlined RED -- the
    SAME "red outline = claimed" convention run_var_omic.py's plot_edge_heatmaps uses, one per task. 
    With n_folds>1, this is a fold-mean heatmap, not a single point estimate."""
    attr_mean = np.abs(np.mean(task_result["attrs"], axis=0))
    claimed_union = set().union(*task_result["claimed_sets"])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attr_mean, cmap="viridis", aspect="auto")
    for (i, t) in claimed_union:
        ax.add_patch(plt.Rectangle((t - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5))
    ax.set_xlabel("source window position t")
    ax.set_ylabel("source variable i")
    n_folds = len(task_result["attrs"])
    ax.set_title(f"{title}\n(red = claimed in >=1 of {n_folds} CV fold(s), fold-mean |attribution|)")
    fig.colorbar(im, ax=ax, label="mean |attribution| across folds")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_med_heatmap(task_result: dict, title: str, out_path: str) -> None:
    D, T = task_result["D"], task_result["T"]
    med_mean = task_result["fold_agg"]["med_mean"]
    claimed_union = set().union(*task_result["claimed_sets"])

    grid = np.full((D, T), np.nan)
    for (i, t, _j), v in med_mean.items():
        if (i, t) in claimed_union:
            grid[i, t] = v
    masked = np.ma.masked_invalid(grid)

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(float(np.abs(masked).max()), 1e-8) if masked.count() else 1e-8
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax)
    for (i, t) in claimed_union:
        ax.add_patch(plt.Rectangle((t - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.5))
    for i in range(D):  # E5: hatch the structurally-empty-M(s) boundary column
        ax.add_patch(plt.Rectangle((T - 1 - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#868e96", linewidth=0))
    ax.set_xlabel("source window position t  (hatched column: M(s) empty by construction)")
    ax.set_ylabel("source variable i")
    n_folds = len(task_result["attrs"])
    ax.set_title(f"{title}\nMed(e) = Delta_tot - Delta_dir, mean across {n_folds} CV fold(s), claimed cells only")
    fig.colorbar(im, ax=ax, label="Med(e)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_dirtot_scatter(task_result: dict, title: str, out_path: str) -> None:
    """omic_new.md F1: every candidate edge at (Delta_dir(e), Delta_tot(e)), coloured by
    claimed/denied, diagonal drawn -- the SAME convention as run_var_omic.py/
    run_var_mediation.py's own scatter, for the real-data task. Uses the FIRST fold's raw
    deltas (a per-edge snapshot, not an n_folds ensemble -- matching run_var_omic.py's own "F1"
    convention for its reference-draw scatter)."""
    fd = task_result["fold_results"][0]
    candidate_edges, delta_dir_all, delta_tot_all = fd["candidate_edges"], fd["delta_dir_all"], fd["delta_tot_all"]
    claimed_edges = set(fd["claimed_edges"])

    x = np.array([delta_dir_all[e] for e in candidate_edges])
    y = np.array([delta_tot_all[e] for e in candidate_edges])
    claimed_mask = np.array([e in claimed_edges for e in candidate_edges])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x[~claimed_mask], y[~claimed_mask], color="#adb5bd", s=14, alpha=0.5, label="denied")
    ax.scatter(x[claimed_mask], y[claimed_mask], color="#e76f51", s=26, alpha=0.85, label="claimed")
    lim = max(float(x.max()), float(y.max()), 1e-8) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1, label=r"$\Delta_{tot}=\Delta_{dir}$")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\Delta_{dir}(e)$")
    ax.set_ylabel(r"$\Delta_{tot}(e)$")
    ax.set_title(f"(Beijing, real data)\n{title}\nMed(e) = vertical distance from the diagonal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mediation_curve(task_result: dict, title: str, out_path: str) -> None:
    """omic_new.md F2: mediation fraction as a CURVE over the scale-free threshold m
    (Med(e)/Delta_tot(e) >= m), mean +/-1 std over CV folds -- the SAME convention as
    run_var_mediation.py's own curve plot, for the real-data task."""
    curves = []
    m_grid = None
    for fd in task_result["fold_results"]:
        conf = mediation(fd["delta_dir_all"], fd["delta_tot_all"], claimed_edges=fd["claimed_edges"])
        m_grid = conf["m_grid"]
        curves.append(conf["fraction_curve"])
    curves = np.array(curves)
    mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(m_grid, mean, color="#6a4c93", linewidth=2.5)
    ax.fill_between(m_grid, mean - std, mean + std, color="#6a4c93", alpha=0.2)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="headline m=0.5")
    ax.set_xlabel(r"$m$ (Med(e)/$\Delta_{tot}$(e) $\geq m$ threshold)")
    ax.set_ylabel("mediation fraction")
    ax.set_title(f"(Beijing, real data)\n{title}\nmediation fraction vs. threshold m (mean +/-1 std, CV folds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)





def run_beijing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="small sizes, for a fast smoke run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_cache", action="store_true", help="disable explanation cache (always recompute KARMA/TimeSHAP/TIMING fresh)")
    args = parser.parse_args()

    cfg_yaml = load_config(args.config)
    out_dir = str(dagfaith_results_dir(cfg_yaml))
    bcfg = cfg_yaml["beijing_multitask"]
    base = bcfg["quick"] if args.quick else bcfg

    device = torch.device(args.device)
    set_global_seed(bcfg["seed"])
    ckpt_dir = Path(base.get("ckpt_dir", bcfg["ckpt_dir"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    omic_kwargs = dict(B=base["omic"]["B"], rho_max=base["omic"]["rho_max"], top_frac=base["omic"]["top_frac"])
    delta_B = base["omic"]["B"]
    karma_overrides = base.get("karma_overrides")
    n_folds = base.get("n_folds", 1)
    run_tag = "quick" if args.quick else "full"  # explanation-cache key prefix -- keeps quick/full runs from colliding
    use_cache = not args.no_cache

    print(f"=== Block B2: Beijing multi-task (forecasting/KARMA, regression/TimeSHAP, classification/TIMING), {n_folds}-fold CV ===")
    print(f"    explanation cache: {'ON' if use_cache else 'OFF (--no_cache)'} -- {_explanation_cache_dir(ckpt_dir)}")

    print("\n-- forecasting / KARMA --")
    fc = run_forecasting_task(bcfg["seed"], base["n_eval"], n_folds, ckpt_dir, device, omic_kwargs, delta_B, karma_overrides, run_tag, use_cache)

    print("\n-- regression / TimeSHAP --")
    reg = run_regression_task(bcfg["seed"], base["n_eval"], n_folds, ckpt_dir, device, base["ts_nsamples"], omic_kwargs, delta_B, karma_overrides, run_tag, use_cache)

    print("\n-- classification / TIMING --")
    clf = run_classification_task(bcfg["seed"], base["n_eval"], n_folds, ckpt_dir, device, base["clf_epochs"], base["timing_nsamples"], omic_kwargs, delta_B, karma_overrides, run_tag, use_cache)

    tasks = {"forecasting (KARMA)": fc, "regression (TimeSHAP)": reg, "classification (TIMING)": clf}

    print("\n=== omic_new.md E1: baselines on the same edge set, per task ===")
    baseline_rows = []
    for name, t in tasks.items():
        D, T = t["D"], t["T"]
        per_baseline = {b: {"omic_support_dir": [], "auomic_dir": [], "auomic_tot": []} for b in
                         ("RANDOM", "GRAD_X_INPUT", "IG_ZERO", "COND_OCCLUSION")}
        for fs, fd in enumerate(t["fold_results"]):
            x_fold_dt = fd["x_fold"].transpose(0, 2, 1).astype(np.float32)
            e1 = run_e1_baselines(
                t["model"], x_fold_dt, device, D, T, fd["candidate_edges"],
                fd["delta_dir_all"], fd["delta_tot_all"], fd["n_plus"],
                omic_kwargs["rho_max"], bcfg["seed"] + fs,
            )
            for b, res in e1.items():
                per_baseline[b]["omic_support_dir"].append(res["omic_support_dir"])
                per_baseline[b]["auomic_dir"].append(res["auomic_dir"])
                per_baseline[b]["auomic_tot"].append(res["auomic_tot"])
        for b, vals in per_baseline.items():
            baseline_rows.append({
                "task": name, "baseline": b,
                "omic_support_dir": float(np.nanmean(vals["omic_support_dir"])),
                "auomic_dir": float(np.nanmean(vals["auomic_dir"])),
                "auomic_tot": float(np.nanmean(vals["auomic_tot"])),
            })
    baseline_df = pd.DataFrame(baseline_rows)
    print(baseline_df.to_string(index=False))
    baseline_csv_path = os.path.join(out_dir, "beijing_baselines.csv")
    baseline_df.to_csv(baseline_csv_path, index=False)
    baseline_tex_path = os.path.join(out_dir, "tab_beijing_baselines.tex")
    with open(baseline_tex_path, "w") as fh:
        fh.write(
            "% Table: Beijing baselines on the same edge set (omic_new.md E1), grouped under\n"
            "% each task. COND_OCCLUSION (attribution = |Delta_dir| itself) UPPER-BOUNDS what\n"
            "% any explainer can score on AUOMIC_dir -- without this row the real method's own\n"
            "% row can't be interpreted. RANDOM fixes the empirical chance level at the actual\n"
            "% |E+|/|E| ratio (omic_iclr.md A3).\n"
        )
        fh.write(baseline_df.to_latex(index=False, float_format="%.3f", escape=False))


    slugs = {"forecasting (KARMA)": "fc", "regression (TimeSHAP)": "reg", "classification (TIMING)": "clf"}
    fig_paths = []
    for name, t in tasks.items():
        slug = slugs[name]
        p_omic = os.path.join(out_dir, f"fig_beijing_{slug}_omic.pdf")
        p_auomic = os.path.join(out_dir, f"fig_beijing_{slug}_auomic.pdf")
        p_edges = os.path.join(out_dir, f"fig_beijing_{slug}_edges.pdf")
        plot_omic_curve(t["fold_agg"], f"Beijing {name}: OMIC ranking curve", p_omic)
        plot_cumulative_auomic(t["fold_agg"], f"Beijing {name}: cumulative AUOMIC", p_auomic)
        plot_edges(t, f"Beijing {name}: claimed edges", p_edges)
        fig_paths += [p_omic, p_auomic, p_edges]

        p_med = os.path.join(out_dir, f"fig_beijing_{slug}_med.pdf")
        plot_med_heatmap(t, f"Beijing {name}: mediation", p_med)
        fig_paths.append(p_med)

        p_scatter = os.path.join(out_dir, f"fig_beijing_{slug}_dirtot_scatter.pdf")
        plot_dirtot_scatter(t, f"Beijing {name}: mediation", p_scatter)  # omic_new.md F1
        fig_paths.append(p_scatter)

        p_mcurve = os.path.join(out_dir, f"fig_beijing_{slug}_mcurve.pdf")
        plot_mediation_curve(t, f"Beijing {name}: mediation", p_mcurve)  # omic_new.md F2
        fig_paths.append(p_mcurve)

    print(f"\nSaved:{baseline_csv_path}, {baseline_tex_path},\n  " + ",\n  ".join(fig_paths))
