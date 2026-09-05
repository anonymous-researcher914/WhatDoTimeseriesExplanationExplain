"""Edge-graph metrics: thresholding and SHD/edge-F1 (experiments.md's metrics.py).

Generic, method-agnostic infrastructure -- operates on (D, T, D_out) score/boolean graph
tensors (dbn.py's own (i, t, j) convention), independent of which explainer produced the
scores. TEIG (teig.teig) is the only score-producer in this codebase; this module scores
whatever graph it thresholds to against a reference (dbn.py's Gstar / gf_analytic).
"""
from __future__ import annotations

import numpy as np


def threshold_scores(scores: np.ndarray, mode: str = "f1_optimal", G_ref: np.ndarray | None = None, thr: float | None = None) -> np.ndarray:
    """Binarize a (D, T, D_out) score tensor into an edge mask.

    mode="fixed": threshold at `thr` (scores > thr).
    mode="f1_optimal": pick the threshold maximizing edge-F1 against `G_ref`, searched over
        every distinct score value (each candidate threshold is tried; ties broken by the
        first-found max, i.e. the largest threshold achieving the max F1).
    """
    scores = np.asarray(scores, dtype=float)

    if mode == "fixed":
        if thr is None:
            raise ValueError("mode='fixed' requires `thr`")
        return scores > thr

    if mode != "f1_optimal":
        raise ValueError(f"unsupported mode={mode!r} (use 'fixed' or 'f1_optimal')")
    if G_ref is None:
        raise ValueError("mode='f1_optimal' requires `G_ref`")

    G_ref = np.asarray(G_ref, dtype=bool)
    candidates = np.unique(scores)
    if candidates.size == 0:
        return np.zeros_like(scores, dtype=bool)

    best_f1 = -1.0
    best_mask = scores > candidates[-1]  # all-False fallback
    for c in candidates:
        mask = scores > (c - 1e-12)  # includes score == c as "kept"
        f1 = edge_f1(mask.astype(int), G_ref.astype(int))
        if f1 >= best_f1:
            best_f1 = f1
            best_mask = mask
    return best_mask


def shd(G_pred: np.ndarray, G_true: np.ndarray) -> int:
    """Structural Hamming Distance: count of edge add/remove to match `G_true`. Edge direction
    (input position -> forecast target) is fixed by construction here, so SHD reduces to the
    symmetric-difference count on the edge set."""
    G_pred = np.asarray(G_pred, dtype=bool)
    G_true = np.asarray(G_true, dtype=bool)
    return int(np.sum(G_pred != G_true))


def edge_f1(G_pred: np.ndarray, G_true: np.ndarray) -> float:
    """F1 of the predicted edge set against the true edge set. 1.0 if both are empty (nothing
    to find, nothing missed); 0.0 if exactly one of precision/recall is undefined (no true
    positives to divide by)."""
    G_pred = np.asarray(G_pred, dtype=bool)
    G_true = np.asarray(G_true, dtype=bool)
    tp = int(np.sum(G_pred & G_true))
    fp = int(np.sum(G_pred & ~G_true))
    fn = int(np.sum(~G_pred & G_true))
    if tp == 0:
        return 1.0 if (fp == 0 and fn == 0) else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
