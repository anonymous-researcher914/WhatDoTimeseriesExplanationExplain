"""VAR synthetic: the CONTROLLED metric-behaviour experiment.

A linear vector-autoregression (dagfaith.dbn.sample_dbn with D_out=D: predicting all D
variables from a T-lag window of the same D variables IS a VAR(T) forecaster by construction)
with known coefficient support = true edges, scored by the EXACT analytic conditional
(dagfaith.cond_baseline.analytic_gaussian_cond_model_for_ar_inputs) -- no estimation error in
either the forecaster or the on-manifold intervention, isolating the METRIC's own behaviour
(the point of this experiment, per the doc's honesty guards) from any model/conditional
estimation error.

Reuses the codebase's existing VAR-generator/analytic-oracle/analytic-conditional machinery
(dagfaith.dbn, dagfaith.oracle.RawWindowOracle, dagfaith.cond_baseline) rather than a separate
generator -- omic_iclr.md's own "aligns with existing data.py" framing.

Three attributions over the SAME shared Delta (computed once, since Delta is a property of the
forecaster/data, not of the attribution method):
  - GOOD:   claimed edges = the true support Gstar, ranked by |true coefficient|
  - RANDOM: a random subset of candidate edges (same size as GOOD's claim), random ranking
  - BAD:    the SAME true-support claim as GOOD, but with a genuinely INVERTED ranking
            (reciprocal, not a sign flip -- omic_ranking_curve ranks by |attribution|, and
            abs(-x) == abs(x), so negating alone does not invert anything; see tests/test_metrics.py)

Repeats over `n_seeds` -- each a FRESH VAR draw). See run_var_omic's own docstring for why GOOD/BAD/ANTI's
bands still come out visually flat while RANDOM's is wide -- a real property of the
construction, not a leftover plotting bug.

Usage:
    python experiments/run_var_omic.py [--config configs/default.yaml] [--quick]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dagfaith import dbn
from dagfaith.cond_baseline import analytic_gaussian_cond_model_for_ar_inputs
from dagfaith.config import load_config, results_dir, seed_everything
from dagfaith.intervention import delta_dict
from dagfaith.intervention_tot import delta_tot_dict
from dagfaith.oracle import RawWindowOracle
from dagfaith.omic import auomic, kendall_tau_restricted, omic_ranking_curve, omic_ranking_curve_restricted, omic_support

METHOD_COLORS = {"GOOD": "#2a9d8f", "RANDOM": "#e9c46a", "BAD": "#e76f51", "ANTI": "#6a4c93"}


def _all_edges(D: int, T: int, D_out: int) -> list[tuple[int, int, int]]:
    return [(i, t, j) for i in range(D) for t in range(T) for j in range(D_out)]


def build_attributions(
    candidate_edges: list, Gstar: np.ndarray, B_coef: np.ndarray, rng: np.random.Generator
) -> dict[str, tuple[list, dict]]:
    """GOOD/RANDOM/BAD/ANTI (claimed_edges, attribution) pairs -- see module docstring.

    BAD and ANTI are deliberately DIFFERENT kinds of "bad", testing different failure modes:
      - BAD  = the CORRECT set (true support), genuinely rank-inverted within it. Its
               OMIC_support stays ~1.0 (still claims the RIGHT edges -- a set-level property)
               while its AUOMIC drops, but NOT all the way to chance: this is the intended
               OMIC_support-vs-AUOMIC distinction (A3), not an under-powered "bad".
      - ANTI = the WRONG set entirely: the |S| candidate edges with the LOWEST |true
               coefficient| (i.e., confidently claiming the LEAST important edges as important),
               ranked normally by their own (still-low) magnitude. This is a genuinely different
               failure mode -- getting the CLAIM itself wrong, not just its internal order --
               and should score OMIC_support well below chance, unlike BAD.

    BAD's AUOMIC FLOOR DEPENDS ON GRAPH DENSITY, not just correctness -- worth stating plainly
    since it surprised an earlier reader of this exact figure. At level k, BAD's worst-first
    picks are compared against E_k^- = (remaining, higher-Delta support members) UNION
    (non-support edges). At LOW sparsity, non-support edges dominate E_k^- everywhere, so BAD's
    picks mostly still "win" (their comparisons against the tiny remaining-support minority
    barely move the average) -- AUOMIC lands ~0.8-0.99, correct per A3's own validated number
    but not a stringent demonstration. At HIGHER sparsity, remaining-support members are a much
    bigger share of E_k^-, so BAD's early picks genuinely lose more often -- configs/
    default.yaml's var_omic.sparsity is set to 0.5 for exactly this reason (see its own
    comment): GOOD~1.0 > BAD~0.63 > RANDOM~0.47 > ANTI~0.19, a much cleaner separation than
    sparsity=0.2's GOOD~1.0 > BAD~0.87-0.99 > RANDOM~0.44 > ANTI~0.39. Past ~0.65 sparsity the
    ordering collapses (BAD crosses below RANDOM) -- this is a real, reproducible boundary of
    the metric's behavior on a VAR graph this size, not a tuning accident.
    """
    ranked_by_coef = sorted(candidate_edges, key=lambda e: abs(B_coef[e]))  # ascending
    good_claimed = [e for e in candidate_edges if Gstar[e]]
    good_attr = {e: float(abs(B_coef[e])) for e in good_claimed}

    random_claimed = [
        candidate_edges[k]
        for k in rng.choice(len(candidate_edges), size=len(good_claimed), replace=False)
    ]
    random_attr = {e: float(rng.random()) for e in random_claimed}

    # Genuinely inverted (reciprocal, not a sign flip -- see module docstring).
    bad_attr = {e: 1.0 / (abs(B_coef[e]) + 1e-8) for e in good_claimed}

    anti_claimed = ranked_by_coef[: len(good_claimed)]  # the |S| LEAST important edges
    anti_attr = {e: float(abs(B_coef[e])) for e in anti_claimed}

    return {
        "GOOD": (good_claimed, good_attr),
        "RANDOM": (random_claimed, random_attr),
        "BAD": (good_claimed, bad_attr),
        "ANTI": (anti_claimed, anti_attr),
    }


def run_var_omic(
    D: int, T: int, n: int, sparsity: float, noise_std: float, seed: int,
    delta_B: int, n_seeds: int, rho_max: float,
) -> dict:
    """Returns {method: {"rho", "omic_mean", "omic_std", "omic_support_mean",
    "omic_support_std", "auomic_mean", "auomic_std", "cum_rho", "cum_auomic_mean",
    "cum_auomic_std", "n_plus"}}, plus "Gstar"/"B_coef"/"claimed" for the edge plot.

    Each of the n_seeds repeats draws a FRESH VAR sample (X, B_coef, Gstar) and a fresh RANDOM
    claim, not just fresh Delta draw-noise on one fixed graph -- with the data held fixed, the
    ranking curve is close to deterministic (GOOD/BAD/ANTI's claimed sets never change, and
    delta_B=32 already averages out most on-manifold draw noise), so a band built from draw-
    noise alone was genuinely, correctly near-invisible, not a plotting bug. Redrawing the whole
    controlled experiment each seed answers the more useful question -- "how much would this
    metric's read on GOOD/RANDOM/BAD/ANTI vary if we reran this on a different random VAR of the
    same size/sparsity".

    Even with this fix, GOOD/BAD/ANTI's bands stay visually FLAT while RANDOM's is wide -- this
    is real, not a residual bug: GOOD/BAD/ANTI are SYSTEMATIC constructions (the top/bottom of a
    ranking by true coefficient magnitude), and a pairwise-concordance statistic over dozens of
    edges built that way concentrates tightly regardless of which specific random VAR draw
    produced the coefficients -- omic_support for GOOD/BAD/ANTI is EXACTLY 1.0/1.0/0.0 every
    single seed (a mathematical certainty of the construction: BAD claims the identical true-
    support set GOOD does; ANTI claims the bottom-|S| set), so only their AUOMIC (rank-order-
    sensitive) shows even the tiny remaining Delta-draw-noise band. RANDOM's claim, by contrast,
    genuinely differs seed to seed (a fresh arbitrary subset, whose overlap with the sparse true
    support varies a lot draw to draw) -- so its band is wide, correctly. Read a flat GOOD/BAD/
    ANTI line here as "OMIC's read on a well-defined systematic claim is highly reproducible",
    not as a plotting defect.

    The candidate-pool SIZE (n_plus/K) stays identical every seed (sparsity's edge count is
    deterministic given D/T/D_out), so the shared rho grid is still valid to average across
    seeds. The illustrative edge-heatmap plot uses the FIRST seed's draw only (Gstar/B_coef
    below), not an average across seeds -- there is no meaningful "average graph" to show.
    """
    per_method_curves = {m: [] for m in ("GOOD", "RANDOM", "BAD", "ANTI")}
    per_method_support = {m: [] for m in ("GOOD", "RANDOM", "BAD", "ANTI")}
    per_method_auomic = {m: [] for m in ("GOOD", "RANDOM", "BAD", "ANTI")}
    per_method_auomic_restricted = {m: [] for m in ("GOOD", "RANDOM", "BAD", "ANTI")} 
    per_method_tau_restricted = {m: [] for m in ("GOOD", "RANDOM", "BAD", "ANTI")}      
    per_method_n_plus: dict[str, int] = {}
    rho_grid = {}
    ref_Gstar = ref_B_coef = None
    ref = {} 

    for s in range(n_seeds):
        data_seed = seed * 1000 + s
        seed_everything(data_seed)
        X, _Y, B_coef, Gstar, _ = dbn.sample_dbn(
            D=D, T=T, n=n, D_out=D, sparsity=sparsity, nonlinear=False, noise_std=noise_std, seed=data_seed
        )
        oracle = RawWindowOracle(B_coef)
        cond_model = analytic_gaussian_cond_model_for_ar_inputs(D, T, ar=0.5, noise_std=noise_std)
        candidate_edges = _all_edges(D, T, D)
        if ref_Gstar is None:
            ref_Gstar, ref_B_coef = Gstar, B_coef

        build_rng = np.random.default_rng(data_seed + 777)  # RANDOM's own claim, fresh per seed
        attributions = build_attributions(candidate_edges, Gstar, B_coef, build_rng)

        cond_rng = np.random.default_rng(data_seed + 999)
        delta_rng = np.random.default_rng(data_seed + 1999)
        cond_sampler = cond_model.as_cond_sampler(cond_rng)
        delta = delta_dict(oracle.numpy_forward, X, candidate_edges, cond_sampler, B=delta_B, rng=delta_rng)

        if not ref:  
            ref = {
                "X": X, "oracle": oracle, "cond_model": cond_model, "candidate_edges": candidate_edges,
                "delta_dir": delta, "random_claimed": attributions["RANDOM"][0],
            }

        for method, (claimed, attribution) in attributions.items():
            claimed_set = set(claimed)  
            d_plus = np.array([delta[e] for e in claimed])
            all_others = [e for e in candidate_edges if e not in claimed_set]
            d_minus = np.array([delta[e] for e in all_others])
            support = omic_support(d_plus, d_minus)

            rho, omic_curve = omic_ranking_curve(claimed, attribution, delta, candidate_edges, rho_max=rho_max)
            rho_grid[method] = rho
            per_method_curves[method].append(omic_curve)
            per_method_support[method].append(support)
            per_method_auomic[method].append(auomic(rho, omic_curve))
            per_method_n_plus[method] = len(claimed) 
            
            rho_r, omic_r = omic_ranking_curve_restricted(claimed, attribution, delta, rho_max=rho_max)
            per_method_auomic_restricted[method].append(auomic(rho_r, omic_r))
            per_method_tau_restricted[method].append(kendall_tau_restricted(claimed, attribution, delta))


    ref_delta_tot = delta_tot_dict(
        ref["oracle"].numpy_forward, ref["X"], ref["candidate_edges"], ref["cond_model"], D, T,
        B=delta_B, rng=np.random.default_rng(seed + 424242),
    )
    dirtot = {
        "candidate_edges": ref["candidate_edges"],
        "delta_dir": ref["delta_dir"],
        "delta_tot": ref_delta_tot,
        "Gstar": ref_Gstar,
    }

    results = {
        "Gstar": ref_Gstar, "B_coef": ref_B_coef, "D": D, "T": T, "dirtot": dirtot,
        "random_claimed": ref["random_claimed"], 
    }
    for method in per_method_curves:
        curves = np.array(per_method_curves[method])  # (n_seeds, K)
        rho = rho_grid[method]
        cum_auomic = np.array([
            [auomic(rho[:m], curve[:m]) for m in range(2, len(rho) + 1)]
            for curve in curves
        ])  # (n_seeds, K-1)
        results[method] = {
            "rho": rho,
            "omic_mean": np.nanmean(curves, axis=0),
            "omic_std": np.nanstd(curves, axis=0),
            "omic_support_mean": float(np.nanmean(per_method_support[method])),
            "omic_support_std": float(np.nanstd(per_method_support[method])),
            "auomic_mean": float(np.nanmean(per_method_auomic[method])),
            "auomic_std": float(np.nanstd(per_method_auomic[method])),
            "cum_rho": rho[1:],
            "cum_auomic_mean": np.nanmean(cum_auomic, axis=0),
            "cum_auomic_std": np.nanstd(cum_auomic, axis=0),
            "n_plus": per_method_n_plus[method],
            "auomic_restricted_mean": float(np.nanmean(per_method_auomic_restricted[method])),
            "auomic_restricted_std": float(np.nanstd(per_method_auomic_restricted[method])),
            "tau_restricted_mean": float(np.nanmean(per_method_tau_restricted[method])),
            "tau_restricted_std": float(np.nanstd(per_method_tau_restricted[method])),
        }
    return results


def plot_omic_curves(results: dict, out_path: str) -> None:
    """OMIC Metric Validation (synthetic, KNOWN ground truth) -- pointwise
    OMIC_k at EACH ranking level k (Eq. 9-10), one line per method, shaded band = +/-1 std
    across the n_seeds repeats (varying only Delta's own on-manifold draw RNG; the claimed-edge
    sets themselves are fixed once per method -- see build_attributions). Distinct from
    plot_cumulative_auomic below: this panel answers "how faithful is the k-th most-confident
    claimed edge", not "what single number would the whole claim earn" -- they are genuinely
    different quantities (the second is a running INTEGRAL of the first over rho), not one
    plot repeated.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in ("GOOD", "RANDOM", "BAD", "ANTI"):
        r = results[method]
        color = METHOD_COLORS[method]
        ax.plot(r["rho"], r["omic_mean"], label=method, color=color, markersize=4, linewidth=2.5)


        ax.fill_between(
            r["rho"], r["omic_mean"] - r["omic_std"], r["omic_mean"] + r["omic_std"],
            color=color, alpha=0.2,
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel(r"$\rho$ (fraction of claimed edges retained, POINTWISE at level $k$)")
    ax.set_ylabel(r"$OMIC_k$")
    ax.set_title("OMIC Metric Validation (synthetic VAR, known ground truth)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cumulative_auomic(results: dict, out_path: str) -> None:
    """Cumulative AUOMIC-up-to-k curve, one line per method -- DISTINCT from
    the pointwise ranking curve above (plot_omic_curves), not a redundant re-plot of it: this is
    auomic(rho[:m], omic[:m]) for increasing m, i.e. the running INTEGRAL of the pointwise curve
    over rho -- it shows how the single AUOMIC summary number in tab_var.tex would settle as
    more of E+ is retained, which the pointwise panel alone cannot show (a pointwise curve that
    dips briefly in the middle can still integrate to a high AUOMIC, and vice versa)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in ("GOOD", "RANDOM", "BAD", "ANTI"):
        r = results[method]
        color = METHOD_COLORS[method]
        ax.plot(r["cum_rho"], r["cum_auomic_mean"], label=method, color=color, markersize=4, linewidth=2.5)
        ax.fill_between(
            r["cum_rho"], r["cum_auomic_mean"] - r["cum_auomic_std"],
            r["cum_auomic_mean"] + r["cum_auomic_std"], color=color, alpha=0.2,
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel(r"$\rho$ (CUMULATIVE fraction retained, integral up to $\rho$)")
    ax.set_ylabel("Cumulative AUOMIC")
    ax.set_title("OMIC Metric Validation (synthetic VAR, known ground truth)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_edge_heatmaps(results: dict, out_path: str) -> None:
    """KARMA-level-2 style top-claimed-edges plot: source->target heatmap (max |coef| over
    lag, matching karma_edges_to_attr's own aggregation rule) -- true VAR support vs RANDOM's
    claimed edges, RED-OUTLINED the same way experiments/run_beijing_multitask.py's own edge
    plots mark claimed cells (Block B3's cross-experiment style-consistency item: both figures
    use the SAME "red outline = claimed" visual language, even though B1's grid is (D, D)
    [source var -> target var, collapsed over lag] and B2's is (D, T) [source var -> source
    window position] -- the two experiments' underlying attribution shapes genuinely differ
    [B1 has a target-variable axis to collapse; B2's single-scalar-output tasks don't], so full
    pixel-for-pixel identical layout isn't meaningful, but the claim-marking convention is).

    omic_new.md H6: this used to collapse Gstar for BOTH panels (GOOD claims exactly Gstar by
    construction -- see build_attributions -- so the two panels were pixel-identical and the
    figure could never show a mismatch even in principle). Uses RANDOM's actual claim instead,
    which genuinely can (and typically does) differ from the true support -- a real recovery
    sanity check, not a tautology.
    """
    D, T = results["D"], results["T"]
    Gstar, B_coef = results["Gstar"], results["B_coef"]
    random_claimed = results["random_claimed"]  # (i, t, j) edges, may differ from Gstar

    def _collapse_mask(mask: np.ndarray) -> np.ndarray:
        grid = np.zeros((D, D))
        for i in range(D):
            for t in range(T):
                for j in range(D):
                    v = abs(B_coef[i, t, j]) if mask[i, t, j] else 0.0
                    grid[i, j] = max(grid[i, j], v)
        return grid

    random_mask = np.zeros((D, T, D), dtype=bool)
    for (i, t, j) in random_claimed:
        random_mask[i, t, j] = True

    true_grid = _collapse_mask(Gstar)
    claimed_grid = _collapse_mask(random_mask)
    true_cells = {(i, j) for i in range(D) for j in range(D) if Gstar[i, :, j].any()}
    claimed_cells = {(i, j) for (i, t, j) in random_claimed}

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    vmax = max(true_grid.max(), claimed_grid.max(), 1e-8)
    panels = ((axes[0], true_grid, true_cells, "True VAR support"), (axes[1], claimed_grid, claimed_cells, "RANDOM claimed edges"))
    for ax, grid, cells, title in panels:
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax)
        for (i, j) in cells:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.2))
        ax.set_xlabel("target variable j")
        ax.set_ylabel("source variable i")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="max |coefficient| over lag", shrink=0.8)
    fig.suptitle(
        "OMIC Metric Validation (synthetic VAR): source->target edge structure\n"
        "(RANDOM's claim, not GOOD's -- GOOD claims exactly the true support by construction, see H4)"
    )
    fig.savefig(out_path)
    plt.close(fig)


def plot_dirtot_scatter(results: dict, out_path: str) -> None:
    """Every candidate edge at (Delta_dir(e), Delta_tot(e)), coloured by
    claimed (true support, GOOD's own claim -- the canonical "claimed" set for this experiment)
    vs denied, diagonal drawn -- "REPLACES THREE PARAGRAPHS OF PROSE" per F1's own framing: a
    denied/irrelevant cell sits at the origin, a genuinely direct edge sits far out ON the
    diagonal (Delta_tot ~= Delta_dir, no mediation contribution), and a cell ABOVE the diagonal
    (Delta_tot > Delta_dir) is doing some of its work through mediation -- Med(e) is exactly the
    vertical distance from the diagonal at that point, made visually legible instead of implicit
    in a table column. A single reference draw (not averaged over n_seeds), computed once in
    run_var_omic alongside the ensemble loop.
    """
    dirtot = results["dirtot"]
    candidate_edges = dirtot["candidate_edges"]
    delta_dir = dirtot["delta_dir"]
    delta_tot = dirtot["delta_tot"]
    Gstar = dirtot["Gstar"]

    claimed_mask = np.array([bool(Gstar[e]) for e in candidate_edges])
    x = np.array([delta_dir[e] for e in candidate_edges])
    y = np.array([delta_tot[e] for e in candidate_edges])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x[~claimed_mask], y[~claimed_mask], color="#adb5bd", s=18, alpha=0.6, label="denied")
    ax.scatter(x[claimed_mask], y[claimed_mask], color=METHOD_COLORS["GOOD"], s=28, alpha=0.85, label="claimed (true support)")
    lim = max(x.max(), y.max(), 1e-8) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1, label=r"$\Delta_{tot}=\Delta_{dir}$")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\Delta_{dir}(e)$")
    ax.set_ylabel(r"$\Delta_{tot}(e)$")
    ax.set_title("Direct-vs-total effect per candidate edge (synthetic VAR)\nMed(e) = vertical distance from the diagonal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_var():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="small sizes, for a fast smoke run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = str(results_dir(cfg))
    vcfg = cfg["var_omic"]
    base = vcfg["quick"] if args.quick else vcfg

    print("=== Block B1: VAR synthetic (OMIC metric validation) ===")
    results = run_var_omic(
        D=base["D"], T=base["T"], n=base["n"], sparsity=base["sparsity"], noise_std=base["noise_std"],
        seed=vcfg["seed"], delta_B=base["delta_B"], n_seeds=base["n_seeds"], rho_max=base["rho_max"],
    )

    n_candidates = results["D"] * results["T"] * results["D"] 
    rows = []
    for method in ("GOOD", "RANDOM", "BAD", "ANTI"):
        r = results[method]
        rows.append({
            "method": method,
            "n_plus": r["n_plus"],
            "s=|S|/|E|": r["n_plus"] / n_candidates, 
            "omic_support": r["omic_support_mean"],
            "omic_support_std": r["omic_support_std"],
            "auomic": r["auomic_mean"],
            "auomic_std": r["auomic_std"],
            "auomic_restricted": r["auomic_restricted_mean"],
            "auomic_restricted_std": r["auomic_restricted_std"],
            "tau_restricted": r["tau_restricted_mean"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, "var_omic.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(out_dir, "tab_var.tex")
    with open(tex_path, "w") as fh:
        fh.write(
            "% Table: VAR synthetic OMIC_support/AUOMIC, GOOD/RANDOM/BAD/ANTI attribution\n"
            "% ( the controlled metric-validation experiment)\n"
            "% s=|S|/|E| is the graph density -- the full-complement AUOMIC column is a\n"
            "% CLOSED FORM of s for GOOD/BAD/ANTI (omic_new.md H0), not an empirical finding;\n"
            "% auomic_restricted/tau_restricted are density-INDEPENDENT ranking-only reads.\n"
        )
        fh.write(df.to_latex(index=False, float_format="%.3f", escape=False))

    fig_omic_path = os.path.join(out_dir, "fig_var_omic.pdf")
    fig_auomic_path = os.path.join(out_dir, "fig_var_auomic.pdf")
    fig_edges_path = os.path.join(out_dir, "fig_var_edges.pdf")
    fig_scatter_path = os.path.join(out_dir, "fig_var_dirtot_scatter.pdf")
    plot_omic_curves(results, fig_omic_path)
    plot_cumulative_auomic(results, fig_auomic_path)
    plot_edge_heatmaps(results, fig_edges_path)
    plot_dirtot_scatter(results, fig_scatter_path)  # omic_new.md Block F1

    print(f"\nSaved: {csv_path}, {tex_path}, {fig_omic_path}, {fig_auomic_path}, {fig_edges_path}, {fig_scatter_path}")

