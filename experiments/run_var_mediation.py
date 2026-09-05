""" VAR with a PLANTED mediator: the CONTROLLED direct-vs-total
mediation demonstration, known ground truth (dagfaith.var_mediated.sample_var_mediated).

X1@t=0 influences the forecast target Y ONLY by driving X2 one step later (X2 in turn feeds Y
directly) -- so Delta_dir(X1->Y) == 0 EXACTLY (Y's structural formula never reads X1) while
Delta_tot(X1->Y) > 0 (moving X1 moves X2, which Y reads). X3@(T-1) is a KNOWN direct edge.

A "marginal predictive relevance" claimant is used as the attribution method under test: it is exactly the
kind of method that conflates predictiveness with direct dependence, so it SHOULD claim the
mediated X1@0->Y edge with high attribution. We then show it scores WELL on OMIC_tot/AUOMIC_tot
(tot alone doesn't expose the mediation -- it isn't supposed to) and POORLY on OMIC_dir/AUOMIC_dir
(dir does) -- the gap (not either metric alone) is the mediation, on KNOWN ground truth.

Usage:
    python experiments/run_var_mediation.py [--config configs/default.yaml] [--quick]
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

from dagfaith.config import load_config, results_dir, seed_everything
from dagfaith.intervention import delta_dict
from dagfaith.intervention_tot import delta_tot_dict
from dagfaith.metrics_tot import mediation, evaluate_dir_tot
from dagfaith.var_mediated import sample_var_mediated


def _all_edges(D: int, T: int) -> list[tuple[int, int, int]]:
    return [(i, t, 0) for i in range(D) for t in range(T)]


def marginal_correlation_attribution(X: np.ndarray, Y: np.ndarray, candidate_edges) -> dict:
    """|corr(X[:, t, i], Y[:, 0])| over the sample -- a black-box "is this cell predictive of
    the target at all" score, blind to whether the path is direct or mediated. Stands in for
    omic_total.md B1's "marginal perturbation / vanilla IG" claimant: any method that ranks
    edges by raw predictiveness (not causal directness) will claim a mediated edge just as
    happily as a direct one, which is exactly the failure mode this experiment demonstrates."""
    y = Y[:, 0]
    attr = {}
    for e in candidate_edges:
        i, t, _j = e
        x = X[:, t, i]
        c = np.corrcoef(x, y)[0, 1]
        attr[e] = float(abs(c)) if np.isfinite(c) else 0.0
    return attr


def run_var_mediation(
    n: int, T: int, m: float, ar: float, beta: float, gamma: float,
    noise_std: float, target_noise_std: float, seed: int,
    delta_B: int, n_seeds: int, rho_max: float, top_k: int,
) -> dict:
    seed_everything(seed)
    gen = sample_var_mediated(
        n=n, T=T, m=m, ar=ar, beta=beta, gamma=gamma,
        noise_std=noise_std, target_noise_std=target_noise_std, seed=seed,
    )
    X, Y, f, cond_model, D = gen["X"], gen["Y"], gen["f"], gen["cond_model"], gen["D"]
    candidate_edges = _all_edges(D, T)

    attribution = marginal_correlation_attribution(X, Y, candidate_edges)
    claimed = sorted(candidate_edges, key=lambda e: attribution[e], reverse=True)[:top_k]

    per_seed_dir, per_seed_med, per_seed_conf, per_seed_curve = [], [], [], []
    rho_dir_grid = rho_tot_grid = m_grid = None

    for s in range(n_seeds):
        rng_dir = np.random.default_rng(seed * 1000 + s)
        rng_tot = np.random.default_rng(seed * 1000 + s + 500)
        delta_dir_all = delta_dict(f, X, candidate_edges, cond_model.as_cond_sampler(rng_dir), B=delta_B)
        delta_tot_all = delta_tot_dict(f, X, candidate_edges, cond_model, D, T, B=delta_B, rng=rng_tot)

        result = evaluate_dir_tot(claimed, attribution, delta_dir_all, delta_tot_all, candidate_edges, rho_max=rho_max)
        conf = mediation(delta_dir_all, delta_tot_all, claimed_edges=claimed)  # omic_new.md F2: scale-free Med/Delta_tot >= m

        rho_dir_grid, rho_tot_grid = result["rho_dir"], result["rho_tot"]
        m_grid = conf["m_grid"]
        per_seed_dir.append(result)
        per_seed_med.append(conf["med"])
        per_seed_conf.append(conf["mediation_fraction"])
        per_seed_curve.append(conf["fraction_curve"])

    omic_curve_dir = np.array([r["omic_curve_dir"] for r in per_seed_dir])
    omic_curve_tot = np.array([r["omic_curve_tot"] for r in per_seed_dir])
    med_mean = {e: float(np.mean([md[e] for md in per_seed_med])) for e in candidate_edges}
    fraction_curve_arr = np.array(per_seed_curve)

    return {
        "D": D, "T": T,
        "mediator_edge": gen["mediator_edge"], "direct_edge": gen["direct_edge"],
        "claimed": claimed, "attribution": attribution,
        "rho_dir": rho_dir_grid, "rho_tot": rho_tot_grid,
        "omic_curve_dir_mean": np.nanmean(omic_curve_dir, axis=0),
        "omic_curve_dir_std": np.nanstd(omic_curve_dir, axis=0),
        "omic_curve_tot_mean": np.nanmean(omic_curve_tot, axis=0),
        "omic_curve_tot_std": np.nanstd(omic_curve_tot, axis=0),
        "omic_support_dir_mean": float(np.nanmean([r["omic_support_dir"] for r in per_seed_dir])),
        "omic_support_tot_mean": float(np.nanmean([r["omic_support_tot"] for r in per_seed_dir])),
        "auomic_dir_mean": float(np.nanmean([r["auomic_dir"] for r in per_seed_dir])),
        "auomic_dir_std": float(np.nanstd([r["auomic_dir"] for r in per_seed_dir])),
        "auomic_tot_mean": float(np.nanmean([r["auomic_tot"] for r in per_seed_dir])),
        "auomic_tot_std": float(np.nanstd([r["auomic_tot"] for r in per_seed_dir])),
        "gap_mean": float(np.nanmean([r["gap"] for r in per_seed_dir])),
        "med": med_mean,
        "mediation_fraction_mean": float(np.nanmean(per_seed_conf)),
        
        "m_grid": m_grid,
        "fraction_curve_mean": np.nanmean(fraction_curve_arr, axis=0),
        "fraction_curve_std": np.nanstd(fraction_curve_arr, axis=0),
    }


def plot_dir_tot_curves(results: dict, out_path: str) -> None:
    """OMIC_dir vs OMIC_tot pointwise ranking curves for the SAME claimed set (the marginal-
    correlation claimant) -- the two curves diverge exactly where mediation is doing the work:
    OMIC_tot stays high (the claim IS predictive), OMIC_dir drops (it isn't direct)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(results["rho_dir"], results["omic_curve_dir_mean"], label="OMIC_dir", color="#e76f51", linewidth=2.5)
    ax.fill_between(
        results["rho_dir"],
        results["omic_curve_dir_mean"] - results["omic_curve_dir_std"],
        results["omic_curve_dir_mean"] + results["omic_curve_dir_std"],
        color="#e76f51", alpha=0.2,
    )
    ax.plot(results["rho_tot"], results["omic_curve_tot_mean"], label="OMIC_tot", color="#2a9d8f", linewidth=2.5)
    ax.fill_between(
        results["rho_tot"],
        results["omic_curve_tot_mean"] - results["omic_curve_tot_std"],
        results["omic_curve_tot_mean"] + results["omic_curve_tot_std"],
        color="#2a9d8f", alpha=0.2,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel(r"$\rho$ (fraction of claimed edges retained)")
    ax.set_ylabel(r"$OMIC_k$")
    ax.set_title(
        "Direct-vs-Total mediation"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mediation_curve(results: dict, out_path: str) -> None:
    """omic_new.md F2: mediation fraction as a CURVE over the scale-free threshold m (Med(e) /
    Delta_tot(e) >= m), mean +/-1 std over n_seeds -- the B3 headline value (at the default
    m=0.5, dashed vertical line) is one point read off this curve, not a number reported alone
    at an unstated threshold."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mean, std = results["fraction_curve_mean"], results["fraction_curve_std"]
    ax.plot(results["m_grid"], mean, color="#6a4c93", linewidth=2.5)
    ax.fill_between(results["m_grid"], mean - std, mean + std, color="#6a4c93", alpha=0.2)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="headline m=0.5")
    ax.set_xlabel(r"$m$ (Med(e)/$\Delta_{tot}$(e) $\geq m$ threshold)")
    ax.set_ylabel("mediation fraction")
    ax.set_title("mediation fraction vs. threshold m\n(the headline number is one point on this curve)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_med_heatmap(results: dict, out_path: str) -> None:
    """Med(e) = Delta_tot(e) - Delta_dir(e), restricted to CLAIMED cells only (every other cell
    is masked blank, not just left unoutlined) -- KARMA-level-2 style, black-outlined claimed
    cells (same convention as run_var_omic.py/run_beijing_multitask.py). Med(e) is still
    computed over the full candidate grid internally (needed for the SCALE-FREE
    Med(e)/Delta_tot(e) ratio -- omic_new.md F2, dagfaith.metrics_tot.mediation) -- only the
    DISPLAY here is restricted, so the color scale isn't stretched by an unclaimed cell's
    Med(e). A bright cell that is NOT a known direct edge is exactly the mediation signature:
    claimed/predictive but mediated."""
    D, T = results["D"], results["T"]
    med = results["med"]
    claimed_cells = {(i, t) for (i, t, _j) in results["claimed"]}

    grid = np.full((D, T), np.nan)
    for (i, t, _j), v in med.items():
        if (i, t) in claimed_cells:
            grid[i, t] = v
    masked = np.ma.masked_invalid(grid)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vmax = max(float(np.abs(masked).max()), 1e-8) if masked.count() else 1e-8
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax)
    for (i, t) in claimed_cells:
        ax.add_patch(plt.Rectangle((t - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.5))
    mi, mt, _ = results["mediator_edge"]
    di, dt, _ = results["direct_edge"]
    ax.text(mt, mi, "M", ha="center", va="center", color="black", fontweight="bold")
    ax.text(dt, di, "D", ha="center", va="center", color="black", fontweight="bold")
    ax.set_xlabel("source window position t")
    ax.set_ylabel("source variable i")
    ax.set_title(
        "Med(e) = Delta_tot - Delta_dir, claimed cells only"
    )
    fig.colorbar(im, ax=ax, label="Med(e)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="small sizes, for a fast smoke run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = str(results_dir(cfg))
    vcfg = cfg["var_mediation"]
    base = vcfg["quick"] if args.quick else vcfg

    print("=== Block B1 (omic_total.md): VAR with a planted mediator (direct-vs-total mediation) ===")
    results = run_var_mediation(
        n=base["n"], T=base["T"], m=base["m"], ar=base["ar"], beta=base["beta"], gamma=base["gamma"],
        noise_std=base["noise_std"], target_noise_std=base["target_noise_std"], seed=vcfg["seed"],
        delta_B=base["delta_B"], n_seeds=base["n_seeds"], rho_max=base["rho_max"], top_k=base["top_k"],
    )

    print(f"claimed edges (marginal-correlation claimant, top {base['top_k']}): {results['claimed']}")
    print(f"known mediated edge (dir=0, tot>0): {results['mediator_edge']}")
    print(f"known direct edge:                  {results['direct_edge']}")
    med = results["med"]
    print(
        f"mediated edge: Med(e)={med[results['mediator_edge']]:.4f}  "
        f"direct edge: Med(e)={med[results['direct_edge']]:.4f}"
    )

    row = {
        "omic_support_dir": results["omic_support_dir_mean"],
        "omic_support_tot": results["omic_support_tot_mean"],
        "auomic_dir": results["auomic_dir_mean"],
        "auomic_dir_std": results["auomic_dir_std"],
        "auomic_tot": results["auomic_tot_mean"],
        "auomic_tot_std": results["auomic_tot_std"],
        "gap": results["gap_mean"],
        "mediation_fraction": results["mediation_fraction_mean"],
    }
    df = pd.DataFrame([row])
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, "var_mediation.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(out_dir, "tab_var_mediation.tex")
    with open(tex_path, "w") as fh:
        fh.write(
            "% Table: VAR planted-mediator mediation (omic_total.md Block B1)\n"
            "% Same claimed set (marginal-correlation claimant) scored on AUOMIC_dir vs AUOMIC_tot\n"
        )
        fh.write(df.to_latex(index=False, float_format="%.3f", escape=False))

    fig_curves_path = os.path.join(out_dir, "fig_var_mediation_curves.pdf")
    fig_med_path = os.path.join(out_dir, "fig_var_mediation_med.pdf")
    fig_mcurve_path = os.path.join(out_dir, "fig_var_mediation_mcurve.pdf")
    plot_dir_tot_curves(results, fig_curves_path)
    plot_med_heatmap(results, fig_med_path)
    plot_mediation_curve(results, fig_mcurve_path)  # omic_new.md F2

    print(f"\nSaved: {csv_path}, {tex_path}, {fig_curves_path}, {fig_med_path}, {fig_mcurve_path}")


if __name__ == "__main__":
    main()
