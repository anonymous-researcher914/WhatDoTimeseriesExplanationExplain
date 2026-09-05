"""Give the VAR metric-validation experiment a genuine noise floor.

`run_var_omic.py`'s own GOOD=1.0/ANTI=0.0 are tautologies under `RawWindowOracle`: non-support
edges have coefficient EXACTLY 0, so Delta is exactly 0 off-support and exactly nonzero
on-support -- the two Delta distributions never overlap, so the support-level score is decided
before any attribution is involved and the experiment cannot demonstrate discrimination (H3).

This script adds a FLOOR: every non-support edge gets a small-but-nonzero coefficient (scale
`floor`), so "support" (large coefficients) and "non-support" (floor-scale coefficients) Delta
distributions genuinely overlap once `floor` is large enough relative to the support scale, and
sweeps `floor` to find where OMIC's GOOD/RANDOM/BAD/ANTI separation degrades -- THAT sweep is
the metric-validation experiment Fig. 1 is currently claimed to be.

Usage:
    python experiments/run_var_noise_floor.py [--config configs/default.yaml] [--quick]
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
from dagfaith.oracle import RawWindowOracle
from dagfaith.omic import auomic, evaluate

from experiments.run_var_omic import METHOD_COLORS, _all_edges, build_attributions


def sample_dbn_with_noise_floor(
    D: int, T: int, n: int, D_out: int, sparsity: float, noise_std: float, floor: float, seed: int,
):
    """`dagfaith.dbn.sample_dbn`, then ADD a small N(0, floor^2) coefficient to every cell that
    was exactly 0 -- "support" (the original sparsity-selected cells, coefficient scale ~0.8)
    and "non-support" (now coefficient scale `floor`) Delta distributions overlap once `floor`
    is large enough, unlike the tautological floor=0 case. `Gstar` (the support DEFINITION used
    by GOOD/BAD/ANTI) still marks the ORIGINAL sparse cells -- only their Delta separation from
    the rest changes, not what counts as "true support"."""
    rng = np.random.default_rng(seed)
    X, _Y, B, Gstar, _cs = dbn.sample_dbn(
        D=D, T=T, n=n, D_out=D_out, sparsity=sparsity, nonlinear=False, noise_std=noise_std, seed=seed
    )
    if floor > 0:
        zero_mask = ~Gstar
        B = B.copy()
        B[zero_mask] += rng.normal(scale=floor, size=int(zero_mask.sum()))
        Y = np.einsum("itj,nti->nj", B, X) + rng.normal(scale=0.1, size=(n, D_out))
    else:
        Y = _Y
    return X, Y, B, Gstar


def run_noise_floor_sweep(
    D: int, T: int, n: int, sparsity: float, noise_std: float, seed: int,
    delta_B: int, rho_max: float, floors: list[float],
) -> pd.DataFrame:
    rows = []
    for floor in floors:
        seed_everything(seed)
        X, _Y, B, Gstar = sample_dbn_with_noise_floor(D, T, n, D, sparsity, noise_std, floor, seed)
        oracle = RawWindowOracle(B)
        cond_model = analytic_gaussian_cond_model_for_ar_inputs(D, T, ar=0.5, noise_std=noise_std)
        candidate_edges = _all_edges(D, T, D)

        build_rng = np.random.default_rng(seed + 777)
        attributions = build_attributions(candidate_edges, Gstar, B, build_rng)

        rng = np.random.default_rng(seed + 999)
        cond_sampler = cond_model.as_cond_sampler(rng)
        delta = delta_dict(oracle.numpy_forward, X, candidate_edges, cond_sampler, B=delta_B, rng=rng)

        # overlap diagnostic: how much do the support/non-support |Delta| distributions overlap?
        support_delta = np.abs([delta[e] for e in candidate_edges if Gstar[e]])
        nonsupport_delta = np.abs([delta[e] for e in candidate_edges if not Gstar[e]])
        overlap = float(np.mean(nonsupport_delta > np.median(support_delta))) if len(support_delta) else float("nan")

        row = {"floor": floor, "overlap": overlap}
        for method, (claimed, attribution) in attributions.items():
            result = evaluate(claimed, attribution, delta, candidate_edges, rho_max=rho_max)
            row[f"{method}_auomic"] = result["auomic"]
            row[f"{method}_support"] = result["omic_support"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_noise_floor_sweep(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in ("GOOD", "RANDOM", "BAD", "ANTI"):
        axes[0].plot(df["floor"], df[f"{method}_auomic"], label=method, color=METHOD_COLORS[method], marker="o", linewidth=2)
        axes[1].plot(df["floor"], df[f"{method}_support"], label=method, color=METHOD_COLORS[method], marker="o", linewidth=2)
    for ax, title in ((axes[0], "AUOMIC"), (axes[1], "OMIC_support")):
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("noise floor (non-support coefficient scale)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. noise floor (H3 overlap sweep)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="center left", fontsize=8)
    fig.suptitle("H3: GOOD/RANDOM/BAD/ANTI separation as support/non-support Delta distributions overlap")
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
    vcfg = cfg["var_noise_floor"]
    base = vcfg["quick"] if args.quick else vcfg

    print("=== omic_new.md Block H3: VAR noise-floor overlap sweep ===")
    df = run_noise_floor_sweep(
        D=base["D"], T=base["T"], n=base["n"], sparsity=base["sparsity"], noise_std=base["noise_std"],
        seed=vcfg["seed"], delta_B=base["delta_B"], rho_max=base["rho_max"], floors=base["floors"],
    )
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, "var_noise_floor.csv")
    df.to_csv(csv_path, index=False)
    fig_path = os.path.join(out_dir, "fig_var_noise_floor.pdf")
    plot_noise_floor_sweep(df, fig_path)

    degrade_floor = None
    for _, row in df.iterrows():
        if abs(row["BAD_auomic"] - row["RANDOM_auomic"]) < 0.05 or row["GOOD_auomic"] < 0.95:
            degrade_floor = row["floor"]
            break
    print(f"\nSeparation first degrades around floor={degrade_floor} "
          f"({'no degradation in this sweep range' if degrade_floor is None else 'BAD/RANDOM converge or GOOD drops off ceiling'})")

    print(f"\nSaved: {csv_path}, {fig_path}")


if __name__ == "__main__":
    main()
