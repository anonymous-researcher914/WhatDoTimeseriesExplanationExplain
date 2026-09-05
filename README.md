# OMIC: On-manifold Intervention Consistency


---

## Repository layout

```

dagfaith/                   Supporting library for experiments and its evaluations are built on
  config.py                 Global seeding, YAML config loader, results_dir
  dbn.py                    DBN / MDDAG data generation , on-manifold conditional samplers
  cond_baseline.py          Conditional baseline E[x_s|x_-s] and interpolation path TEIG
                             integrates along; analytic (exact) Gaussian conditional models
  oracle.py                 Analytic oracle forecasters (exact conditional-mean models) whose
                             true graph is known exactly, for confound-free evaluation
  models.py                 GRU/LSTM/TCN trained forecasters (optional, for estimated-conditional
                             evaluation instead of the analytic oracle)
  intervention.py           delta_effect: the on-manifold interventional effect primitive OMIC
                             is built on (black-box, gradient-free)
  omic.py                   OMIC — On-Manifold Interventional Consistency metric (Mann-Whitney
                             U-statistic + bootstrap CI), scored on claimed edges
  metrics.py                Graph metrics: SHD, edge-F1, F1-optimal thresholding
  real_data.py               Real-data windowing + empirical on-manifold conditional sampler

experiments/
  run_beijing_multitask.py   Run main paper case studies on classification + TIMING, forecasting + KARMA,
                             and regression + TimeSHAP on BeijingPM2.5 dataset
  run_var_mediation.py       Mediation ablation experiment for supplementary material
  run_var_omic.py            Run main paper Figure 1 experiment

configs/
  default.yaml               All hyperparameters (seed, DBN size, sparsity, OMIC
                             bootstrap settings, …)

data/
  generated                  Real generated multivariate series 
  raw                        Real raw unprocessed dataset

results/                     Auto-created; CSVs and LaTeX tables written here
run_all.py                   One-shot runner for main paper

```

---

## Installation

```bash
pip install -r requirements.txt
```

Deps: `numpy scipy pandas torch pyyaml` — no baseline-explainer libraries (SHAP/LIME/captum/
tigramite/…) are needed; TEIG is the only attribution method in this codebase.

---

## Quickstart

```bash
# Run everything (recovery + faithfulness) in one shot
python run_all.py

# Fast smoke run with small sizes
python run_all.py --quick

# Or run each experiment individually
python experiments/run_recovery.py
python experiments/run_faithfulness.py

# TEIG + OMIC on a real trained forecaster (GRU/LSTM/TCN), not the analytic oracle
python experiments/run_trained.py
```

All commands accept `--config <path>` (default `configs/default.yaml`) and `--quick`.

---

## Experiment outputs

Written to `results/` (created automatically)


---

## Configuration

Edit `configs/default.yaml`:

- `seed` — global random seed

---

