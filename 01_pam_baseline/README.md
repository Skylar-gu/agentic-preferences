# 01 — PAM Baseline: Q1/Q2/Q3 Agenticity Analysis

Benchmarks Planning Agenticity Measures (PAMs) across random and human-made finite MDPs using exact dynamic programming. No learning loops.

## Steps

| Script | What it does | Runtime |
|---|---|---|
| `01_build.py` | Runs Q1/Q2/Q3 experiment + R-scale sweep, saves `results/pam_experiment.pkl` | ~2 min |
| `02_eval.py` | Loads results, prints Q1/Q2/Q3 summary tables and Pearson correlation matrices | <1 sec |
| `03_plot.py` | Loads results, generates Figures 01–08 in `figures/` | ~5 sec |
| `check_job_status.py` | Reports whether build output is ready | instant |

## Quick start

```bash
# From this directory:
python 01_build.py          # full run
python 01_build.py --fast   # smoke test (~10 sec, small n)
python 02_eval.py
python 03_plot.py
```

## Dependencies

Python 3.9+, NumPy, SciPy, Matplotlib. Shared library: `../environments/` (added to path at runtime — no install needed).

## Experiment design

**Q1 — PAM correlation (T fixed, R varies):** For each (S, R_type), one canonical transition matrix T is held fixed while n=50 reward matrices are sampled. PAM scores are recorded as an (n × 5) matrix. Isolates variance attributable to R.

**Q2 — Variance decomposition (T vs R):** For each (S, R_type, T_structure), n_fixed_T=5 independent transition matrices are sampled. For each T, n_R=50 reward matrices are evaluated. The ratio `between_T_var / total_var` measures how much T topology (vs reward structure) drives composite agenticity scores.

**Q3 — Human-made MDPs:** Eight built-in environments (chain and grid variants) are scored against the full PAM suite for interpretability grounding.

## Figures

| Figure | Description |
|---|---|
| `01_q3_heatmap.pdf` | Q3: 8 human MDPs × 7 normalised PAMs (YlOrRd heatmap, sorted by composite) |
| `02_q3_grouped_bar.pdf` | Q3: composite + adv_gap, vstar_var, H_eps, ctrl_adv side by side per MDP |
| `03_q1_boxplots_by_Rtype.pdf` | Q1: PAM distributions by reward type at S=10 (boxplots; potential = negative control) |
| `04_q2_variance_decomp.pdf` | Q2: T-share of composite variance per (S, R_type, T_structure) |
| `05_q1_scatter_gap_vs_entropy.pdf` | Q1: adv_gap vs MCE entropy scatter coloured by R_type (S=5, S=10) |
| `06_q1_radar_by_Rtype.pdf` | Q1: mean normalised PAM profile per R_type (polar chart) |
| `07_reward_dist_agenticity.pdf` | Q1: agenticity score histograms per reward type with high-agency fraction labels |
| `08_rscale_sweep_agenticity.pdf` | Gaussian R_scale sweep (σ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}); tests scale sensitivity |

## Key PAMs

| Metric | Function | Shaping-invariant? |
|---|---|---|
| Advantage gap | `advantage_gap` | Yes |
| V\*−V^rand variance | `vstar_variance` | Yes |
| Effective planning horizon | `effective_planning_horizon` | Yes |
| MCE policy entropy | `mce_policy_entropy` | Yes |
| Early-action MI | `early_action_mi` | Yes (excluded by default; slow) |
| Composite | `agenticity_score` | Yes |
