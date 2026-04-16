# 02 — Sampling Sweeps: Group 1/2/3 Experiments

Investigates how PAM scores vary with reward sparsity, discount factor, transition topology, and state-space size using random MDP sampling. Corresponds to `pipeline/pipeline2.md` Groups 1–3.

## Steps

| Script | What it does | Runtime |
|---|---|---|
| `01_build.py` | Runs all four sweep experiments, saves `results/sweep_results.pkl` | ~3–5 min |
| `02_eval.py` | Loads results, prints per-group summary statistics and correlations | <1 sec |
| `03_plot.py` | Loads results, generates Figures 09–13 in `figures/` | ~5 sec |
| `check_job_status.py` | Reports whether build output is ready | instant |

## Quick start

```bash
# From this directory:
python 01_build.py          # full run
python 01_build.py --fast   # smoke test (~15 sec)
python 02_eval.py
python 03_plot.py
```

## Dependencies

Python 3.9+, NumPy, SciPy, Matplotlib. Shared library: `../environments/` (added to path at runtime — no install needed).

## Experiment design

**Group 1 — Reward sparsity p-sweep (Plots 09-10):**
Fix one canonical T (S=10, A=4, Dirichlet). Sweep R_scale=p ∈ {0.01, 0.05, 0.2, 0.5, 1.0} for both Bernoulli and spike-and-slab reward types. Tests whether PAMs correctly distinguish lottery failure (Bernoulli p=1.0, all entries equal) from genuinely sparse agentic rewards.

**Group 2 — T-sensitivity (Plot 12):**
Fix n=100 reward matrices, score each under Uniform T and Deterministic T independently. Scatter plot reveals whether PAMs measure an (R,T) joint property or R alone. Ideal: points on the diagonal (T-invariant). Off-diagonal → PAM is confounded by topology.

**Group 3a — γ sweep (Plot 11):**
Fix S=20, T=random. Sweep γ ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 0.99} for spike-and-slab (p=0.1) and Gaussian (σ=1) rewards. Prediction: PAM scores increase monotonically with γ; H_eps is the most sensitive.

**Group 3b — S sweep (Plot 13):**
Fix Gaussian R (σ=1), A=4, γ=0.95. Sweep S ∈ {5, 10, 20, 50, 100} × T type ∈ {Uniform, Dirichlet(0.1), Deterministic}. Prediction: Uniform T → adv_gap and vstar_var collapse as S→∞ (averaging suppresses Q-spread); Dirichlet and Deterministic remain bounded.

## Figures

| Figure | Description |
|---|---|
| `09_bernoulli_p_sweep.pdf` | Bernoulli reward sparsity sweep: adv_gap, vstar_var, H_eps, MCE entropy vs p |
| `10_spike_slab_p_sweep.pdf` | Spike-and-slab sparsity sweep: same metrics; no collapse at p=1 (unlike Bernoulli) |
| `11_gamma_sweep.pdf` | PAM scores vs γ for spike_slab and gaussian R (mean ± std, S=20) |
| `12_t_sensitivity.pdf` | Scatter: same R scored under Uniform vs Deterministic T (S=20, n=100) |
| `13_s_sweep.pdf` | adv_gap and vstar_var vs S × T type; Uniform T predicted to collapse |

## Key findings (from `environments/plots_desc.md`)

- **Bernoulli p=1.0** correctly gives adv_gap=0 (lottery failure mode — all entries identical).
- **Spike-and-slab** avoids this: non-zero magnitude variation at p=1.0 still produces Q-spread.
- **H_eps** tracks γ, not R structure — it increases monotonically with γ but does not distinguish spike_slab from gaussian.
- **T-sensitivity**: adv_gap is high under Uniform T even for random R (R-driven action differentiation not suppressed by normalization); vstar_var is near-diagonal and T-insensitive.
- **S sweep**: range-normalization makes adv_gap S-invariant under Uniform T (numerator and denominator both scale as 1/√S, cancelling); vstar_var collapses for all T types due to Gaussian R dilution.
