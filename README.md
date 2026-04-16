# agentic-preferences

A finite MDP research testbed for measuring **agenticity** — how much an agent's planning matters in a given environment — using exact dynamic programming. No RL training loops.

All metrics are invariant to potential-based reward shaping (`R' = R + γΦ(s') − Φ(s)`).

---

## Repo structure

```
environments/           Shared library (import from here)
  core.py               MDP dataclass + all DP / MCE / shaping primitives
  metrics.py            Shaping-invariant baseline metrics (control_advantage, one_step_recovery)
  pams.py               PAMs + agenticity_score composite
  envs.py               Hand-constructed MDPs (chain, grid variants)
  runners.py            Random MDP generator + batch experiment runners

01_pam_baseline/        Experiment 1 — Q1/Q2/Q3 PAM analysis (Plots 01–08)
02_sampling_sweeps/     Experiment 2 — Group 1/2/3 sampling sweeps (Plots 09–13)
03_mce_calibration/     Experiment 3 — MCE entropy alpha calibration (Plots 14–17)
04_results_summary/     Cross-experiment summary — metric-organised figures + values doc

plans+notes/            Synthesized research notes (research_notes.md)
```

---

## Running experiments

Each experiment folder is self-contained:

```bash
# Experiment 1 — Q1/Q2/Q3 PAM analysis
cd 01_pam_baseline
python 01_build.py          # run experiment, save results/ (~2 min)
python 01_build.py --fast   # smoke test (~10 sec)
python 02_eval.py           # print summary tables
python 03_plot.py           # generate figures/ (PDFs, Plots 01–08)

# Experiment 2 — sampling sweeps
cd 02_sampling_sweeps
python 01_build.py          # run all sweeps, save results/ (~3–5 min)
python 01_build.py --fast   # smoke test (~15 sec)
python 02_eval.py
python 03_plot.py           # Plots 09–13

# Experiment 3 — MCE alpha calibration
cd 03_mce_calibration
python 01_sweep.py          # N_T=5 canonical T's, N_R=500 R samples (~3 min)
python 01_sweep.py --fast   # smoke test (~15 sec)
python 02_plot.py           # Plots 14–17

# Cross-experiment summary (requires experiments 1 and 2 built first)
cd 04_results_summary
python generate.py          # figures A–E + results/summary.md
```

---

## Dependencies

Python 3.9+, NumPy, SciPy, Matplotlib. No install needed — `environments/` is added to `sys.path` at runtime by each experiment script.

---

## Library reference

### `environments/core.py`

MDP dataclass and all DP / MCE primitives.

- `MDP(S, A, T, R, gamma, terminal, d0)` — tabular MDP
- `value_iteration` → `(V*, Q*, π*)`
- `policy_evaluation` → `V^π`
- `value_under_random_policy` → `V^rand`
- `discounted_occupancy` → `d^π`
- `finite_horizon_lookahead_policy` → depth-h lookahead with value bootstrap
- `soft_value_iteration(mdp, alpha)` → MCE soft Bellman `(V_soft, Q_soft, π_mce)`
- `add_potential_shaping` → applies `F = γΦ(s') − Φ(s)`

### `environments/metrics.py`

Shaping-invariant baseline metrics (cancel Φ via value differences, d0-weighted).

- `control_advantage(mdp, pi0)` — `E_{d0}[V* − V^{π0}]`
- `one_step_recovery(mdp, pi0)` — recovery advantage after a random action from optimal occupancy

### `environments/pams.py`

PAMs and composite agenticity score. All core PAMs are shaping-invariant.

| PAM | Default weight | Notes |
|---|---|---|
| `advantage_gap` | 0.50 | Mean best−worst action spread per state, normalised by range(V*−V^rand) |
| `vstar_variance` | 0.50 | Variance of range-normalised (V*−V^rand); in [0, 0.25], ×4 → [0,1] |
| `early_action_mi` | 0.50 | I(early actions; G\|s0) − I(late actions; G\|s0); off by default (`compute_mi=False`) |
| `mce_policy_entropy` | 0.50 | Entropy of MCE policy (R normalised by std(R), α=0.25); in composite when `compute_entropy=True` (default) |
| `effective_planning_horizon` | diagnostic | H_eps: min lookahead depth to close J*−J^rand gap; auto-computed max_k |
| `agenticity_gap` | diagnostic | J(π*) − J(π*_myopic); not shaping-invariant |
| `option_value` | diagnostic | E_s[V*(s) − max_a E[R(s,a)]]; not shaping-invariant |
| `agenticity_score` | — | Composite + all individual metrics in one dict |

Composite weights are equal and auto-renormalised across active components. Set `compute_entropy=False` or `compute_mi=True` to change which metrics are included.

**MCE normalisation note:** Other PAMs normalise their *outputs* (ratios of V*−V^rand quantities). MCE normalises its *input* — R is divided by std(R) before soft VI — because α lives in Q-value space and is otherwise scale-sensitive. α=0.25 was chosen by empirical sweep over 6 α values × 5 fixed T matrices × 500 R samples (see Experiment 3).

### `environments/envs.py`

- `make_chain_mdp(n, reward_type, backtrack)` — 10-state chain (A=2); types: `terminal`, `dense`, `lottery`, `progress`
- `make_grid_mdp(rows, cols, reward_type)` — 5×5 grid (A=4); types: `goal`, `local`, `cliff`
- `gridworld(w, h, goal_xy, slip, gamma)` — grid with slip, absorbing goal

### `environments/runners.py`

- `random_mdp(S, A, gamma, k, R_type, R_scale, T_type, T_alpha, ...)` — random tabular MDP
  - R_types: `gaussian`, `uniform`, `bernoulli`, `spike_slab`, `potential`, `goal`, `uniform_simplex`
  - T_types: `random`, `dirichlet`, `uniform`, `deterministic`
- `run_pam_experiment(...)` — Q1/Q2/Q3 batch (used by `01_pam_baseline/`)
- `run_p_sweep / run_gamma_sweep / run_t_sensitivity / run_s_sweep` — sweep runners (used by `02_sampling_sweeps/`)
- `fraction_agentic(prior, threshold, n_samples, n_bootstrap, ...)` — estimates P(composite > threshold) under a reward prior with 95% bootstrap CI; comparable to Turner et al. (2021)

---

## Experiments

### `01_pam_baseline` — Q1/Q2/Q3 PAM analysis (Plots 01–08)

Benchmarks PAMs across random and human-made MDPs.

| Plot | Description |
|---|---|
| 01 | Q3 heatmap — 8 human MDPs × normalised PAMs |
| 02 | Q3 grouped bar — composite + key metrics per MDP |
| 03 | Q1 boxplots — PAM distributions by reward type (S=10) |
| 04 | Q2 variance decomp — T vs R share of composite variance |
| 05 | Q1 scatter — adv_gap vs vstar_var by reward type |
| 06 | Q1 radar — mean PAM profile per reward type |
| 07 | Q1 histograms — agenticity score distributions per reward type |
| 08 | R_scale sweep — Gaussian σ ∈ {0.1, 0.5, 1.0, 2.0, 5.0} |

### `02_sampling_sweeps` — Group 1/2/3 sampling sweeps (Plots 09–13)

Investigates how PAM scores vary with reward sparsity, discount factor, topology, and state-space size.

| Plot | Description |
|---|---|
| 09 | Bernoulli p-sweep — sparsity vs agenticity; lottery failure at p=1.0 confirmed |
| 10 | Spike-and-slab p-sweep — magnitude heterogeneity prevents lottery failure |
| 11 | γ-sweep — PAM scores vs discount factor (S=20) |
| 12 | T-sensitivity — same R scored under Uniform vs Deterministic T (S=20) |
| 13 | S-sweep — adv_gap and vstar_var vs state-space size × T type |

### `03_mce_calibration` — MCE alpha calibration (Plots 14–17)

Calibrates the α hyperparameter for MCE entropy using N_T=5 fixed T matrices and N_R=500 R samples per cell.

| Plot | Description |
|---|---|
| 14 | Per-alpha histograms by R_type; T matrices overlaid (T-sensitivity visible as band spread) |
| 15 | Mean ± std vs α per R_type; thin lines = individual T's, band = between-T std |
| 16 | T-sensitivity grid at α=0.25 — rows = R_type, columns = T; dashed = T mean, dotted = grand mean |
| 17 | Between-type variance vs α — identifies discrimination sweet-spot; marks chosen α=0.25 |

### `04_results_summary` — Cross-experiment summary

Aggregates results from experiments 1 and 2 into metric-organised views. Run `python generate.py` after both experiments are built.

| Output | Description |
|---|---|
| `figures/A_adv_gap_norm.pdf` | Advantage gap across all test conditions |
| `figures/B_vstar_var_norm.pdf` | V* variance across all test conditions |
| `figures/C_mce_entropy_norm.pdf` | MCE entropy across all test conditions |
| `figures/D_H_eps_norm.pdf` | Planning horizon across all test conditions |
| `figures/E_composite.pdf` | Composite score across all test conditions |
| `results/summary.md` | Tables of mean ± std by metric × test type (Q3, Q1, σ-sweep, p-sweeps, γ-sweep, T-sensitivity Pearson r, S-sweep) |

Each figure has 6 panels: Q3 human MDPs, Q1 by R_type, Gaussian σ sweep, p-sweep (Bernoulli + spike_slab), γ-sweep, S-sweep. Plot references annotated on each panel.

---

## Research notes

See `plans+notes/research_notes.md` for the synthesized research notes covering:
- Reward priors (Gaussian, spike-and-slab, uniform simplex) and their justifications
- Agenticity definitions (A–F) and known failure modes
- Theoretical predictions for all sampling experiments
- Key experimental findings and metric fixes (H_eps, vstar_variance, MCE entropy)
- Corpus analysis protocol for 15 real-world reward functions
- Remaining open questions and next steps
