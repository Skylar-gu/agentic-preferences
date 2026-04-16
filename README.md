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

plans+notes/            Research plans, design notes, and experimental observations
```

---

## Running experiments

Each experiment folder is self-contained. Run from inside the folder:

```bash
# Experiment 1 — Q1/Q2/Q3 PAM analysis
cd 01_pam_baseline
python 01_build.py          # run experiment, save results/ (~2 min)
python 01_build.py --fast   # smoke test (~10 sec)
python 02_eval.py           # print summary tables
python 03_plot.py           # generate figures/ (PDFs)
python check_job_status.py  # check if build output is ready

# Experiment 2 — sampling sweeps
cd 02_sampling_sweeps
python 01_build.py          # run all sweeps, save results/ (~3-5 min)
python 01_build.py --fast   # smoke test (~15 sec)
python 02_eval.py
python 03_plot.py
python check_job_status.py
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
- `soft_value_iteration` → MCE soft Bellman `(V_soft, Q_soft, π_mce)`
- `add_potential_shaping` → applies `F = γΦ(s') − Φ(s)`

### `environments/metrics.py`

Shaping-invariant baseline metrics (d0-weighted, cancel Φ via value differences).

- `control_advantage(mdp, pi0)` — `E_{d0}[V* − V^{π0}]`
- `one_step_recovery(mdp, pi0)` — recovery advantage after a random action from optimal occupancy

### `environments/pams.py`

PAMs and composite agenticity score. All shaping-invariant.

| PAM | Composite weight | Notes |
|---|---|---|
| `advantage_gap` | 0.25 | normalised by range(V*−V^rand) |
| `vstar_variance` | 0.40 | range-normalised variance of V*−V^rand; in [0, 0.25] |
| `early_action_mi` | 0.35 | I(early actions; G\|s0) − I(late actions; G\|s0); slow, excluded by default |
| `effective_planning_horizon` | diagnostic | H_eps: min lookahead depth to close J*−J^rand gap; auto-computed max_k |
| `mce_policy_entropy` | diagnostic | entropy of MCE policy weighted by d0 |
| `agenticity_score` | — | composite + all individual metrics in one dict |

### `environments/envs.py`

- `make_chain_mdp(n, reward_type, backtrack)` — 10-state chain (A=2); types: `terminal`, `dense`, `lottery`, `progress`
- `make_grid_mdp(rows, cols, reward_type)` — 5×5 grid (A=4); types: `goal`, `local`, `cliff`
- `gridworld(w, h, goal_xy, slip, gamma)` — grid with slip, absorbing goal (shaping-invariant formulation)

### `environments/runners.py`

- `random_mdp(S, A, gamma, k, R_type, T_type, ...)` — random tabular MDP generator
  - R_types: `gaussian`, `uniform`, `bernoulli`, `spike_slab`, `potential`, `goal`
  - T_types: `random`, `dirichlet`, `uniform`, `deterministic`
- `run_pam_experiment(...)` — Q1/Q2/Q3 batch (used by `01_pam_baseline/`)
- `run_p_sweep / run_gamma_sweep / run_t_sensitivity / run_s_sweep` — sweep runners (used by `02_sampling_sweeps/`)

---

## Experiments

### `01_pam_baseline` — Q1/Q2/Q3 PAM analysis (Plots 01–08)

Benchmarks PAMs across random and human-made MDPs.

| Plot | Description |
|---|---|
| 01 | Q3 heatmap — 8 human MDPs × 7 normalised PAMs |
| 02 | Q3 grouped bar — composite + key metrics per MDP |
| 03 | Q1 boxplots — PAM distributions by reward type (S=10) |
| 04 | Q2 variance decomp — T vs R share of composite variance |
| 05 | Q1 scatter — adv_gap vs MCE entropy by reward type |
| 06 | Q1 radar — mean PAM profile per reward type |
| 07 | Q1 histograms — agenticity score distributions per reward type |
| 08 | R_scale sweep — Gaussian σ ∈ {0.1, 0.5, 1.0, 2.0, 5.0} |

### `02_sampling_sweeps` — Group 1/2/3 sampling sweeps (Plots 09–13)

Investigates how PAM scores vary with reward sparsity, discount factor, topology, and state-space size.

| Plot | Description |
|---|---|
| 09 | Bernoulli p-sweep — sparsity vs agenticity (lottery failure at p=1.0) |
| 10 | Spike-and-slab p-sweep — magnitude variation prevents lottery failure |
| 11 | γ-sweep — PAM scores vs discount factor (S=20) |
| 12 | T-sensitivity — same R scored under Uniform vs Deterministic T (S=20) |
| 13 | S-sweep — adv_gap and vstar_var vs state-space size × T type |

---

## Plans and notes

See `plans+notes/` for research plans, design specs, and experimental observations:

| File | Contents |
|---|---|
| `pipeline.md` | Corpus analysis plan — 15 reward functions from classical RL, RLHF, and verifiable rewards |
| `pipeline2.md` | Random sampling experiment design — theoretical predictions for Groups 1/2/3 |
| `mar31.md` | Experimental observations — key findings, metric fixes (H_eps, vstar_var), PAM status |
| `specmar25.md` | Implementation spec for PAMs and random MDP generator; updated status |
| `description.md` | Library reference — module descriptions and design notes |
| `mar16.md` | Environment and metrics reference with result tables |
| `notes.md` | Open research questions — KL penalty, PRM/ORM, MACHIAVELLI |
| `plots_desc.md` | Per-plot observations for all 13 figures |
