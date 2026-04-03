# agentic-preferences

A finite MDP research testbed for measuring **agenticity** — proxies for how much an agent's planning matters — using exact dynamic programming (no RL training loops).

## Dependencies

Python 3.9+, NumPy, SciPy.

## Running

```bash
# Main experiments + plots
python environments/plot_pams.py      # plots 01–08: Q1/Q2/Q3 PAM analysis
python environments/plot_sampling.py  # plots 09–13: p-sweep, γ-sweep, T-sensitivity, S-sweep

# Legacy demos
python environments/mdp.py            # Week 2 shaping-invariant metrics demo
```

See `environments/plots_desc.md` for a description of each plot.

---

## File Reference

### `environments/core.py`
MDP dataclass and all DP primitives. Everything else imports from here.

- `MDP` — dataclass holding `T[S,A,S]`, `R[S,A,S]`, `gamma`, `terminal`, `d0`
- `value_iteration` — synchronous VI returning V*, Q*, π*
- `policy_evaluation` — exact V^π via linear system solve
- `value_under_random_policy` — V under uniform random policy
- `discounted_occupancy` — discounted state visitation measure d^π
- `finite_horizon_optimal_policy` — exact h-step backwards induction
- `finite_horizon_lookahead_policy` — h-step lookahead with arbitrary terminal bootstrap V
- `soft_value_iteration` — entropy-regularised (MCE) VI returning soft V*, Q*, π
- `mce_policy` — softmax policy from Q (temperature α)
- `mce_objective` — MCE objective value
- `add_potential_shaping` — applies F = γΦ(s') − Φ(s) to R; used to verify shaping invariance

---

### `environments/week2.py`
Week 2 shaping-invariant baseline metrics. Both use d0 weighting and cancel Φ offsets.

- `control_advantage` — E_{s~d0}[V*(s) − V^π0(s)]: how much better optimal is vs baseline
- `one_step_recovery` — E_{s~d*,a~Uniform,s'~T}[V*(s') − V^π0(s')]: recovery advantage at landing state after a random action

---

### `environments/week3.py`
Week 3/4 PAMs and the composite agenticity score. All metrics are shaping-invariant.

- `advantage_gap` — mean(max_a A* − min_a A*) over non-terminal states, normalised by range(V* − V^rand). Measures per-state action differentiation independent of reward scale.
- `vstar_variance_corrected` — Var of range-normalised (V* − V^rand); measures spread of the value landscape relative to its own dynamic range. Returns value in [0, 0.25].
- `early_action_mi` — I(A_{1:k}; G | s0) − I(A_{k+1:T}; G | s0): whether early actions are disproportionately decisive for return. Uses ε-greedy rollouts conditioned on s0.
- `advantage_sparsity` — fraction of non-terminal (s,a) pairs with |A*(s,a)| < threshold (diagnostic only)
- `effective_planning_horizon` — H_eps: minimum lookahead depth k for k-step policy to achieve ratio (J* − J^k)/(J* − J^rand) ≤ ε; bootstrap from V^rand. Normalised as H_eps/max_k.
- `mce_policy_entropy` — entropy of the MCE policy weighted by d0; high = near-uniform policy (flat reward), low = near-deterministic (strong reward signal)
- `agenticity_score` — composite score combining the above. Default weights: adv_gap=0.25, vstar_var=0.40, mi_diff=0.35. Optionally includes W2 metrics when `w2_scales` is provided.

---

### `environments/envs.py`
Hand-constructed MDP environments used in Q3 and as sanity checks.

- `gridworld` — rectangular grid with slip, absorbing goal; used in legacy Week 2 demo
- `make_chain_mdp` — linear chain (A=2: stay/advance); reward types: terminal, dense, lottery, progress
- `make_grid_mdp` — 5×5 grid (A=4: UDLR); reward types: goal, local, cliff

---

### `environments/experiments.py`
Random MDP generator and batch experiment runners.

- `random_mdp` — samples a random tabular MDP with configurable S, A, γ, k, R_type, T_type
- `norm_w2` — normalises W2 metrics via 1−exp(−x/scale) using empirical 95th-pct scales
- `run_pam_experiment` — main Q1/Q2/Q3 batch experiment:
  - Q1: fix T, vary R — PAM distributions and cross-PAM correlations by reward type
  - Q2: vary T structure — variance decomposition of composite into T-share vs R-share
  - Q3: score 7 hand-constructed MDPs (chain × 4 + grid × 3)
- `_print_pam_results` — pretty-prints Q1/Q2/Q3 results to stdout
- `run_p_sweep` — sweeps reward sparsity p for bernoulli or spike_slab R types
- `run_gamma_sweep` — sweeps discount factor γ across R conditions
- `run_t_sensitivity` — scores same R samples under uniform vs deterministic T
- `run_s_sweep` — sweeps state space size S across T types

---

### `environments/plot_pams.py`
Generates plots 01–08 from `run_pam_experiment` output. See `environments/plots_desc.md`.

### `environments/plot_sampling.py`
Generates plots 09–13 from the sweep runners. See `environments/plots_desc.md`.

### `environments/mdp.py`
Legacy Week 2 standalone demo. Superseded by the modular architecture above.

---

## Architecture Notes

**Shaping invariance:** all active metrics (adv_gap, vstar_var, early_action_mi, mce_entropy, H_eps) are invariant to potential shaping F(s,a,s') = γΦ(s') − Φ(s). The W2 metrics (control_advantage, one_step_recovery) are also shaping-invariant. H_eps and mce_entropy are additionally invariant to positive reward scaling and S'-redistribution.

**T-sensitivity:** metrics respond to both R and T by design. Agenticity is a property of an agent in a specific environment, not of R alone. The Q2 variance decomposition characterises how much T vs R drives scores in a given deployment context.

**Composite:** adv_gap and vstar_var share a common reference scale (range(V* − V^rand)), making their contributions commensurable. mi_diff is normalised to [0,1] via (raw + 1)/2. H_eps and mce_entropy are diagnostic and excluded from the composite.
