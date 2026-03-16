# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run the MDP lab demos:
```bash
python environments/mdp.py    # Week 2: shaping-invariant metrics (control advantage, one-step recovery, planning pressure)
python environments/mdp2.py   # Week 3: agenticity proxy testbed (Q-decomp ratio, V* variance, early-action MI)
```

Dependencies: Python 3.9+, NumPy, SciPy.

## Architecture

This is a **finite MDP research testbed** for measuring "agenticity" — proxies for how much an agent's planning matters — using exact dynamic programming (no RL training loops).

### Two parallel implementations

**`environments/mdp.py`** (Week 2) — shaping-invariant metrics:
- `FiniteMDP` class: holds `T[S,A,S]`, `R[S,A,S]`, `gamma`, `d0`
- Core DP: `value_iteration`, `policy_evaluation`, `discounted_occupancy`, `finite_horizon_optimal_policy`, `finite_horizon_lookahead_policy`
- `add_potential_shaping`: applies `F = γΦ(s') − Φ(s)` to rewards; metrics must be invariant to this
- Metrics: `control_advantage`, `one_step_recovery`, `planning_pressure`
- Demo: 5×5 gridworld with slip; verifies metrics give ~0 difference under shaping

**`environments/mdp2.py`** (Week 3) — agenticity proxies:
- `MDP` dataclass: similar to `FiniteMDP` but uses `@dataclass` and tracks `terminal` states
- Proxies: `q_decomposition_ratio` (NOT shaping-invariant), `vstar_variance_corrected`, `early_action_mi`, `reward_sparsity`
- `agenticity_score`: composite score with weights Q=0.25, V*var=0.40, MI=0.35
- Demo: benchmarks chain MDPs (terminal/dense/lottery/progress) and grid MDPs (goal/local/cliff)

### Key design constraint

All metrics in `mdp.py` are designed to cancel potential-shaping offsets (differences between value functions, not absolute values). The `mdp2.py` proxies are NOT all shaping-invariant — this is documented and flagged per proxy.

### Known limitations (from `mdp2.py` docstring)

- `early_action_mi` uses epsilon-greedy (ε=0.3) sampling; needs `n_episodes >= 500` for reliable estimates
- `q_decomposition_ratio` is not invariant to potential shaping
- Chain MDP is too simple to distinguish lottery vs terminal reward via MI — need a branching MDP
