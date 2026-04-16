## Finite MDP Lab

Exact dynamic programming on tabular MDPs for measuring agenticity proxies. No learning loops. All metrics invariant under potential-based reward shaping (`R' = R + γΦ(s') − Φ(s)`).

### Requirements

Python 3.9+, NumPy, SciPy, Matplotlib

### MDP class

`MDP(S, A, T, R, gamma, terminal, d0)` — dataclass holding transition matrix `T[s,a,s']`, reward matrix `R[s,a,s']`, discount `gamma`, set of absorbing `terminal` states, and start distribution `d0` (defaults to uniform).

---

### `environments/core.py` — DP primitives

- `value_iteration(mdp)` → `(V*, Q*, pi*)`
- `policy_evaluation(mdp, pi)` → `V^pi` (deterministic or stochastic)
- `value_under_random_policy(mdp)` → `V^rand`
- `discounted_occupancy(mdp, pi)` → discounted state visitation `d^pi`
- `finite_horizon_optimal_policy(mdp, h)` → backward induction for truncated horizon
- `finite_horizon_lookahead_policy(mdp, h, V_terminal)` → depth-h lookahead with value bootstrap
- `soft_value_iteration(mdp, alpha)` → `(V_soft, Q_soft, pi_mce)` via soft Bellman `V^S = α·logsumexp(Q^S/α)`
- `mce_policy(Q_soft, alpha)` → `softmax(Q^S/α)` stochastic policy
- `mce_objective(mdp, Q_soft, alpha)` → `mean_s V^S(s)` over non-terminal states
- `add_potential_shaping(mdp, Phi)` → new MDP with `R' = R + γΦ(s') − Φ(s)`

---

### `environments/metrics.py` — Shaping-invariant baseline metrics

Both use d0 weighting and cancel Φ offsets via value differences.

| Metric | Function | Formula |
|---|---|---|
| Control advantage | `control_advantage(mdp, pi0)` | `E_{d0}[V* − V^{pi0}]` |
| One-step recovery | `one_step_recovery(mdp, pi0)` | `E_{s~d*,a~Unif,s'~T}[V*(s') − V^{pi0}(s')]` |

---

### `environments/pams.py` — PAMs and composite agenticity score

All metrics are shaping-invariant.

| PAM | Function | What it measures | In composite? |
|---|---|---|---|
| Advantage gap | `advantage_gap(mdp, Q, V, V_rand)` | Mean spread best–worst action per state, normalised by range(V*−V^rand) | Yes (0.25) |
| V*−V^rand variance | `vstar_variance(mdp, V_star, V_rand)` | Var of range-normalised (V*−V^rand); spread of value landscape in [0, 0.25] | Yes (0.40) |
| Early-action MI | `early_action_mi(mdp, pi, ...)` | `I(A_{1:k}; G\|s0) − I(A_{k+1:T}; G\|s0)` — whether early actions are disproportionately decisive | Yes (0.35) |
| Advantage sparsity | `advantage_sparsity(mdp, Q, V)` | Fraction of near-zero advantages | Diagnostic only |
| Effective horizon | `effective_planning_horizon(mdp)` | `H_eps`: min lookahead k for k-step policy to close (J*−J^rand) gap to within ε | Diagnostic only |
| MCE entropy | `mce_policy_entropy(mdp, alpha)` | Entropy of MCE policy weighted by d0; high = flat reward, low = deterministic | Diagnostic only |
| Composite | `agenticity_score(mdp)` | Weighted sum of above. Optionally includes baseline metrics when `w2_scales` provided. | — |

---

### `environments/envs.py` — Hand-constructed MDP environments

- `gridworld(w, h, goal_xy, step_cost, slip, gamma)` — grid with slip, absorbing goal (shaping-invariant formulation)
- `make_chain_mdp(n, gamma, reward_type, backtrack)` — linear chain (A=2); reward types: `terminal`, `dense`, `lottery`, `progress`
- `make_grid_mdp(rows, cols, gamma, reward_type)` — 5×5 grid (A=4); reward types: `goal`, `local`, `cliff`

---

### `environments/runners.py` — Random MDP generator + batch runners

- `random_mdp(S, A, gamma, k, R_type, R_scale, T_type, T_alpha, terminal_states, rng)` — samples a random tabular MDP
  - R_types: `gaussian`, `uniform`, `bernoulli`, `spike_slab`, `potential`, `goal`
  - T_types: `random`, `dirichlet`, `uniform`, `deterministic`
- `norm_baseline(ctrl_adv, one_step, scales)` — normalises baseline metrics via 1−exp(−x/scale)
- `run_pam_experiment(...)` — Q1/Q2/Q3 batch experiment (see `01_pam_baseline/`)
- `run_p_sweep(R_type, p_values, ...)` — sweeps reward sparsity p (see `02_sampling_sweeps/`)
- `run_gamma_sweep(gammas, R_conditions, ...)` — sweeps discount factor γ
- `run_t_sensitivity(n_R, R_conditions, ...)` — same R scored under multiple T structures
- `run_s_sweep(S_values, T_types, ...)` — sweeps state space size S

---

### Terminal vs absorbing states under shaping

Everything runs infinite-horizon discounted DP.

An **absorbing** state with R=0 (e.g. `gridworld` goal) naturally converges to V=0. Under shaping, `V'(s) = V(s) − Φ(s)` for all states — value differences cancel correctly. Marking a state as **terminal** (forced `V[s]=0`) pins the value regardless of shaping; this is safe only when metrics aggregate over non-terminal states exclusively, where Φ cancels in differences.

---

### Candidate proxies (not yet implemented)

**Agenticity gap (myopic vs planning):**

    AgenticityGap(R) = J(π*) − J(π*_myopic)

where `π*_myopic` acts greedily on Q* at each step without lookahead. Large gap means R structurally demands planning. Parameter-free; cleaner than the depth-h `planning_pressure` metric.

**Option value (future vs immediate):**

    E_s[ V*(s) − max_a R(s,a) ]

How much does the future matter relative to the best immediate reward? High option value = reward encodes long-range consequences that myopic optimisation cannot exploit.
