## Finite MDP Lab

Exact dynamic programming on tabular MDPs for measuring agenticity proxies. No learning loops. All metrics invariant under potential-based reward shaping (`R' = R + γΦ(s') − Φ(s)`).

### Requirements

* Python 3.9+, NumPy, SciPy

### MDP class

`MDP(S, A, T, R, gamma, terminal, d0)` — dataclass holding transition matrix `T[s,a,s']`, reward matrix `R[s,a,s']`, discount `gamma`, set of absorbing `terminal` states, and start distribution `d0` (defaults to uniform).

### Core DP

* `value_iteration(mdp)` → `(V*, Q*, pi*)`
* `policy_evaluation(mdp, pi)` → `V^pi` (deterministic or stochastic `pi`)
* `value_under_random_policy(mdp)` → `V^rand`
* `discounted_occupancy(mdp, pi)` → discounted state visitation `d^pi`
* `finite_horizon_optimal_policy(mdp, h)` → backward induction for truncated horizon
* `finite_horizon_lookahead_policy(mdp, h, V_terminal)` → depth-h lookahead with value bootstrap

### MCE (Maximum Causal Entropy)

* `soft_value_iteration(mdp, alpha)` → `(V_soft, Q_soft, pi_mce)` via soft Bellman `V^S = α·logsumexp(Q^S/α)`
* `mce_policy(Q_soft, alpha)` → `softmax(Q^S/α)` stochastic policy
* `mce_objective(mdp, Q_soft, alpha)` → `mean_s V^S(s)` over non-terminal states

### Potential shaping

`add_potential_shaping(mdp, Phi)` → new MDP with `R' = R + γΦ(s') − Φ(s)`

### Week 2 metrics (use d0)

| Metric | Function | What it measures |
|---|---|---|
| Control advantage | `control_advantage(mdp, pi0)` | `E_{d0}[V* − V^{pi0}]` — benefit of optimal over baseline |
| One-step recovery | `one_step_recovery(mdp, Ppert, pi0)` | Recovery after state perturbation (`V* − V^{pi0}` under perturbed distribution) |
| Planning pressure | `planning_pressure(mdp, h)` | `E_{d0}[V* − V^{pi_h}]` — cost of depth-h myopia |

### Week 3 proxies (agenticity)

| Proxy | Function | What it measures |
|---|---|---|
| Advantage gap | `advantage_gap(mdp, Q, V)` | Mean action-value spread per state (`max A* − min A*`) |
| V\*−V^rand variance | `vstar_variance_corrected(mdp, V*, V_rand)` | Reward-induced landscape beyond geometry |
| Early-action MI | `early_action_mi(mdp, pi, ...)` | `I(A_{1:k}; G \| s_0) − I(A_{k+1:T}; G \| s_0)` — early-action importance |
| Advantage sparsity | `advantage_sparsity(mdp, Q, V)` | Fraction of near-zero advantages (diagnostic) |
| Composite | `agenticity_score(mdp)` | Weighted sum (AdvGap=0.25, V\*var=0.40, MI=0.35) |

### Environments

* `gridworld(w, h, goal_xy, step_cost, goal_reward, slip, gamma)` — 4-action grid, absorbing goal, optional slip (week 2; goal NOT marked terminal for shaping compatibility)
* `make_chain_mdp(n, gamma, reward_type)` — linear chain (terminal/dense/lottery/progress)
* `make_grid_mdp(rows, cols, gamma, reward_type)` — 5×5 grid (goal/local/cliff)

### Demo

`python environments/mdp.py` runs all three sections:

1. **Week 2** — shaping invariance verification on 5×5 gridworld with slip; prints metric deltas (~0)
2. **Week 3** — agenticity proxy benchmark across 7 MDPs; prints composite ranking
3. **MCE** — soft Bellman on Grid-Cliff at α ∈ {0.01, 0.1, 1, 10}; shows convergence to hard optimal as α→0

### Terminal vs absorbing states under shaping

Everything runs infinite-horizon discounted DP. The distinction isn't finite vs infinite horizon — it's whether metrics touch terminal-state values directly.

An **absorbing** state with R=0 (e.g. `gridworld` goal) naturally gives V=0 via the Bellman fixed point. Under shaping, `R'[goal,:,goal] = (γ−1)Φ(goal)`, so V converges to `−Φ(goal)`. This is correct: Ng et al. guarantees `V'(s) = V(s) − Φ(s)` for all states including absorbing ones.

Marking a state as **terminal** (explicit `V[s]=0` forcing) pins the value at 0 regardless of shaping, breaking the `−Φ(s)` shift. This matters for week 2 metrics because `d0 @ (V* − V^pi)` sums over *all* states — if the occupancy reaches the goal, the forced zero corrupts the difference.

Week 3 envs can use explicit `terminal` because their proxies (`A* = Q* − V*`, `V* − V^rand`) only aggregate over non-terminal states, where the Phi shift cancels in the difference. Terminal states get excluded from every computation.

### Candidate proxies (not yet implemented)

**Agenticity gap (myopic vs planning)**. Probably the cleanest handle: a reward function that *requires* long-horizon reasoning to optimize — where greedy/myopic policies perform significantly worse than π\* — is a meaningful intrinsic property of R. Connects directly to the Early-Action MI metric. Formalize as:

    AgenticityGap(R) = J(π*) − J(π*_myopic)

where `π*_myopic = argmax_a Q*(s,a)` at each step without lookahead. Large gap means R structurally demands planning. (Note: this is close to `planning_pressure` but uses a 1-step greedy policy rather than a depth-h lookahead — simpler and parameter-free.)

**Option value (future vs immediate)**. More intrinsic to R, less dependent on T:

    E_s[ V*(s) − max_a R(s,a) ]

How much does the *future* matter relative to the *immediate* best reward? If large, R is structurally future-oriented regardless of the transition dynamics. High option value means the reward function encodes long-range consequences that can't be captured by myopic optimization.

### Other design notes

* Week 2 metrics cancel shaping via value *differences* (`V* − V^pi`), not absolute values.
* Week 3 proxies are built on `A*(s,a) = Q* − V*` or `V* − V^rand`, both shaping-invariant.
* `early_action_mi` uses ε-greedy (ε=0.3) sampling; needs `n_episodes ≥ 500` for reliable estimates.
