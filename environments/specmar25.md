# Implementation Spec — New PAMs + Random MDP Generator
## File: `environments/mdp.py` (append to existing code)

---

## 1. Effective Time Horizon

### What it measures
The reward-weighted centre-of-mass across time steps under π*. Captures how far
into the future the reward function forces planning.

### Formula
H_eff = Σ_t  t · w_t    where    w_t = |E[r_t]| / Σ_{t'} |E[r_{t'}]|

r_t = R(s_t, a_t, s_{t+1}) along trajectories under π* from d0.

Computed via forward simulation: run π* for `horizon` steps, track discounted
reward mass at each timestep, compute weighted mean timestep.

### NOT shaping-invariant — flag explicitly
Potential shaping F(s,a,s') = γΦ(s') − Φ(s) adds a nonzero constant at every
step (even with constant Φ), spreading reward mass across all timesteps and
shifting H_eff. This must be documented in the docstring.

### Signature
```python
def effective_time_horizon(
    mdp: MDP,
    pi: np.ndarray,          # deterministic policy, shape (S,)
    horizon: int = 200,
    n_episodes: int = 1000,
    rng: np.random.Generator = None
) -> float:
```

### Implementation notes
- Roll out `n_episodes` trajectories from d0 under pi (deterministic, no epsilon)
- At each step t, accumulate |r_t| into a vector `mass[t]`
- H_eff = Σ_t (t * mass[t]) / Σ_t mass[t]
- If Σ_t mass[t] < 1e-10 (zero-reward MDP), return 0.0
- Use gamma-discounting on mass: mass[t] += gamma^t * |r_t|, so discounting
  is already baked in and the measure respects the agent's actual time preference
- Terminate episode early if terminal state reached
- Return scalar float

### Normalisation for composite score
H_eff is unbounded above. Normalise as: 1 / (1 + exp(-H_eff / scale)) where
scale is a hyperparameter (default: 10.0). Add normalised version alongside raw.

---

## 2. Entropy of MCE Policy

### What it measures
How concentrated the MCE (entropy-regularised) policy is. A reward function
with high advantage gap → near-deterministic π_MCE → low entropy. A flat
reward → uniform π_MCE → maximum entropy = log(A).

### Formula
H(π_MCE) = −Σ_s d0(s) Σ_a π_MCE(s,a) log π_MCE(s,a)

Weighted by d0, NOT d* (for consistency with control_advantage, planning_pressure).
Exclude terminal states from sum.

### Signature
```python
def mce_policy_entropy(
    mdp: MDP,
    alpha: float = 1.0,
    eps: float = 1e-12
) -> dict:
```

### Implementation notes
- Call soft_value_iteration(mdp, alpha) to get pi_mce of shape (S, A)
- Compute per-state entropy: H_s = -Σ_a π(s,a) * log(π(s,a) + eps)
- Weight by d0: H = Σ_s d0(s) * H_s, summing only over non-terminal states
  (renormalise d0 over non-terminal states before weighting)
- Return dict:
  {
    'entropy_raw': float,           # H(π_MCE) in nats
    'entropy_norm': float,          # H / log(A), in [0,1]; 1=uniform, 0=deterministic
    'alpha': alpha,
  }

### Note on alpha choice
Alpha is a free parameter. For the composite score, use alpha=1.0 as default.
The caller should be able to pass alpha. In the random MDP experiments, sweep
alpha in {0.1, 1.0, 10.0} and report all three — the sensitivity of entropy
to alpha is itself informative about the reward structure.

### Shaping invariance
H(π_MCE) IS invariant under all three reward equivalence transformations:

1. Positive scaling: scales Q by constant c, cancels inside softmax
2. Potential shaping: adds γΦ(s') − Φ(s); the Φ(s) term is action-independent
   so cancels inside softmax at each state s
3. S'-redistribution: δ(s,a,s') with Σ_{s'} T(s,a,s')δ(s,a,s')=0 for all
   (s,a). This preserves E_{s'}[R(s,a,s')] exactly, so Q^S(s,a) is unchanged,
   so π_MCE is unchanged.

Document all three in docstring.

---

## 3. Random MDP Generator

### Signature
```python
def random_mdp(
    S: int,
    A: int,
    gamma: float = 0.95,
    k: int = 3,                    # successor states per (s,a)
    R_type: str = 'gaussian',      # 'gaussian' | 'uniform' | 'bernoulli'
    R_scale: float = 1.0,          # std for gaussian, sparsity for bernoulli
    terminal_states: int = 1,      # number of terminal states (0 = none)
    rng: np.random.Generator = None
) -> MDP:
```

### Transition matrix T
- For each (s,a) pair independently:
  - If s is a terminal state: T[s,a,s] = 1.0 (absorbing), skip reward
  - Otherwise: sample k successor indices uniformly without replacement from
    the k non-terminal states (or all states if terminal_states=0)
  - Sample Dirichlet(α=1,...,1) weights of length k → assign to T[s,a,successors]
  - Result: each row T[s,a,:] sums to 1, at most k nonzero entries

### Reward matrix R (over S×A×S')
Three variants, each producing R of shape (S, A, S):

**'gaussian'**: R[s,a,s'] ~ N(0, R_scale) i.i.d. for all (s,a,s')
  - Set R[s,:,:] = 0 for terminal states s

**'uniform'**: R[s,a,s'] ~ Uniform(0, 1) i.i.d.
  - Set R[s,:,:] = 0 for terminal states s

**'bernoulli'**: R[s,a,s'] = 1 with probability R_scale, else 0
  (R_scale is the sparsity parameter; default R_scale=0.1 for sparse rewards)
  - Set R[s,:,:] = 0 for terminal states s

### Terminal states
- Pick the last `terminal_states` indices as terminal: {S-terminal_states, ..., S-1}
- Set T[s,a,s] = 1.0 for these states (absorbing)
- Set R[s,:,:] = 0.0 for these states

### Start distribution d0
- Uniform over non-terminal states only

### Returns
MDP dataclass instance (same as existing code uses throughout)

---

## 4. Integration into agenticity_score

Add `compute_mi: bool = True` parameter to `agenticity_score`. When False, skip
`early_action_mi` entirely and set mi fields to None in result dict.

```python
def agenticity_score(mdp, weights=None, horizon=60, n_episodes=2000,
                     verbose=True, rng=None, compute_mi=True) -> dict:
    ...
    if compute_mi:
        mi = early_action_mi(mdp, pi_star, horizon=horizon,
                             n_episodes=n_episodes, rng=rng)
    else:
        mi = {'mi_early': None, 'mi_late': None, 'mi_diff': None}

    # Composite: if MI excluded, renormalise remaining weights
    if compute_mi:
        composite = (weights['adv_gap'] * ag_norm +
                     weights['vstar_var'] * vv_norm +
                     weights['mi_diff'] * mi_norm)
    else:
        w_sum = weights['adv_gap'] + weights['vstar_var']
        composite = (weights['adv_gap'] * ag_norm +
                     weights['vstar_var'] * vv_norm) / w_sum
```

Add both new PAMs to result dict:
```python
result['h_eff_raw'] = round(h_eff, 4)
result['h_eff_norm'] = round(1 / (1 + np.exp(-h_eff / 10.0)), 4)
result['mce_entropy_raw'] = round(mce_ent['entropy_raw'], 4)
result['mce_entropy_norm'] = round(mce_ent['entropy_norm'], 4)

# Do NOT add to composite score yet — validate correlation with existing
# PAMs first in the random MDP experiments before deciding on weighting
```

---

## 5. Batch experiment function (for Joar's Q1/Q2/Q3)

```python
def run_pam_experiment(
    n_random_mdps: int = 200,
    S_values: list = [5, 10, 20],
    A: int = 4,
    gamma: float = 0.95,
    k: int = 3,
    R_types: list = ['gaussian', 'uniform', 'bernoulli'],
    n_fixed_T: int = 10,           # for Q2: fix T, vary R
    rng_seed: int = 42,
    verbose: bool = False,
) -> dict:
```

### What it runs

**Q1 — PAM correlation + base rate (R varies, T fixed):**
For each S in S_values, for each R_type:
- Sample one canonical T per (S, seed=0) and hold it fixed
- Sample n_random_mdps random R's, compute all PAMs for each
- Return matrix of shape (n_random_mdps, n_PAMs)
- Record fraction with composite > 0.5 ("high agenticity" base rate)
- Record Pearson and Spearman correlation matrix across PAMs

This isolates variance attributable purely to R, with T held constant.

**Q2 — T-dependence (variance decomposition):**
For each S in S_values, for each R_type:
- Sample n_fixed_T independent transition functions
- For each T, sample n_R random R's (default n_R=50), compute composite PAM score
- Compute:
    - within_var  = mean over T's of Var(composite | T)   [variance due to R]
    - between_var = Var(mean(composite | T)) over T's     [variance due to T]
    - ratio = between_var / (within_var + between_var)    [T's share of total variance]
- If ratio >> 0.5: T dominates, agenticity is not intrinsic to R
- If ratio << 0.5: R dominates, consistent with environment-invariant definition

This is a one-way ANOVA with T as grouping factor, R as within-group factor.
Return within_var, between_var, ratio for each (S, R_type) cell.

**Q3 — Human-made vs random:**
Run agenticity_score on the 7 existing MDPs (chain and grid variants).
Return their PAM vectors alongside the random MDP distributions for comparison.

### Returns
```python
{
  'q1': {
    # key: (S, R_type), value: (n_mdps x n_PAMs) array
    (S, R_type): np.ndarray,
    ...
  },
  'q2': {
    # key: T_index, value: dict with mean/std/per_R_type composite distributions
    t_idx: {'mean': float, 'std': float, 'composites': np.ndarray},
    ...
  },
  'q3': {
    # PAM vectors for the 7 human-made MDPs
    mdp_name: dict,  # same format as agenticity_score output
    ...
  },
  'meta': {
    'S_values': S_values,
    'n_random_mdps': n_random_mdps,
    'PAM_names': ['adv_gap', 'vstar_var', 'mi_diff', 'h_eff', 'mce_entropy'],
  }
}
```

### Performance note
early_action_mi is slow. run_pam_experiment passes compute_mi=False to
agenticity_score by default. Add a separate `include_mi: bool = False`
parameter to run_pam_experiment. When True, reduces n_episodes to 300 inside
the batch run and flags noisier estimates in the returned metadata.

---

## Summary of new functions to add

| Function | Location | Depends on |
|---|---|---|
| `effective_time_horizon(mdp, pi, horizon, n_episodes, rng)` | mdp.py | value_iteration |
| `mce_policy_entropy(mdp, alpha, eps)` | mdp.py | soft_value_iteration |
| `random_mdp(S, A, gamma, k, R_type, R_scale, terminal_states, rng)` | mdp.py | MDP dataclass |
| `run_pam_experiment(...)` | mdp.py or new experiments.py | all above |

---

## Implementation Status (updated March 2026)

### Completed

**`random_mdp()` extensions (experiments.py)**
- `T_type` parameter: `'random'` (Dirichlet(1) over k successors, default), `'uniform'` (1/S to all states), `'dirichlet'` (Dirichlet(T_alpha) over k successors, concentrated at low alpha), `'deterministic'` (each (s,a) → one fixed s')
- `T_alpha` parameter: concentration for Dirichlet T (default 0.1 = highly concentrated)
- `R_type='spike_slab'`: mask × N(0,1), where mask ~ Bernoulli(R_scale). R_scale = p = sparsity probability

**New experiment functions (experiments.py)**
- `run_p_sweep(R_type, p_values, ...)`: Group 1 — fix T, sweep R_scale=p. Returns `{p: [score dicts]}`
- `run_gamma_sweep(gammas, R_conditions, ...)`: Group 3a — sweep γ per (R_type, R_scale). Returns `{(R_type, R_scale, gamma): [score dicts]}`
- `run_t_sensitivity(n_R, R_conditions, T_types, ...)`: Group 2 — for each R, score under multiple T structures. Returns `{(R_type, R_scale): {T_type: [score dicts]}}` where index i is the same R_i across T_types
- `run_s_sweep(S_values, T_types, ...)`: Group 3b — sweep S × T_type. Returns `{(S, T_type): [score dicts]}`

**New plot script: `plot_sampling.py`**
- Plot 09: Bernoulli p sweep (spike-and-slab-const)
- Plot 10: spike-and-slab p sweep (mask × N(0,1))
- Plot 11: γ sweep for spike_slab and gaussian R
- Plot 12: T-sensitivity scatter (Uniform vs Deterministic T, same R)
- Plot 13: S sweep per T type

### Not yet implemented

- `R_type='uniform_simplex'`: Dirichlet(1,...,1) over all (s,a,s') triples; needed for Turner comparison. Deferred — structurally distinct from all other R_types.
- Fraction-agentic estimator with bootstrap CIs (§5.3): deferred until composite weights are calibrated from correlation analysis.

### Open design questions carried forward

- The `'spike_slab'` R_type hardcodes σ=1.0 for non-zero entries. Varying σ independently of p requires a separate parameter.
- T-sensitivity experiment generates a fresh random T for each (R_i, T_type) evaluation. This tests the distribution of scores per T_type but not the exact same T across types. Acceptable for distribution comparison; use a fixed T seed per R_i if exact pairing is needed.
