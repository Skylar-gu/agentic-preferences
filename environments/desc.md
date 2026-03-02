## README

### What this file is

A minimal “finite MDP lab” for defining and testing reward-based agency proxies that are invariant to potential-based reward shaping. It uses exact dynamic programming; no learning loops.

### Requirements

* Python 3.9+
* NumPy

### Core concepts

* **Reward** `R[s,a,s’]`: immediate payoff for transition `s --a--> s’`.
* **Value** `V(s)`: expected discounted sum of future rewards from state `s` under a policy.
* **Optimal value** `V*` and policy `π*`: computed by value iteration.
* **Potential shaping** adds `F(s,a,s’) = γΦ(s’) − Φ(s)` to rewards without changing optimal policies (in discounted finite MDPs).

### What it computes

#### 1) Optimal control objects

* `value_iteration(mdp)` returns `(V_star, Q_star, pi_star)`.

#### 2) Baseline evaluation

* `policy_evaluation(mdp, pi)` computes `V^pi`.

#### 3) Discounted occupancy (state visitation weighting)

* `discounted_occupancy(mdp, pi)` computes the discounted state distribution under a policy.  

### Metrics implemented

All metrics are designed so potential-shaping offsets cancel.

#### A) Control advantage

`control_advantage(mdp, pi_baseline)`

 A_ctrl = E_{s~d0}[ V*(s) - V^{pi0}(s) ]

Interpretation: benefit of optimal policy (control) over a passive baseline policy.

#### B) One-step recovery

`one_step_recovery(mdp, perturb_kernel)`

Perturb the state, then compare the reward from the next step under optimal vs baseline policy 

Interpretation: measures how much the difference in ability to get back on track after perturbation 

#### C) Planning pressure

`planning_pressure(mdp, h)`

Compute a depth-h lookahead policy `π_h` (optimal for h-step return), then evaluate its infinite-horizon performance:

[
P_h = \mathbb{E}_{s\sim d_0}[V^*(s) - V^{\pi_h}(s)]
]

Interpretation: how costly myopia is; larger means long-horizon consequences matter.

Implementation details:

* `finite_horizon_optimal_policy(mdp,h)` does backward induction to find `π_h`.
* Then `policy_evaluation` evaluates `π_h` on the full infinite-horizon MDP.

### Demo behavior

Running the file:

* Builds a 5×5 gridworld with slip.
* Computes `A_ctrl`, `R1`, and `P_h` for `h=3`.
* Applies random potential shaping and recomputes metrics.
* Prints differences; they should be near zero (floating point noise).

### How to extend

* Add new toy MDPs (chain MDP, traps) by constructing `T` and `R`.
* Add new metrics by ensuring they depend on **differences between policies** or **trajectory/occupancy comparisons**, not absolute value baselines.
* If you move to continuous control later, replace exact `V*` with a strong approximate policy/value estimator; keep the metric definitions unchanged.

### Expected pitfalls

* If `Differences` are not ~0 under shaping, the usual causes are:

  * shaping implemented incorrectly (wrong indices, missing γ),
  * perturbation kernel not row-normalized,
  * a metric using absolute `V(s)` rather than a difference that cancels `−Φ(s)`.
