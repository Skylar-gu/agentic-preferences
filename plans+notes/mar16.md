# Environments + Metrics Reference

Finite MDP testbed for measuring agenticity — how much an agent's planning actually matters. All computation is exact dynamic programming (no RL training); metrics are designed to be invariant to potential-based reward shaping.

Reward functions across environments: 
  gridworld() (5×5, slip=0.1)                                                                                                                                                                                             
  ┌──────────────────┬─────────────────────────────────────────────────────────────┐                                                                                                  
  │       Name       │                         R(s, a, s')                         │                                                                                                  
  ├──────────────────┼─────────────────────────────────────────────────────────────┤
  │ Step-cost + goal │ −1 on every transition; 0 at goal (absorbing, not terminal) │
  └──────────────────┴─────────────────────────────────────────────────────────────┘

  Used only for shaping invariance verification, not agenticity benchmarking.

  ---
  make_chain_mdp() (10-state linear chain, 2 actions: stay/advance)

  ┌──────────┬───────────────────────────────────────────────────────────────────────┐
  │   Name   │                              R(s, a, s')                              │
  ├──────────┼───────────────────────────────────────────────────────────────────────┤
  │ Terminal │ +1 only when landing on last state (s=9)                              │
  ├──────────┼───────────────────────────────────────────────────────────────────────┤
  │ Dense    │ s' / 9 — reward proportional to destination index, every step         │
  ├──────────┼───────────────────────────────────────────────────────────────────────┤
  │ Lottery  │ Uniform(0.5, 1.5) drawn once at construction, paid only on last state │
  ├──────────┼───────────────────────────────────────────────────────────────────────┤
  │ Progress │ +0.1 for advancing (action 1), plus +1 on reaching last state         │
  └──────────┴───────────────────────────────────────────────────────────────────────┘

  ---
  make_grid_mdp() (5×5 grid, 4 actions: UDLR)

  ┌───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Name  │                                                      R(s, a, s')                                                       │
  ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Goal  │ +1 only when landing on goal (bottom-right corner)                                                                     │
  ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Local │ −dist(s, goal) / (rows+cols) — negative Manhattan distance, paid from the source state; rewards proximity continuously │
  ├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Cliff │ +1 at goal, −1 for landing on any cliff state (top row, columns 1–3)                                                   │
  └───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  The file has three parts:

  1. Core DP utilities — value iteration, policy evaluation, discounted occupancy, finite-horizon lookahead, potential shaping transform.

  2. Shaping-invariant metrics (Baseline metrics) — three scalar summaries of an MDP's decision structure, verified to be numerically invariant under reward shaping R' = R + γΦ(s') − Φ(s):

  ┌────────────────────────────┬───────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │           Metric           │                      Formula                      │                            What it captures                            │
  ├────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Control Advantage (A_ctrl) │ E_{s~d0}[V*(s) − V^{π0}(s)]                       │ How much better optimal is vs. a baseline policy                       │
  ├────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ One-Step Recovery (R1)     │ E_{s~d*} E_{a~Unif} E_{s'~T}[V*(s') − V^{π0}(s')] │ Recovery value after taking a random action from the optimal occupancy │
  ├────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Planning Pressure (P_h)    │ E_{s~d0}[V*(s) − V^{πh}(s)]                       │ Cost of using a depth-limited lookahead instead of full planning       │
  └────────────────────────────┴───────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  3. Agenticity proxies (PAM proxies) — four normalized proxies benchmarked across 7 MDPs, combined into a composite score:

  ┌────────────────────┬─────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │       Proxy        │     Weight      │                                    What it captures                                    │
  ├────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ Advantage Gap      │ 0.25            │ Mean spread between best and worst actions per state                                   │
  ├────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ V* Variance        │ 0.40            │ Variance of V* − V^rand; how much state values differ under optimal vs. random         │
  ├────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ MI Differential    │ 0.35            │ I(early actions; return | s₀) − I(late actions; return | s₀); measures irreversibility │
  ├────────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ Advantage Sparsity │ diagnostic only │ Fraction of near-zero advantages                                                       │
  └────────────────────┴─────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  4. MCE (Maximum Causal Entropy) — entropy-regularised Bellman equations with temperature α. As α → 0 the policy recovers the hard-optimal; as α → ∞ it becomes uniform.

  ---
  Shaping Invariance Check (5×5 gridworld, slip=0.1)

  All three Baseline metrics metrics are numerically invariant under a random potential Φ (differences are floating-point noise):

  ┌────────┬────────┬─────────────────┐
  │ Metric │ Value  │ Δ under shaping │
  ├────────┼────────┼─────────────────┤
  │ A_ctrl │ 40.397 │ 4.7×10⁻⁹        │
  ├────────┼────────┼─────────────────┤
  │ R1     │ 2.961  │ 3.2×10⁻¹⁰       │
  ├────────┼────────┼─────────────────┤
  │ P_h(3) │ 0.000  │ −1.1×10⁻¹⁴      │
  └────────┴────────┴─────────────────┘

  ---
  Agenticity Proxy Results (7 MDPs)

  Composite score ∈ [0, 1]; higher = planning matters more.

  ┌──────────────────┬───────────┬─────────────────┬───────────────┬─────────┬────────────────────────────────────────────────────────────────────────┐
  │       MDP        │ Composite │ Adv. Gap (norm) │ V* Var (norm) │ MI Diff │                                 Notes                                  │
  ├──────────────────┼───────────┼─────────────────┼───────────────┼─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Chain — Dense    │ 0.901     │ 1.000           │ 0.878         │ +0.714  │ High-agenticity baseline; dense rewards make every step count          │
  ├──────────────────┼───────────┼─────────────────┼───────────────┼─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Grid — Cliff     │ 0.840     │ 0.887           │ 0.670         │ +1.720  │ Highest MI: irreversible early mistakes dominate trajectory            │
  ├──────────────────┼───────────┼─────────────────┼───────────────┼─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Grid — Local     │ 0.817     │ 0.979           │ 0.662         │ +0.754  │ Dense local rewards on grid; high advantage gap                        │
  ├──────────────────┼───────────┼─────────────────┼───────────────┼─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Chain — Progress │ 0.695     │ 0.578           │ 0.501         │ +1.001  │ Per-step progress rewards raise MI above terminal/lottery              │
  ├──────────────────┼───────────┼─────────────────┼───────────────┼─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Grid — Goal      │ 0.676     │ 0.601           │ 0.501         │ +0.858  │ Sparse goal reward; grid topology provides MI signal                   │
  ├──────────────────┼───────────┼─────────────────┼───────────────┼─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Chain — Lottery  │ 0.622     │ 0.558           │ 0.501         │ +0.610  │ Low agenticity; chain topology too simple to distinguish from terminal │
  ├──────────────────┼───────────┼─────────────────┼───────────────┼─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Chain — Terminal │ 0.619     │ 0.551           │ 0.501         │ +0.607  │ Lowest agenticity; agent's choices barely affect return                │
  └──────────────────┴───────────┴─────────────────┴───────────────┴─────────┴────────────────────────────────────────────────────────────────────────┘

  Key observations:
  - MI differential is the most sensitive proxy for irreversibility (Cliff >> all others by 2×).
  - Advantage gap best captures dense-reward agenticity (Dense chain scores maximum).
  - Chain MDPs cluster together and cannot be separated by advantage gap or V* variance alone — branching topology needed to distinguish lottery vs. terminal reward structures.

  - Reasons for picking a dense reward function as the high agenticity baseline: intermediate actions matter not just the final result, every step has a clear best action w states all having different values 

  ---
  MCE Temperature Sweep 

  ┌────────┬───────────────────┬───────────────────┐
  │        │   Chain — Dense   │   Grid — Cliff    │
  ├────────┼───────────────────┼───────────────────┤
  │ α=0.01 │ J=16.81, L1=0.100 │ J=0.87, L1=0.611  │
  ├────────┼───────────────────┼───────────────────┤
  │ α=0.1  │ J=16.86, L1=0.201 │ J=2.63, L1=1.332  │
  ├────────┼───────────────────┼───────────────────┤
  │ α=1    │ J=20.45, L1=0.966 │ J=26.52, L1=1.413 │
  ├────────┼───────────────────┼───────────────────┤
  │ α=10   │ J=75.60, L1=1.227 │ J=272.1, L1=1.548 │
  └────────┴───────────────────┴───────────────────┘

