# Research Note: Random Reward Sampling & Agenticity Distribution

**SPAR Project — Defining Agentic Preferences | Draft, March 2026**

---

## 1. Motivation and Setup

Current experiments benchmark fixed reward types (terminal, dense, cliff, etc.) against our proxy suite. This is necessary but insufficient: it tests whether PAMs agree with intuition on hand-crafted examples, not whether they are well-calibrated across the full space of possible reward functions. The natural next step is to sample reward functions from parametric distributions, compute PAM scores, and analyse the resulting distribution.

This note formalises the experimental design, derives theoretical predictions about what to expect (so that deviations are informative), and identifies the asymptotic questions most likely to clarify or undermine our definitions.

---

## 2. Distributional Priors on R

Three priors are used, chosen for distinct conceptual justifications rather than mathematical convenience. Each answers a different version of "what is the natural distribution over reward functions?"

### 2.1 Gaussian Prior — pre-training analogy

```
R ~ N(0, σ², size=(S, A, S′))   — entries sampled i.i.d.
```

**Justification:** a neural reward model r_θ(s,a,s′) with random initialisation produces outputs that are approximately Gaussian by the CLT (sum of many random weights). The Gaussian prior therefore models the pre-RLHF-training distribution — what reward functions look like before any conditioning on preference data. This is the natural baseline for asking "what does an unconstrained reward model look like?"

Two important properties:
- Dense by construction — every transition receives a non-zero signal almost surely. This will bias PAM scores upward relative to human-written reward functions, which are sparse.
- Symmetric around zero — no preferred direction of travel. Advantage gap and V* variance will be inflated by σ but MI differential should correctly score near zero (no irreversibility in a random dense reward).

> ⚠ The Gaussian prior models the initialisation distribution, not the posterior after RLHF fine-tuning. Whether agenticity at initialisation predicts agenticity post-training is itself an empirical question worth flagging for future work.

### 2.2 Spike-and-Slab Prior — realistic human design

```
R(s,a,s′) = mask(p) * N(0, σ²),   mask ~ Bernoulli(p)
```

**Justification:** human-designed reward functions are sparse — engineers assign non-zero signal at specific transitions (goal reached, cliff fallen into, task completed) and leave everything else at zero. The spike-and-slab prior models this directly: with probability 1−p the reward is exactly zero; with probability p it is drawn from N(0,σ²).

Critical distinction — three parameters are independent:
- **p**: sparsity (fraction of non-zero entries). Controls how rare the reward signal is.
- **σ**: magnitude of non-zero rewards. Controls signal strength, independent of sparsity.
- **Heterogeneity**: variance of the non-zero distribution. With N(0,σ²) non-zero values, rewarded transitions differ in quality — the agent must discriminate good from bad rewarded states. Replacing N(0,σ²) with a constant collapses heterogeneity to zero.

Do NOT conflate p and magnitude. A sparse reward with constant non-zero value (Bernoulli × const) is a lottery — the agent cannot plan to reach a better rewarded state because all rewarded states are identical. Spike-and-slab with varying magnitudes is structurally richer and more realistic.

> ⚠ Small p directly instantiates the lottery counterexample. MI differential should correctly score this as low-agenticity even when advantage gap scores it as high (because the single rewarded transition creates a large Q-value spread). This cross-metric divergence is the most important thing to detect in this experiment.

### 2.3 Uniform Simplex Prior — Turner comparability

```
R ~ Dirichlet(1, 1, ..., 1)  over the S×A×S′ simplex
```

**Justification:** Turner et al. (2021) derive their power-seeking results using a uniform prior over the reward simplex — every normalised reward vector is equally likely. Using the same prior allows direct comparison of our fraction-agentic estimate to Turner's theoretical prediction that "most reward functions make power-seeking optimal."

Properties: non-negative by construction; sums to 1 across all (s,a,s′). Mathematically clean but empirically unrealistic — real reward functions are not probability distributions over transitions. Use this prior only for the Turner comparison.

> ⚠ Any claim about "fraction of reward functions that are agentic" is prior-relative. A discrepancy between our fraction-agentic number and Turner's theoretical prediction is not an error — it is a research finding, because the priors differ. Explaining the discrepancy is itself a contribution.

### 2.4 Experimental Grid

T structure is an independent variable in all experiments (see §6.2 for justification). Each cell runs N = 200 sampled reward functions × 3 T instances = 600 PAM evaluations.

| Prior | Key parameter | T instances | Primary question |
|---|---|---|---|
| Gaussian | σ ∈ {0.1, 0.5, 1, 5} | Uniform / Dirichlet(0.1) / Deterministic | PAM scaling with σ; T-sensitivity of scores |
| Spike-and-slab | p ∈ {0.01, 0.05, 0.2, 0.5, 1.0} | Uniform / Dirichlet(0.1) / Deterministic | Lottery detection; sparsity-agenticity independence |
| Spike-and-slab × const | p ∈ {0.01, 0.1, 0.5} | Uniform / Dirichlet(0.1) / Deterministic | Isolate heterogeneity: zero vs non-zero |
| Uniform simplex | — (single condition) | Uniform / Dirichlet(0.1) / Deterministic | Turner fraction-agentic comparison |

---

## 3. Theoretical Predictions

### 3.1 Advantage Gap Under Gaussian R

Fix the transition matrix T. For a given state s, the Q-value satisfies:

```
Q(s, a) = Σ_{s′} T(s,a,s′) [R(s,a,s′) + γ V*(s′)]
```

With R entries i.i.d. N(0, σ²) and T fixed:

```
Q(s, a) ~ N(γ Σ_{s′} T(s,a,s′) V*(s′),  σ² Σ_{s′} T(s,a,s′)²)
```

The variance term is σ² ‖T(s,a,·)‖², which is bounded above by σ² (equality iff T is deterministic) and equals σ²/S for uniform T. The expected advantage gap for A actions scales as:

```
E[AdvGap] ≈ 2σ ‖T(s,a,·)‖ · √(2 ln A)
```

Two consequences:
- Advantage gap is proportional to σ — a pure scale parameter. Normalise by σ when comparing across distributions.
- Advantage gap shrinks with S under uniform T, since ‖T(s,a,·)‖² = 1/S → 0. In a large flat environment, any single action's expected reward contribution is washed out by averaging over many successor states.

> ⚠ Gaussian rewards in large MDPs will have LOW advantage gap scores — not because they are non-agentic, but because environmental averaging suppresses Q-value spread. V* variance should be checked for consistency.

### 3.2 V* Variance Under Gaussian R

V* = (I − γ P_π*)⁻¹ r_π* is a linear transform of the reward vector. For R i.i.d. Gaussian, V*(s) is a weighted sum of Gaussian entries. By the CLT, as S → ∞ with uniform T:

```
V*(s) →_d N(0, σ² · Var_weight)
```

where the variance weight depends on T and γ. Under uniform T, all states have the same distribution of V*(s), so Var(V*) → 0. The same averaging phenomenon as §3.1.

> ✓ Both advantage gap and V* variance going to zero under Gaussian sampling with large S is a CORRECT mathematical prediction, not a metric failure. A random dense symmetric reward has no structural directionality — it is genuinely non-agentic. The test is whether sparser or more structured distributions score higher.

### 3.3 MI Differential Under Gaussian R

Under Gaussian R with no terminal structure, the return G is a sum of all rewards along a trajectory. Early and late actions contribute roughly equally to G (no irreversibility), so the differential should be near zero.

MI differential is therefore the most discriminating PAM for this experiment: it should correctly score Gaussian rewards as low-agenticity (differential ≈ 0) even when advantage gap may be non-trivially positive. Significant MI differential requires that early choices are more consequential than late ones — which requires structured, not random, reward.

### 3.4 Asymptotic Behaviour as γ → 1

As γ → 1, the effective planning horizon H_eff ≈ 1/(1−γ) → ∞. Three effects compound:

- Value functions diverge for any non-zero average reward. Must switch to average-reward formulation or work with differential values.
- The advantage gap between optimal and suboptimal actions grows (bad decisions compound over longer horizons), so agentic rewards score higher.
- For zero-mean rewards (e.g. symmetric Gaussian), divergence is avoided, but V* still accumulates more signal from distant states.

Theoretical prediction: agenticity scores should increase monotonically with γ, with the rate of increase characterising how long-range the planning requirement is. A reward requiring k-step planning should show a knee in the score-vs-γ curve near γ ≈ 1 − 1/k.

> ⚠ As γ → 1, value iteration convergence slows to O(1/(1−γ)) iterations. Use exact matrix inversion for policy evaluation. Also re-verify S′-redistribution invariance at each γ — shaping offsets scale as γΦ(s′) − Φ(s) and become more significant near γ = 1.

### 3.5 Asymptotic Behaviour as S → ∞

Three regimes depending on T structure:

| T structure | Advantage gap | V* variance | MI differential |
|---|---|---|---|
| Uniform (1/S each) | → 0 (averaging) | → 0 (CLT) | ≈ 0 (no structure) |
| Sparse (k-NN, k fixed) | Bounded | Bounded | Depends on graph topology |
| Deterministic path | Bounded | Bounded | Potentially high |

The sparse-T case is most realistic for AI systems. PAM scores should stabilise as S → ∞ for sparse T but collapse for dense T. Empirically: vary T structure via the Dirichlet concentration parameter, independently of S.

---

## 4. Correlation Analysis Between PAMs

### 4.1 Predicted Correlation Structure

| PAM pair | Predicted correlation | Reason |
|---|---|---|
| Advantage gap ↔ V* variance | Positive (moderate) | Both scale with reward heterogeneity |
| Advantage gap ↔ MI differential | Weak (near zero) | Gap = per-state; MI = path-level irreversibility |
| V* variance ↔ MI differential | Weak to moderate | V* variance can reflect terminal structure; MI captures it directly |
| All three ↔ sparsity | Negative | Dense rewards inflate gap and variance; MI is sparsity-independent |

If advantage gap and V* variance are highly correlated (r > 0.85), one is redundant — drop from composite or merge with equal weight. If MI differential is uncorrelated with both, it is measuring a genuinely independent dimension of agenticity (irreversibility), which justifies its separate weight.

### 4.2 Correlation Matrix

Compute per distributional condition (N ≥ 200 samples each):

|  | Adv. Gap | V* Variance | MI Diff |
|---|---|---|---|
| **Adv. Gap** | 1.00 | r₁₂ = ? | r₁₃ = ? |
| **V* Variance** | r₁₂ | 1.00 | r₂₃ = ? |
| **MI Diff** | r₁₃ | r₂₃ | 1.00 |

Interpretation key:
- **r₁₂ > 0.85** → advantage gap and V* variance are redundant; drop one or merge with equal weight.
- **r₁₃, r₂₃ ≈ 0** → MI differential is measuring an independent dimension (irreversibility); its separate weight in the composite is justified.
- **r₁₃, r₂₃ ≈ r₁₂** → all three collapse onto one axis; scalar composite is well-founded (agenticity g-factor).

Report this matrix separately for each (prior × T_type) condition. If the correlation structure changes substantially across conditions — e.g. MI is independent under Gaussian R but correlated under spike-and-slab — that is a finding: the PAMs measure different things depending on reward structure, which undermines the composite score as a universal summary.

---

## 5. Implementation Specification

### 5.1 Random Reward Generator (extend existing)

Extend `make_random_mdp()` to accept:
- `reward_dist`: `'gaussian'` | `'spike_slab'` | `'spike_slab_const'` | `'uniform_simplex'`
- `reward_scale`: σ for Gaussian; (p, σ) for spike-and-slab; ignored for simplex
- `transition_type`: `'uniform'` | `'dirichlet_alpha'` | `'deterministic'`

For each `(dist, scale, T_type)` combination, sample N = 200 reward functions. Compute all three PAMs (`compute_mi=True`; batch overnight). Store results as a DataFrame with columns:

```
[dist, scale, T_type, S, gamma, adv_gap_norm, vstar_var_norm, mi_diff_norm, composite]
```

### 5.2 Plotting

- Histogram of composite scores per distribution type — do priors produce distinguishable distributions?
- Scatter plots: advantage gap vs V* variance, advantage gap vs MI diff, V* variance vs MI diff — colour-coded by distribution type
- Score-vs-σ (Gaussian) and score-vs-p (spike-and-slab) curves per PAM — check monotonicity
- Score-vs-γ curves at fixed R distribution — expect monotone increasing for structured rewards
- Score-vs-S curves at fixed T structure — collapse predicted for uniform T

### 5.3 Fraction-Agentic Estimator

Joar asks: "what fraction of reward functions are agentic?" Define:

```
Frac_agentic(prior) = P_{R ~ prior}(composite_score(R) > threshold)
```

where threshold = 0.6 (tentative). Compute for each prior with 95% bootstrap confidence intervals. Compare to Turner et al.'s prediction that "most reward functions make power-seeking optimal."

> ⚠ Turner's result is measure-theoretic over a specific simplex with graph-symmetry conditions. Our fraction-agentic number is prior-dependent. The two numbers need not agree — explaining the discrepancy is itself a contribution.

---

## 6. Open Questions and Follow-Up

### 6.1 The Right Prior

None of Gaussian, spike-and-slab, or uniform simplex are "the" prior over reward functions. The scientifically meaningful prior is over human-written reward functions and more of these need to be pooled and studied in the future.

### 6.2 Environment Invariance: Why T Is an Independent Variable

PAMs compute `Q(s,a) = Σ_{s′} T(s,a,s′)[R(s,a,s′) + γV*(s′)]`. T acts as a smoothing kernel over R: under uniform T, `Q(s,a) ≈ (1/S)Σ_{s′}R + const`, so all actions look the same regardless of R structure. Under deterministic T, `Q(s,a) = R(s,a,s′(s,a)) + γV*(s′(s,a))`, preserving the full reward signal. The same sampled R can therefore score high or low agenticity depending purely on T — meaning fixing T conflates (R,T) properties with R-intrinsic properties.

Protocol: for every sampled R, run PAMs against three T instances:
- **Uniform T** — T(s,a,s′) = 1/S for all s′. Maximally smoothing; lower bound on agenticity.
- **Sparse Dirichlet T** — rows drawn from Dirichlet(α=0.1), concentrated over few successor states. Realistic for most RL environments.
- **Deterministic T** — each (s,a) maps to exactly one s′. No smoothing; upper bound on agenticity for a given R.

Report: (1) PAM scores under each T; (2) max score across T instances as the existential-quantifier agenticity estimate; (3) variance across T instances as a T-sensitivity diagnostic. High T-sensitivity means the PAM is measuring an (R,T) property, not R alone — a violation of the environment-invariance desideratum.

### 6.3 Compositionality

Is the agenticity of R₁ + R₂ a function of the agenticity of R₁ and R₂ separately? Clean test: sample R₁ (high agenticity) and R₂ (low agenticity), compute R = αR₁ + (1−α)R₂ for α ∈ [0,1], and plot composite score vs α. If linear: agenticity is compositionally additive. If nonlinear: characterise the failure mode. (Directly relevant to the InstructGPT KL penalty question in `pipeline.md`.)

### 6.4 Potential Follow-Up with MACHIAVELLI

Pan et al. (2023) provide 134 text-based environments with ground-truth power-seeking annotations. Once the sampling experiment establishes a well-calibrated threshold, apply PAMs to MACHIAVELLI reward functions and check whether the threshold correctly separates power-seeking from non-power-seeking environments. This is the external validation step the corpus analysis alone cannot provide.

---

*Status: Draft for discussion with Joar. Theoretical predictions in §3 are confident; implementation spec in §5 is actionable. Fraction-agentic estimator (§5.3) should be treated as exploratory pending threshold calibration.*


Summary of experiments to run: 
Group 1: PAM calibration across priors
Fixed: S=20, A=4, γ=0.95, T=Dirichlet(0.1). Vary reward distribution.

Sample N=200 R per condition: Gaussian (σ=1), spike-and-slab (p ∈ {0.01, 0.05, 0.2, 0.5, 1.0}), spike-and-slab-const (p ∈ {0.01, 0.1, 0.5}), uniform simplex
Compute all three PAMs per sample
Output: histogram of composite scores per prior; 3×3 correlation matrix per prior

Group 2: T-sensitivity
Fixed: S=20, A=4, γ=0.95. For each of 100 sampled R (Gaussian σ=1 and spike-and-slab p=0.05):

Run all three PAMs under Uniform T, Dirichlet(0.1) T, Deterministic T
Compute: per-T scores, max across T (existential estimate), variance across T (T-sensitivity)
Output: scatter plot of (score under Uniform T) vs (score under Deterministic T) for each PAM

Group 3: Asymptotics
Two sub-experiments, run independently.
3a. γ sweep. Fixed: S=20, A=4, T=Dirichlet(0.1), R=spike-and-slab p=0.1. Vary γ ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}. Use matrix inversion for policy evaluation, not iterative VI. Plot each PAM score vs γ.
3b. S sweep. Fixed: A=4, γ=0.95, R=Gaussian σ=1. Vary S ∈ {5, 10, 20, 50, 100} × T ∈ {Uniform, Dirichlet(0.1), Deterministic}. Plot advantage gap and V* variance vs S per T type.