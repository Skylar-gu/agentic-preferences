# Research Notes — Defining Agentic Preferences

**Project:** Finite MDP testbed for measuring reward function agenticity
**Status:** April 2026

---

## 1. Motivation

The goal is to characterise *agenticity* — how much an agent's planning matters given a particular reward function. This is not a property of the environment alone, but of the (reward, environment) pair: the same reward can be trivially myopic under one transition structure and require deep planning under another. All metrics are designed to be invariant to potential-based reward shaping (`R' = R + γΦ(s') − Φ(s)`), which changes value functions without changing policy ordering.

The broader motivation is understanding which reward functions in real AI systems (RLHF, verifiable rewards, process reward models) exhibit structural properties that incentivise long-horizon planning and power-seeking, in the sense of Turner et al. (2021).

---

## 2. Reward Priors

Three priors are used, each answering a different question about the distribution over reward functions.

### 2.1 Gaussian — pre-training analogy

```
R(s,a,s') ~ N(0, σ²)   i.i.d.
```

Models a neural reward model at random initialisation (CLT over many random weights). Dense by construction — every transition receives a non-zero signal. Represents the pre-RLHF distribution before conditioning on preference data. Advantage gap and V* variance are inflated by σ; MI differential should correctly score near zero (no irreversibility in a random dense reward).

### 2.2 Spike-and-slab — human-designed rewards

```
R(s,a,s') = mask × N(0, σ²),   mask ~ Bernoulli(p)
```

Models human-designed rewards: sparse (most entries zero) with varying magnitude at non-zero entries. Three independent parameters: **p** (sparsity), **σ** (signal magnitude), **heterogeneity** (variance of non-zero entries). Do not conflate p and magnitude. A Bernoulli × constant reward (no heterogeneity) is a lottery — the agent cannot discriminate good from bad rewarded states. Spike-and-slab with N(0,σ²) non-zero entries is structurally richer.

### 2.3 Uniform simplex — Turner comparability

```
R ~ Dirichlet(1,...,1)   over the S×A×S' simplex
```

Turner et al. (2021) derive power-seeking results using a uniform prior over the reward simplex. Using the same prior allows direct comparison of the `fraction_agentic` estimate to Turner's theoretical prediction. Non-negative, sums to 1. Empirically unrealistic as a reward function — use only for the Turner comparison. Any discrepancy between our estimate and Turner's is a research finding, not an error, because the priors differ.

---

## 3. Agenticity Definitions

Six definitions are applied to reward functions in the corpus analysis. Each has known failure modes.

**Definition A — Q-value dominance ratio**
```
α(R,T) = E[γV(s')] / E[R(s,a,s') + γV(s')]
```
α ≈ 1: future value dominates → agentic. α ≈ 0: immediate reward dominates → non-agentic. *Not shaping-invariant* — the Φ offset changes the denominator. Flag this for every application.

**Definition B — Recovery time**
How many steps until expected discounted return recovers to near-optimal after one random action? Short recovery (1–3 steps): non-agentic. Long/never: agentic. Reason from reward structure: does a single mistake compound?

**Definition C — Sparsity**
```
sparsity(R) = |{(s,a,s') : R(s,a,s') = 0}| / |S × A × S'|
```
High sparsity → agent must actively seek reward → agentic. **Known failure mode — the lottery problem:** a random lottery is sparse but not agentic (the agent cannot plan to win it). Flag every sparse reward against this counterexample.

**Definition D — State-space coverage under π***
Under π*, what fraction of the state space is visited in expectation? High coverage: agentic. Known failure mode: constant or random-walk rewards also produce high coverage. Distinguish reward-driven from dynamics-driven coverage.

**Definition E — Potential-shaping invariance check (not a classification)**
For each definition, ask: does the agenticity score change under R' = R + γΦ(s') − Φ(s)? Definitions A and B are most likely to fail. Record `invariant: yes / no / partially` for every reward function.

**Definition F — Turner POWER alignment**
Does maximising this reward cause π* to navigate toward high-POWER states (states with many reachable future states, options preserved)? *Critical distinction:* Turner's POWER is a property of states; this project characterises reward functions. Flag this distinction explicitly in every application.

---

## 4. Theoretical Predictions for Sampling Experiments

### Advantage gap under Gaussian R
With i.i.d. Gaussian entries and fixed T, Q(s,a) has variance σ²‖T(s,a,·)‖². Under uniform T, ‖T(s,a,·)‖² = 1/S → 0 as S grows — Q-value spread is suppressed by averaging. Advantage gap is proportional to σ and shrinks with S under uniform T. Under sparse Dirichlet T (k fixed), the spread is bounded regardless of S. This means Gaussian rewards in large MDPs will have low advantage gap *not because they are non-agentic but because environmental averaging suppresses Q-spread*.

### V* variance under Gaussian R
V* is a linear transform of R; under uniform T, all states have the same V* distribution → Var(V*) → 0. This is a correct mathematical prediction, not a metric failure. Both advantage gap and V* variance going to zero under large S with uniform T is expected — a random dense symmetric reward is genuinely non-agentic.

### MI differential
Under Gaussian R with no terminal structure, early and late actions contribute equally to G — MI differential should be near zero. MI differential is therefore the most discriminating PAM: it requires structured, not random, reward to score high.

### γ-sweep predictions
Agenticity scores should increase monotonically with γ, with a knee near γ ≈ 1 − 1/k for a reward requiring k-step planning. H_eps is expected to respond most cleanly to γ (it directly measures planning depth). As γ → 1, value iteration slows — use exact matrix inversion for policy evaluation.

### S-sweep predictions (three regimes)

| T structure | Advantage gap | V* variance | MI differential |
|---|---|---|---|
| Uniform (1/S each) | → 0 (averaging) | → 0 (CLT) | ≈ 0 (no structure) |
| Sparse (k-NN, k fixed) | Bounded | Bounded | Depends on topology |
| Deterministic | Bounded | Bounded | Potentially high |

---

## 5. Metric Design and Known Issues

### Shaping invariance

The core invariant: `V*(s)` and `V^rand(s)` both shift by `−Φ(s)` under shaping, so their *difference* `V*(s) − V^rand(s)` is shaping-invariant. All PAMs that normalise by quantities derived from this difference preserve invariance. The MCE entropy is invariant because the `Φ(s)` term in `Q(s,a)` is action-independent and cancels inside the softmax.

### H_eps — fix applied (March 2026)

**Problem:** H_eps was saturating at max_k=50 for all MDPs at γ=0.95. Root cause: the convergence bound is ratio_k ≤ γ^k, so at γ=0.95 and ε=0.05 the required max_k is ceil(log(0.05)/log(0.95)) = 59. With max_k=50 the metric *cannot* converge, returning H_eps=50 for all non-zero-gap MDPs. Potential rewards also returned H_eps=50 because numerical residuals of ~1e-9 in J*−J^rand (above the 1e-10 guard) triggered the full loop.

**Fix:** max_k now auto-computed as `ceil(log(ε)/log(γ)) + 10` (capped at 500). Gap guard raised from 1e-10 to 1e-6, correctly returning H_eps=0 for potential rewards.

**Residual behaviour:** H_eps responds primarily to γ (the planning depth imposed by discounting), not to reward structure at fixed γ. Both spike_slab and gaussian lines nearly overlap in the γ-sweep. H_eps is a diagnostic for "how long does planning need to be?" — not for discriminating reward types.

### V* variance — fix applied (March 2026)

**Problem:** Original `Var(V* − V^rand)` was scale-dependent, active only for high-σ Gaussian rewards. Sparse rewards (bernoulli, spike_slab) with small p produce value differences of order p × R_max/(1−γ), which are tiny at small p.

**Fix:** Range-normalise `V* − V^rand` before computing variance:
```
vv = Var( (delta − min(delta)) / (max(delta) − min(delta)) )
```
This puts the metric in [0, 0.25] regardless of σ or R_scale. vv_norm = 4×vv ∈ [0,1]. Maximum 0.25 = bimodal value landscape (half states at min, half at max).

**What it now measures:** Spread of the state-advantage map relative to its own dynamic range. A uniform value landscape scores 0; a peaked or bimodal landscape scores high.

### MCE entropy — fix applied (April 2026)

**Problem:** At α=0.1 (original default), all non-potential MDPs had Q/α with magnitude ~10–100, making the softmax near-deterministic for everything. `1 − H/log(A) ≈ 1.0` for all non-potential types — the metric only distinguished potential from non-potential.

**Fix:** Normalise R by std(R) before soft VI, making α scale-invariant. Other PAMs normalise their *outputs*; MCE must normalise its *input* because α lives in Q-value space. **α=0.25** chosen by empirical sweep (N=40 MDPs × 6 R_types × 8 α values), maximising between-type variance (0.0025) across non-potential types without upper or lower saturation.

**Post-fix behaviour:** potential → 0.0 (always); non-potential → 0.5–0.7 with ±0.07–0.20 within-type spread. MCE is now back in the composite by default.

### Composite weight recalibration (April 2026)

Original weights (adv_gap=0.25, vstar_var=0.40, mi_diff=0.35) were set when vstar_var was nearly inactive. After the range-normalisation fix, within each R_type r(adv_gap, vstar_var) = 0.02–0.23 (independent). The apparent global correlation of 0.871 was a between-group artifact (potential vs non-potential clustering). Equal weighting (all components at 0.50) is the principled choice. Auto-renormalisation means the number of active components doesn't change relative weights.

### T-sensitivity is expected, not a failure

A common misframing is that T-sensitivity is a failure mode — evidence that a metric measures (R,T) properties rather than R-intrinsic properties. This is wrong for this project. An MDP with a goal reward and uniform T is genuinely less agentic than the same reward with a sparse directed T — a good metric should reflect that. T-sensitivity is expected. The Q2 variance decomposition (T share vs R share) is a characterisation, not a pass/fail test.

---

## 6. Key Experimental Findings

### From Experiment 1 (Q1/Q2/Q3, Plots 01–08)

**Reward type ordering (Plot 03, adv_gap):** potential correctly collapses to ~0. gaussian > bernoulli > uniform ≈ goal, with goal showing the widest spread. Non-potential types cluster near 0.9–1.0 under random T — action-indexed R creates Q-value spread regardless of planning structure (known artefact of range-normalisation under random T).

**V* variance (Plot 03):** After the range-normalisation fix, active for all reward types, not just Gaussian. Before the fix it was essentially a Gaussian-reward detector.

**Lottery failure confirmed (Plot 09, Bernoulli):** adv_gap is non-monotone in p. Peaks at p=0.5 and collapses to 0% at p=1.0. At p=1.0 all entries are 1 → all Q(s,a) equal → zero advantage gap. Correctly identified as non-agentic for the wrong structural reason (homogeneity, not sparsity).

**Spike-and-slab resolves lottery failure (Plot 10):** adv_gap is monotone in p, no collapse at p=1.0 — N(0,1) magnitudes create Q-value heterogeneity even at full density. Validates spike-and-slab as strictly richer than Bernoulli.

**γ-sweep (Plot 11):** H_eps is the only metric with clean, large-magnitude response to γ (0.32 at γ=0.5 to 0.85 at γ=0.99). Both R_types overlap nearly perfectly — H_eps is tracking discounting depth, not reward structure. adv_gap is mostly flat; vstar_var is flat for spike_slab and shows a discontinuity near γ=1 for gaussian (expected: variance diverges as γ → 1 for zero-mean rewards).

**T-sensitivity (Plot 12):** For the same R, scores under Uniform T vs Deterministic T have correlations r = 0.01–0.15 — near zero. Under Uniform T, adv_gap saturates near 1.0 for gaussian regardless of R structure. Root cause: under Uniform T, both numerator (mean action gap) and denominator (range(V*−V^rand)) scale as O(σ/√S), so their ratio is S-invariant. The range-normalisation removes the signal the raw metric would have shown.

**S-sweep (Plot 13):** Under Uniform T, adv_gap stays high across all S (~0.85 at S=100) rather than decreasing as predicted. Same root cause as T-sensitivity — range-normalisation makes the ratio S-invariant. Under Dirichlet and Deterministic T, vstar_var decreases from ~0.6 (S=5) to ~0.15 (S=100) — a sample-size artefact of fixed σ=1 Gaussian R diluting across more states, not a structural signal.

**Q2 variance decomposition (Plot 04):** All T-share ratios below 0.5 for all R types — R dominates composite variance in all tested conditions. Goal R has the highest T-share at S=5 (~0.43) because goal reachability depends heavily on T topology. Gaussian and bernoulli near-zero T-share. Note: n_fixed_T=5 is small; may underestimate true between-T variance.

### From Experiment 2 (MCE calibration, Plots 14–17)

Between-type variance peaks at α=0.25 and α=0.5 (0.0023–0.0041 across 5 canonical T's) with no saturation at either end. At α ≤ 0.1 all non-potential types cluster at ~0.9 (upper saturation). At α ≥ 2.0 everything collapses to ~0 (lower saturation). The T-sensitivity of MCE at α=0.25 is visible as within-colour spread in Plot 14 — most R_types show moderate T-sensitivity (±0.07–0.10 between T matrices).

---

## 7. Corpus Analysis Protocol

### Sources (15 reward functions)

**Tier 1 — Classical RL:** Sutton & Barto gridworld/cliff/cartpole; Atari (Bellemare 2013, dense vs sparse games); dm_control MuJoCo suite (Tassa 2020, explicit tolerance() functions); Gymnasium locomotion (HalfCheetah/Ant/Hopper, decompose each component separately); Ng et al. (1999) shaping examples.

**Tier 2 — RLHF:** Christiano et al. (2017) trajectory preference model; Ziegler et al. (2019) LM fine-tuning; Ouyang et al. (2022) InstructGPT `R = r_φ(x,y) − β log(π/π_ref)` — the KL term is a dense per-token secondary reward, *not* just a constraint; Lightman et al. (2023) PRM vs ORM — structurally critical difference in temporality.

**Tier 3 — Verifiable rewards:** DeepSeek-R1 (Guo 2025) binary accuracy + format rewards; RLVR code execution pass@k (Lambert 2024); DeepSeekMath GRPO (Shao 2024) — note GRPO affects training algorithm, not the reward function itself.

**Tier 4 — AI safety:** Turner et al. (2021) POWER — read carefully; Turner & Tadepalli (2022) retargetable decision-makers; MACHIAVELLI (Pan 2023) — 134 text-based environments with ground-truth power-seeking annotations.

### Extraction format

For each reward function: `id`, `source`, `domain`, `R_formula` (mark [inferred] if not explicit), `temporality` (immediate/terminal/episodic-mixed), `sparsity` (sparse/dense/mixed), `composites` (list components if yes).

### Application of definitions

Apply all six definitions (A–F) to every reward function. For composite rewards (e.g. InstructGPT KL penalty, MuJoCo multi-term), classify each component separately before classifying the composite. Never collapse definitions prematurely — keep them separate until meta-analysis. Mark all RLHF classifications [low confidence] unless structural reasoning is clearly sufficient.

Always flag the lottery counterexample when applying Definition C. Always run Definition E (shaping invariance check) — do not skip it.

### Output table columns

`id`, `source`, `domain`, `R_formula`, `temporality`, `sparsity`, `Def_A` [0,1], `Def_B` [0,1], `Def_C` [0,1], `Def_D` [0,1], `Def_F` [0,1], `overall` (majority vote A/D/F with explicit override if needed), `inv_flag` (which definitions failed Definition E), `confidence` (low/medium/high), `notes`.

### Meta-analysis questions (8)

1. **Empirical base rate** — what fraction scores above 0.5? Varies by domain? Connect to Turner's measure-theoretic prediction.
2. **Definition disagreements** — which pairs most often conflict? What does each conflict reveal?
3. **False positives/negatives** — clearest false positive + false negative; propose minimal fix.
4. **Invariance failures** — which definitions, which rewards? Implications for RLHF regularisation.
5. **Proposed refinement** — one refined or composite definition, stated mathematically, with remaining weaknesses.
6. **Relationship to Turner** — does "most reward functions make power-seeking optimal" imply most are agentic under our definitions? What does our definition capture that Turner's does not?
7. **Verifiable rewards** — DeepSeek-R1's binary terminal reward scores as highly agentic under Def C (sparse). Is this a genuine Def C failure, or is the intuition wrong? Argue both sides.
8. **Compositionality** — is agenticity of R₁ + R₂ a function of agenticity of R₁ and R₂ separately? Formal argument or counterexample from the InstructGPT KL penalty case.

---

## 8. Open Questions and Next Steps

Steps 1–5 are complete. Remaining work:

### Done
- **Recalibrate composite weights** — equal weights (0.50 each) after confirming within-R_type r(adv_gap, vstar_var) = 0.02–0.23
- **MCE entropy** — re-enabled with R normalisation and α=0.25; back in composite
- **AgenticityGap and OptionValue** — implemented as diagnostic proxies in `pams.py` (not shaping-invariant; documented)
- **uniform_simplex R_type** — added to `random_mdp` in `runners.py`
- **fraction_agentic estimator** — implemented in `runners.py` with 95% bootstrap CI

### Remaining

**6 — Corpus analysis of real-world reward functions**
Apply the protocol above to the 15-source corpus. Start with tractable tabular cases (gridworld, cliff-walk, chain variants) as ground-truth anchors before moving to MuJoCo and RLHF. Expected outputs: structured classification table + meta-analysis (questions 1–8).

**7 — Compositionality test**
Sample R₁ (high agenticity, e.g. spike_slab p=0.05) and R₂ (low agenticity, e.g. potential), form R = αR₁ + (1−α)R₂ for α ∈ [0,1], plot composite score vs α. If linear: agenticity is additively decomposable. If nonlinear: characterise the failure mode. Directly relevant to InstructGPT's KL penalty (dense low-agenticity term mixed with sparse high-agenticity outcome term).

**8 — MACHIAVELLI external validation**
Pan et al. (2023): 134 text-based environments with ground-truth power-seeking annotations. Once composite + threshold are well-calibrated (step 5 complete), apply PAMs to MACHIAVELLI reward functions and check whether environments empirically producing power-seeking score as agentic. Prerequisites: calibrated composite + threshold; ideally corpus analysis (step 6) for cross-validation.

### Open design questions

- `spike_slab` hardcodes σ=1.0 for non-zero entries. Varying σ independently of p requires a separate `R_scale` parameter (currently ignored for spike_slab).
- Under Uniform T, adv_gap and vstar_var are S-invariant due to range-normalisation cancellation. Whether this is a feature (correct invariance) or a bug (masking a real signal) depends on whether S-scaling matters for the application.
- The right prior for "fraction of reward functions that are agentic" remains open. Gaussian, spike-and-slab, and uniform simplex each weight the reward space differently. The scientifically meaningful prior is over human-written reward functions — more corpus analysis needed.
- Whether agenticity at random initialisation (Gaussian prior) predicts agenticity post-RLHF fine-tuning is itself an empirical question.
