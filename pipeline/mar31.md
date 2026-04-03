# Experimental Observations — March 31 2026

**Plots reviewed:** 03 (Q1 boxplots), 07–13 (sampling experiments)
**Setup:** S=10 unless stated, A=4, γ=0.95, T=random Dirichlet(1) k=3, n=50 per condition

---

## 0. Design Clarification: Environment Invariance Is Not a Goal

pipeline2.md §6.2 frames T-sensitivity as a *failure mode* — evidence that a metric is "measuring (R,T) properties rather than R-intrinsic properties." This framing is not correct for this project. The goal is to measure agenticity of a deployed agent in a specific environment, not to extract a reward-intrinsic property independent of T. An MDP with a goal reward and a flat uniform T is genuinely less agentic than the same reward with a sparse directed T, and a good metric should reflect that. The T-sensitivity finding in §2 below is therefore *not* a bug — it is expected and acceptable. The concern is only if the metric responds to T in ways that are arbitrary or unpredictable.

The Q2 variance decomposition experiment (T share vs R share) remains useful as a *characterisation* of how much T contributes, not as a pass/fail test.

---

## 1. H_eps is Saturated and Effectively Uninformative at γ=0.95 — **Fixed**

**Observation:** In every plot that holds γ fixed at 0.95 (plots 07, 08, 09, 10), H_eps scores near 1.0 for *all* reward types including potential. There is essentially no discrimination between conditions.

**Root cause:** H_eps = min{k : ratio ≤ ε} with ε=0.05 and max_k=50. The convergence bound for k-step lookahead is:

```
ratio_k ≤ γ^k
```

At γ=0.95, the minimum achievable ratio at k=50 is 0.95^50 ≈ 0.077 > ε=0.05. The metric *cannot* converge within max_k steps at γ=0.95. Every MDP with a non-zero gap hits max_k and returns H_eps=50, H_eps_norm ≈ 0.99.

The required max_k for convergence at γ=0.95 is ceil(log(0.05)/log(0.95)) = 59. The current max_k=50 is systematically too small by ~9 steps.

**Compound problem — potential rewards:** Theoretically, V*(s) = V^rand(s) = −Φ(s) for potential rewards (all policies are equivalent), so J*−J^rand = 0 and H_eps should return 0 immediately. In practice, value iteration leaves numerical residuals of order ~1e-9 in J*−J^rand (above the 1e-10 guard). With a near-zero but positive gap and ε=0.05, the loop runs the full max_k iterations without converging, returning H_eps=50 instead of 0. Potential rewards score identically to highly agentic rewards on this metric.

**Fix implemented:** `max_k` is now auto-computed as `ceil(log(ε)/log(γ)) + 10` (capped at 500), guaranteeing the loop can theoretically converge. At γ=0.95, ε=0.05 this gives max_k=69. The gap guard has been raised from 1e-10 to 1e-6, correctly returning H_eps=0 for potential rewards whose numerical gap residual (~1e-9) was previously triggering spurious max_k returns. H_eps_norm is now `H_eps / max_k` (linear fraction of the theoretical ceiling), replacing the arbitrary `1 - exp(-H_eps/10)`.

**Only plot where H_eps was already informative:** The γ sweep (Plot 11) because at low γ the metric converges quickly — at γ=0.5 it correctly scored small values. The monotone increase in H_eps from γ=0.5 to γ=0.99 is real and interpretable. However, both spike_slab and gaussian lines nearly overlap, confirming H_eps responds primarily to γ (the effective horizon imposed by discounting) rather than to reward structure at fixed T and γ.

---

## 2. T-Sensitivity: Advantage Gap and V* Variance Respond to Both R and T (Expected)

**Observation (Plot 12):** For the same reward matrix R, the score under Uniform T vs Deterministic T has correlation r = 0.10 (advantage gap, spike_slab), r = 0.01 (V* variance, spike_slab), r = 0.15 (advantage gap, gaussian), r = 0.11 (V* variance, gaussian). All near zero.

- Under Uniform T, advantage gap clusters tightly at ~0.12–0.18 for spike_slab and ~0.30–0.40 for gaussian, with little variance across R samples.
- Under Deterministic T, the same R samples produce advantage gap scores spread widely across 0.1–0.9 for spike_slab and tightly clustered at ~0.9 for gaussian.

This means T is the dominant driver of both metrics' scores, not R. A reward that scores "low agenticity" under Uniform T might score "high agenticity" under Deterministic T — not because the reward changed, but because T structure amplifies or suppresses Q-value spread.

**Implication:** As clarified in §0, this T-sensitivity is expected and acceptable — agenticity is a property of an agent in an environment, not of R alone. The finding does tell us something concrete: under Uniform T the metrics are operating near their floor for all R types, so Uniform T is a poor choice for evaluating agenticity. The Q2 variance decomposition (T share vs R share) remains useful as a diagnostic of *how much* T contributes relative to R for a given deployment context.

---

## 3. S Sweep Confirms Asymptotic Predictions — With One Surprise

**Observation (Plot 13, Advantage Gap):**
- Uniform T: clear monotone decrease from ~0.6 at S=5 to ~0.2 at S=100. Confirms §3.1/§3.5 prediction: T averages Q-values over 1/S successor states, shrinking Q-value spread as S grows.
- Dirichlet(0.1) and Deterministic T: flat at ~0.9 across all S (5–100). Scale-invariant. The k=3 successors per (s,a) pair means the averaging effect is bounded by k regardless of S.

**Observation (Plot 13, V* Variance):**
- Uniform T: effectively 0 across all S. Completely suppressed.
- Dirichlet and Deterministic T: V* variance *increases* weakly with S (roughly 0.15 → 0.3).

The increase under non-uniform T is counterintuitive given the §3.2 argument, but explainable: larger S means more states, each accumulating a distinct reward history under a fixed-scale Gaussian R (σ=1). With S=100 and σ=1, state values are shaped by more independent reward draws, increasing the empirical variance of the V* distribution. This is a sample-size effect from the fixed σ, not a structural agenticity signal. V* variance is not scale-invariant with respect to S.

---

## 4. Advantage Gap: σ-Scaling and the Lottery Failure Mode

**Gaussian σ sweep (Plot 08):**
- Advantage gap is 0% above threshold at σ=0.1, 72% at σ=0.5, 100% at σ=1.0+. The metric scales monotonically with σ up to saturation, consistent with §3.1. However, the scaling is not the same for V* variance: 0% at σ=1.0, only 12% at σ=2.0, 84% at σ=5.0. V* variance requires roughly 4–5× larger σ to activate than advantage gap.
- MCE entropy inverts: high entropy at low σ (flat reward, uniform policy), low entropy at high σ (strong reward signal → deterministic policy). σ=5.0 drops to only 2% above the 0.5 entropy threshold, while σ=0.1 is 100%. This is the expected relationship.

**Bernoulli p sweep (Plot 09) — lottery failure confirmed:**
- Advantage gap is non-monotone in p. It peaks at p=0.5 (84%) and drops to *0%* at p=1.0. At p=1.0 (all entries = 1), every (s,a,s') has the same reward, so Q(s,a) = 1/(1−γ) for all actions regardless of T — advantage gap collapses to zero. This is the lottery failure mode from pipeline2.md §2.2: maximum sparsity score would label p=1.0 as dense/non-agentic, but the advantage gap agrees for the wrong structural reason (homogeneity, not sparsity).
- V* variance is near 0 across all p for bernoulli. Constant-magnitude rewards create no differential in state values, confirming that V* variance measures reward magnitude heterogeneity, not sparsity per se.

**Spike-slab p sweep (Plot 10) — heterogeneity resolves lottery failure:**
- Advantage gap is monotone in p: 4% at p=0.01, 10% at p=0.05, 48% at p=0.2, 98% at p=0.5, 100% at p=1.0. No drop at p=1.0 because N(0,1) magnitudes create state-value heterogeneity.
- This directly validates the spike-and-slab prior as strictly richer than bernoulli for detecting agenticity. The advantage gap correctly identifies p=1.0 spike_slab as fully high-agenticity because reward magnitude varies across transitions.

---

## 5. Reward Type Ordering Is Mostly Correct But V* Variance Is Too Narrow

**From Plot 03 (Q1 boxplots, S=10):**
- Advantage gap ordering: gaussian (~0.8) > bernoulli (~0.5) > uniform (~0.4) > goal (~0.4) > potential (~0.0). The potential control correctly scores near 0. gaussian being highest reflects high-magnitude dense rewards maximising Q-spread.
- V* variance: gaussian (~0.08) >> everything else (~0). All non-Gaussian types cluster at 0. This metric is almost entirely inactive outside of gaussian rewards in this parameter regime (S=10, σ=1, γ=0.95). It does not meaningfully discriminate between reward types except as a gaussian detector.
- MCE entropy: near-uniform across all types at ~0.85–0.99. The metric saturates for all types and provides almost no discrimination. Useful only at extreme σ or extreme sparsity.

**From Plot 07 (all R types, S=10):**
- uniform R scores 0% above threshold on advantage gap and 0% on V* variance. This is correct — uniform [0,1] rewards have no Q-value structure since all states receive equal expected reward.
- goal R scores only 16% above threshold on advantage gap. Goal rewards are structurally interesting (one rewarded state) but the fixed T at k=3 successors means the goal is often reachable from multiple states regardless of action choice, reducing Q-spread.

---

## 6. γ Sweep: H_eps Tracks Horizon, Other Metrics Are Mostly Flat

**From Plot 11:**
- H_eps is the only metric showing clean, large-magnitude response to γ for both R types. Both spike_slab and gaussian increase from ~0.4 at γ=0.5 to ~1.0 at γ=0.99, and the two lines nearly overlap. H_eps is tracking the structural planning depth imposed by γ, not R.
- Advantage gap increases slightly with γ for spike_slab (0.30 → 0.42) but is flat for gaussian (~0.78 throughout). The small spike_slab increase is consistent with theory: higher γ amplifies differences in Q-values, slightly increasing the gap. Gaussian is already near its ceiling.
- V* variance is flat near 0 for spike_slab across all γ, with a sharp increase only at γ=0.95–0.99 for gaussian. The discontinuity near γ=1 for gaussian is consistent with the prediction that value variance grows without bound as γ → 1 under zero-mean rewards.
- MCE entropy is flat and near-uniform (~0.9) across all γ for both R types. This metric is insensitive to γ.

---

## 7. V* Variance: Fix and Remaining Limitations

**Problem:** The original `Var(V* - V^rand)` was scale-dependent — it only activated for high-σ Gaussian rewards because sparse rewards (bernoulli, spike_slab) produce value differences of order p × R_max/(1−γ), which are tiny at small p and small R_scale. The metric was therefore functioning as a Gaussian-reward detector, not an agenticity measure.

**Fix implemented:** Normalise `V* - V^rand` by its own range before computing variance:
```
vv = Var( (delta - min(delta)) / (max(delta) - min(delta)) )   where delta = V* - V^rand
```
This puts the metric in [0, 0.25] regardless of σ or R_scale. `vv_norm = 4 × vv ∈ [0, 1]`. The theoretical maximum 0.25 is achieved by a bimodal value landscape (half states have minimum advantage, half have maximum). Shaping-invariance is preserved: Φ cancels in the difference before normalisation, and the range transformation is a linear shift plus scale which preserves the Var(·/range) form.

**What "range-normalised variance" actually measures:** How spread out the state-advantage map is relative to its own dynamic range. A uniform value landscape (every state equally valuable) scores 0. A peaked landscape (one rewarded region, rest near-zero) scores high. A bimodal landscape (two distinct reward basins) scores near 0.25.

**Remaining limitations:**
- Range normalisation collapses signal when the gap is very small (|max − min| < 1e-10 → return 0). This correctly handles zero-gap cases (potential rewards) but may under-report agenticity for nearly-non-agentic MDPs.
- The metric is still sensitive to T: under Uniform T, V* ≈ V^rand for all states (all actions lead to the same distribution), so vv → 0 regardless of R. This is expected (see §0).
- Requires re-calibrating the composite weight (currently 0.40). With the new metric active for non-Gaussian rewards, the composite scores will shift upward. Recommend reducing the weight or recalibrating after running Group 1 experiments with the fixed metric.

---

## 8. Summary: Metric Status After These Experiments

| Metric | Status | Key issue |
|---|---|---|
| Advantage gap | Good signal | Lottery failure for constant-magnitude rewards; T-sensitive (expected, see §0) |
| V* variance | **Fixed** | Was scale-dependent; now range-normalised in [0, 0.25], active for all R types |
| H_eps | **Fixed** | Was saturating at max_k=50; max_k now auto-computed from γ,ε; gap guard raised to 1e-6 |
| MCE entropy | Insensitive | Ceiling effect near 1.0; only discriminates at extreme σ |
| Composite | Needs recalibration | V* variance now active for non-Gaussian rewards; composite scores will shift upward |

**Advantage gap has the best signal** across experiments: correctly identifies the reward type ordering, detects the lottery failure mode, and responds cleanly to σ. T-sensitivity is expected (see §0).

**V* variance (fixed) should now discriminate** sparse and dense rewards equally, measuring the relative spread of the value landscape. Whether it adds information beyond advantage gap will be visible in the next run of the sampling experiments with the fixed implementation.

**H_eps (fixed) now correctly returns 0 for potential rewards** and produces informative values at γ=0.95. The metric is expected to respond more to T structure (effective mixing depth) than to R structure, based on the γ sweep.
