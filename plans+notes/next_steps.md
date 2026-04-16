# Next Steps

**Project:** Defining Agentic Preferences — finite MDP testbed for agenticity measurement
**Status as of April 2026**

---

## 1. Recalibrate composite weights

**Why:** The `vstar_variance` fix (range-normalisation applied before variance) made the metric active for all reward types, not just Gaussian. The current weights (adv_gap=0.25, vstar_var=0.40, mi_diff=0.35) were set when vstar_var was nearly inactive. Composite scores will now shift upward for non-Gaussian rewards, and the relative contribution of each PAM has changed.

**How:** Run the full `01_pam_baseline` experiment (`01_build.py`, n≥200, S_values=[5,10,20]), extract the Q1 PAM correlation matrices per R_type, and recalibrate weights to minimise redundancy between adv_gap and vstar_var. If `r(adv_gap, vstar_var) > 0.85` across conditions, merge their weights equally rather than holding both at separate values.

---

## 2. Diagnose and likely drop MCE entropy from the composite

**Why:** Across all experiments (plots 03–13), MCE entropy saturates near 1.0 for all non-potential reward types and provides no discrimination. It responds only at extreme σ (σ≥5 for Gaussian) or extreme sparsity, making it nearly binary in practice. Including it in the composite adds noise without signal.

**How:** Run the full correlation analysis (step 1 above). If MCE entropy is uncorrelated with adv_gap and vstar_var across all conditions (i.e. it is measuring an independent but uninformative dimension), drop it from the composite entirely. It may still be useful as a standalone diagnostic for reward scale.

---

## 3. Implement `AgenticityGap` and `OptionValue` proxies

**Why:** Both are candidate proxies in `description.md` that are cleaner than existing PAMs for specific agenticity dimensions. Neither has been implemented yet.

**`AgenticityGap(R) = J(π*) − J(π*_myopic)`** where `π*_myopic` acts greedily on immediate reward at each step without lookahead. Large gap = R structurally requires multi-step planning. Parameter-free and more interpretable than the depth-h `planning_pressure` metric.

**`OptionValue(R) = E_s[V*(s) − max_a R(s,a)]`** — how much does the future matter relative to the best immediate reward? High = reward encodes long-range consequences that myopic optimisation cannot exploit.

**How:** Add both to `environments/pams.py`. Both are shaping-invariant (differences of value functions; the `max_a R(s,a)` term is action-indexed and does not involve Φ). Run against all 7 human-made MDPs (Q3) to verify the ordering matches intuition before including in sweeps.

---

## 4. Add `uniform_simplex` reward type

**Why:** Turner et al. (2021) derive their power-seeking results using a uniform prior over the reward simplex — every normalised reward vector equally likely. The `fraction_agentic` estimate can only be directly compared to Turner's theoretical prediction if we use the same prior. This R_type is specified in `pipeline2.md` §2.3 but not yet implemented in `runners.py`.

**How:** Add `R_type='uniform_simplex'` to `random_mdp` in `environments/runners.py`. Implementation: sample from `Dirichlet(1,...,1)` over all `S×A×S` entries (non-negative, sums to 1 across all triples). Note: this is unnormalised as a standard RL reward; document this and the Turner comparison context explicitly.

---

## 5. Fraction-agentic estimator with bootstrap CIs

**Why:** Joar's central question — "what fraction of reward functions are agentic?" — requires a principled estimator with uncertainty bounds, not just a point estimate. Specified in `pipeline2.md` §5.3 but deferred until composite weights are calibrated (which is step 1 above).

**How:** After recalibrating weights, implement:
```python
frac_agentic(prior, threshold=0.6, n_bootstrap=1000) -> (point_estimate, ci_lower, ci_upper)
```
Run for all three priors (Gaussian, spike_slab, uniform_simplex) and compare to Turner's measure-theoretic prediction. A discrepancy is a research finding — it means the priors differ in how they weight the reward space, which has substantive implications for what "most reward functions" means.

---

## 6. Corpus analysis of real-world reward functions

**Why:** `pipeline.md` defines a 15-source corpus (Sutton & Barto → MACHIAVELLI) with a structured analysis protocol applying 6 agenticity definitions. This analysis has not been done yet. It would provide the external validation that the sampling experiments cannot: do the PAMs correctly classify reward functions that are intuitively agentic or non-agentic?

**How:** Apply the protocol in `pipeline.md` to the tabular cases first (gridworld, cliff-walk, chain) — these are tractable and serve as ground-truth anchors. Then classify the structured continuous-control rewards (MuJoCo, dm_control) qualitatively. Mark all RLHF rewards as [low confidence] per the pipeline conduct rules.

Priority cases to compute first (computationally tractable, clear expected answers):
- Gridworld step-cost + goal → moderate agenticity
- Cliff-walking → high agenticity (irreversible penalty)
- Chain dense → high agenticity (all steps count)
- Chain terminal → low agenticity (agent choice barely matters)

---

## 7. Compositionality test

**Why:** InstructGPT's reward `R = r_φ(x,y) − β log(π/π_ref)` is a sum of a sparse outcome term and a dense KL regulariser. `pipeline2.md` §6.3 asks whether agenticity of a sum is a function of the agenticity of the components. This is also directly relevant to RLHF reward design.

**How:** Sample `R₁` (high agenticity, e.g. spike_slab p=0.05) and `R₂` (low agenticity, e.g. potential reward), form `R = αR₁ + (1−α)R₂` for α ∈ [0,1], compute composite score vs α. If linear: agenticity is compositionally additive. If nonlinear: identify the failure mode and its implications for RLHF composite rewards.

---

## 8. MACHIAVELLI external validation

**Why:** Pan et al. (2023) provide 134 text-based environments with ground-truth power-seeking annotations. Once step 5 (fraction-agentic estimator) establishes a calibrated threshold, apply PAMs to MACHIAVELLI reward functions and check whether environments empirically producing power-seeking score as agentic under the definitions. This is the external validation step the corpus analysis alone cannot provide.

**Prerequisite:** Steps 1 and 5 must be complete (calibrated composite + threshold) before MACHIAVELLI results are interpretable.

---

## Priority order

| Step | Dependency | Effort |
|---|---|---|
| 1 — Recalibrate weights | Re-run 01_build.py full | Low — just re-run |
| 2 — Drop MCE entropy | Step 1 | Low — edit pams.py |
| 3 — AgenticityGap / OptionValue | None | Medium — new pams |
| 4 — uniform_simplex R_type | None | Low — extend runners.py |
| 5 — Fraction-agentic estimator | Steps 1, 4 | Medium |
| 6 — Corpus analysis | None (parallel) | High — manual analysis |
| 7 — Compositionality test | Steps 1, 2 | Medium |
| 8 — MACHIAVELLI validation | Steps 1, 5 | High |
