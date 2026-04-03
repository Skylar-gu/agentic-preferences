# Plot Descriptions

**Setup (plots 01–08):** S ∈ {5, 10}, A=4, γ=0.95, k=3, n=50 per cell. MI excluded. Q1/Q3 use T=Dirichlet(α=1.0) k=3. Q2 sweeps α ∈ {0.1, 1.0, 10.0}.
**Setup (plots 09–13):** Vary one parameter per plot; see individual entries for specifics.

---

## 01 — Q3 Heatmap (`01_q3_heatmap.png`)

**What:** 7 human-made MDPs × 7 PAMs (adv_gap, vstar_var, H_eps, mce_entropy, ctrl_adv, one_step, composite), YlOrRd color scale, sorted descending by composite. Vertical white line separates composite from individual metrics.

**Observed patterns:**
- Chain-Dense ranks #1 in composite (0.40) despite being the least planning-intensive chain, driven by its anomalously high H_eps (0.93) — the dense reward stream requires 64 lookahead steps to converge, inflating the composite.
- MCE entropy dominates visually: all non-potential MDPs score 0.82–0.98, making the column uniformly dark and uninformative as a discriminator.
- adv_gap separates chains (0.19–0.37) from grids (0.10–0.33) but Grid-Cliff scores lowest (0.10), which is counterintuitive — the cliff penalty should increase action differentiation.
- vstar_var is near-uniform across all MDPs (0.27–0.43), providing little discrimination.
- ctrl_adv and one_step are near-zero for all chain MDPs (0.00–0.01) and only moderately active for grids (0.04–0.29), likely a calibration issue with the empirical W2 scales.
- Composite ordering (Dense > Terminal ≈ Lottery > Progress > Goal > Local > Cliff) does not match expected agenticity ordering and is dominated by H_eps and MCE entropy artefacts.

---

## 02 — Q3 Grouped Bar (`02_q3_grouped_bar.png`)

**What:** Same Q3 data as plot 01 as a grouped bar chart: adv_gap, vstar_var, H_eps, ctrl_adv, and composite per MDP side by side.

**Observed patterns:**
- H_eps is the dominant bar for Chain-Dense (0.93), dwarfing all other metrics and pulling its composite to the top. For all other MDPs H_eps is 0.09–0.12.
- vstar_var (orange) is nearly identical across all 7 MDPs, adding noise rather than signal to the composite.
- ctrl_adv is essentially zero for chain MDPs and small for grids — it provides almost no contribution to the composite in practice.
- Composite (black) closely tracks adv_gap + vstar_var for all MDPs except Chain-Dense, where H_eps distorts the result.

---

## 03 — Q1 Boxplots by R_type (`03_q1_boxplots_by_Rtype.png`)

**What:** Four boxplot panels (adv_gap, vstar_var, H_eps, mce_entropy) at S=10, T fixed, n=50 per R type.

**Observed patterns:**
- **adv_gap:** potential correctly collapses to ~0. All other types (gaussian, uniform, bernoulli, goal) cluster near 1.0, with goal showing the widest spread (IQR ~0.6–1.0). The near-saturation of non-potential types is a known artefact of the current normalization — action-indexed R under random T creates artificial action differentiation even when T structure does not require planning.
- **vstar_var:** potential correctly collapses to ~0. All other types cluster at 0.3–0.5 with substantial overlap, providing weak discrimination. Bernoulli reaches slightly higher (up to 0.6). No type clearly dominates.
- **H_eps:** potential correctly returns 0. All non-potential types cluster tightly at ~0.85 with near-zero variance — H_eps is tracking γ=0.95, not R structure. It is uninformative for R_type discrimination.
- **MCE entropy:** potential correctly returns 1.0 (uniform policy). All other types are near 0.9–1.0, with gaussian showing the most spread (down to ~0.8). Essentially no discrimination between non-potential types.

---

## 04 — Q2 Variance Decomposition (`04_q2_variance_decomp.png`)

**What:** Bar chart of T-share ratio (between_T_var / total_var) per (S, R_type), with three bars per group: Dir(α=0.1), Dir(α=1.0), Dir(α=10.0). One subplot per S. Red dashed line at 0.5.

**α interpretation:** α=0.1 → near-deterministic (weight concentrated on 1 of k=3 successors); α=1.0 → balanced; α=10.0 → near-uniform over k=3 successors.

**Observed patterns:**
- All T-share ratios are below 0.5 for all R types and both S values — R dominates composite variance in all conditions tested. T topology never exceeds 50% contribution.
- **goal R** has the highest T-share at S=5 (~0.43 for α=0.1 and α=10.0), dropping to ~0.16 at S=10. Goal R's T-sensitivity is expected: whether the goal state is reachable depends heavily on which successors each (s,a) pair connects to.
- **gaussian and bernoulli** have near-zero T-share across all α at both S values — the composite is almost entirely determined by R for these types.
- **potential** is near-zero (correctly — potential rewards produce near-identical composite scores regardless of T since V*≈V^rand).
- **α ordering is inconsistent:** no monotone trend in T-share with α, suggesting that the expected pattern (α=0.1 → high T-share because T samples are more variable) is not clearly materialising at these sample sizes (n_fixed_T=5).

---

## 05 — Q1 Scatter: adv_gap vs MCE Entropy (`05_q1_scatter_gap_vs_entropy.png`)

**What:** Two scatter panels (S=5, S=10) plotting adv_gap_norm (x) vs mce_entropy_norm (y) coloured by R_type. Grey anti-diagonal for reference.

**Observed patterns:**
- **Potential** correctly isolates at top-left (adv_gap≈0, entropy≈1.0) in both panels — the non-agentic control is cleanly separated.
- All non-potential types cluster in the top-right (high adv_gap, high entropy ~0.8–1.0). The expected negative correlation (high gap → low entropy) is absent: MCE entropy is saturated near 1.0 regardless of adv_gap magnitude.
- At S=5 the non-potential cluster is very tight (adv_gap 0.9–1.0, entropy 0.8–1.0). At S=10 there is more horizontal spread but entropy remains near ceiling, confirming MCE entropy is not a useful discriminator under these conditions.
- The plot reveals that adv_gap and MCE entropy are largely independent at these parameter settings — they are not measuring complementary aspects of the same signal.

---

## 06 — Q1 Radar Chart (`06_q1_radar_by_Rtype.png`)

**What:** Two polar/spider charts (S=5, S=10) with four axes (adv_gap, vstar_var, H_eps, mce_entropy). Each R type plotted as a filled polygon using mean normalised PAM values.

**Observed patterns:**
- Potential (red) is a collapsed thin spike pointing only along mce_entropy — all other axes near zero. This is the correct non-agentic signature.
- All non-potential types (gaussian, uniform, bernoulli, goal) produce nearly identical large polygons, nearly indistinguishable from each other. The radar provides no R_type discrimination beyond potential vs. non-potential.
- H_eps and mce_entropy axes dominate the polygon size for all non-potential types (both near 0.85–1.0), while adv_gap (~0.9) and vstar_var (~0.35–0.4) are the only axes that vary. The polygon shape is essentially the same across all non-potential types at both S values.

---

## 07 — Reward Distribution Agenticity (`07_reward_dist_agenticity.png`)

**What:** Four histogram panels (adv_gap, vstar_var, H_eps, mce_entropy) at S=10, T fixed, n=50. Dashed line at threshold 0.5; fraction above threshold in legend.

**Observed patterns:**
- **adv_gap:** gaussian (10%), uniform (100%), bernoulli (100%), goal (100%) — extremely high fractions for all non-potential types except gaussian. Gaussian scores lower because the action-indexed Gaussian R with fixed T creates a slightly different distribution. Potential is 0% (correct). The uniformly high fractions for uniform/bernoulli/goal confirm saturation from action-indexed reward artefact.
- **vstar_var:** gaussian (10%), uniform (0%), bernoulli (8%), potential (0%), goal (0%) — all very low fractions. Most scores cluster at 0.3–0.5, below threshold. vstar_var is essentially inactive as a high-agency detector at these parameters.
- **H_eps:** potential (0%), all others (100%) — a binary separator between potential and everything else. No discrimination among non-potential types; all cluster at ~0.85 (γ-driven).
- **mce_entropy:** potential (100%), all others (100%) — completely uninformative. Even potential scores high because mce_entropy=1.0 for a uniform policy over all actions.

---

## 08 — R_scale Sweep: Gaussian Rewards (`08_rscale_sweep_agenticity.png`)

**What:** Four histogram panels (adv_gap, vstar_var, H_eps, mce_entropy) sweeping σ ∈ {0.1, 0.5, 1.0, 2.0, 5.0} for Gaussian R at S=10, T fixed, n=50.

**Observed patterns:**
- **adv_gap:** all σ values give 98–100% above threshold, fully saturated. The range(V*−V^rand) normalisation removes σ-sensitivity — the ratio is σ-invariant under random T (numerator and denominator scale together). This is the same artefact as in plot 07.
- **vstar_var:** strongly σ-sensitive. σ=0.1 → 12%, σ=0.5 → 6%, σ=1.0 → 6%, σ=2.0 → 8%, σ=5.0 → 8%. Despite range-normalisation, vstar_var remains near-zero across all σ values. The metric is not activating reliably for Gaussian R at these S and T settings.
- **H_eps:** all σ → 100%, saturated at ~0.85. Completely σ-insensitive as expected (H_eps tracks γ, not R).
- **mce_entropy:** inverts with σ: σ=0.1 → 100% above 0.5, σ=5.0 → 2%. This is the expected direction — larger rewards create more deterministic optimal policy — but the inversion only becomes apparent at extreme σ. This confirms MCE entropy is only informative at the extremes of the reward scale.

---

## 09 — Bernoulli p Sweep (`09_bernoulli_p_sweep.png`)

**What:** Four histogram panels sweeping sparsity p ∈ {0.01, 0.05, 0.2, 0.5, 1.0} for Bernoulli R (entries 0 or 1 with prob p). S=10, T fixed, n=50.

**Observed patterns:**
- **adv_gap:** non-monotone in p. p=0.01 → 62%, p=0.05–0.5 → 100%, p=1.0 → 0%. The collapse at p=1.0 is the lottery failure mode: when all entries are 1, all Q-values are identical and adv_gap=0. This confirms the metric correctly identifies constant-magnitude rewards as non-agentic.
- **vstar_var:** near-zero across all p (0–12%). Constant-magnitude Bernoulli rewards produce no state-value heterogeneity regardless of sparsity — vstar_var correctly stays low.
- **H_eps:** same γ-tracking as before: 0% at p=0.01 (gap near zero — too sparse to differentiate V* from V^rand), 100% at p=0.05 and above.
- **mce_entropy:** all p → 100%, completely uninformative.

---

## 10 — Spike-and-Slab p Sweep (`10_spike_slab_p_sweep.png`)

**What:** Same structure as plot 09 but for spike-and-slab R (entries 0 w.p. 1−p, else N(0,1)).

**Observed patterns:**
- **adv_gap:** p=0.01 → 50%, p=0.05 → 96%, p=0.2–1.0 → 100%. Monotone increase, and crucially **no collapse at p=1.0** — because N(0,1) magnitudes vary across transitions, full density still creates Q-value heterogeneity. This directly validates spike-and-slab over Bernoulli: the non-zero magnitude variation prevents the lottery failure mode.
- **vstar_var:** near-zero across all p (0–10%), same as Bernoulli. Despite spike-and-slab having richer magnitude structure than Bernoulli, vstar_var remains essentially inactive. The range-normalisation may be suppressing signal here.
- **H_eps:** same γ-tracking: p=0.01 → 62% (sparse enough to sometimes produce zero-gap MDPs), p≥0.05 → 100%.
- **mce_entropy:** all p → 100%, uninformative.

---

## 11 — γ Sweep (`11_gamma_sweep.png`)

**What:** Four line plots (adv_gap, vstar_var, H_eps, mce_entropy) vs γ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99} for spike_slab (p=0.1) and gaussian (σ=1), S=20, T=random, n=50.

**Observed patterns:**
- **adv_gap:** gaussian (~0.9–1.0) stays flat and high across all γ. spike_slab (0.6–0.7) is lower and also flat, with slight upward trend at high γ. No meaningful γ-sensitivity — the range normalisation removes the γ-amplification effect.
- **vstar_var:** both types flat at ~0.27–0.30 across all γ. Completely γ-insensitive. Expected sharp increase near γ=1 (value variance diverges) is absent, likely suppressed by range normalisation.
- **H_eps:** both types increase monotonically from ~0.32 (γ=0.5) to ~0.85 (γ=0.99) with near-identical trajectories. H_eps is the only metric that responds cleanly to γ and provides no R_type discrimination (both lines overlap). Confirms H_eps measures planning depth imposed by discounting, not reward structure.
- **mce_entropy:** spike_slab flat at ~1.0; gaussian flat at ~0.87–0.90. Both γ-insensitive. Gaussian is slightly lower (near-deterministic optimal policy due to large reward magnitudes), spike_slab is at ceiling (sparse rewards → near-uniform policy).

---

## 12 — T-sensitivity Scatter (`12_t_sensitivity.png`)

**What:** 2×2 scatter grid. Rows = R type (spike_slab p=0.1, gaussian σ=1). Columns = metric (adv_gap, vstar_var). Each point: same R sample scored under Uniform T (x) vs Deterministic T (y). S=20, n=100.

**Observed patterns:**
- **adv_gap, spike_slab (r=0.12):** Uniform T clusters at x=0.5–1.0 (high scores), Deterministic T spreads widely y=0.2–1.0. Near-zero correlation. **Unexpected:** Uniform T gives high adv_gap, not low. Root cause: under Uniform T with action-indexed R, E_R[s,a] varies across actions even though T is uniform. The range-normalisation preserves this R-driven action differentiation and the 1/S averaging cancels in the ratio (as shown analytically). adv_gap is measuring immediate reward variation across actions, not planning-relevant differentiation.
- **adv_gap, gaussian (r=0.03):** Uniform T saturates at x=1.0 (all samples). Deterministic T spreads 0.5–1.0. Same artefact, more extreme — Gaussian R saturates adv_gap under Uniform T completely.
- **vstar_var, spike_slab (r=0.06):** Both T types cluster at 0.15–0.45. Scores near the diagonal, no clear floor under Uniform T, no clear inflation under Deterministic T. vstar_var is T-insensitive in the observable range — not the expected behaviour.
- **vstar_var, gaussian (r=−0.02):** Same pattern. Both T types cluster at 0.2–0.4, no systematic difference. vstar_var shows no ability to distinguish T structure.

---

## 13 — S Sweep (`13_s_sweep.png`)

**What:** Two line plots (adv_gap, vstar_var) vs S ∈ {5, 10, 20, 50, 100} for three T types (Uniform, Dirichlet α=0.1, Deterministic). Gaussian R σ=1, A=4, γ=0.95, n=50.

**Observed patterns:**
- **adv_gap:** Uniform T (red) starts at ~1.0 and **stays high across all S** (~0.85 at S=100), remaining the highest line. Dirichlet(α=0.1) and Deterministic both decrease from ~1.0 to ~0.68–0.79. The predicted collapse of Uniform T does not occur. Root cause: under Uniform T with Gaussian R, both numerator (mean action gap) and denominator (range(V*−V^rand)) scale as O(σ/√S), so their ratio is S-invariant. The range-normalisation removes the signal that the raw metric would have shown (raw gap decreases as 1/√S under Uniform T).
- **vstar_var:** all three T types collapse together from ~0.6 (S=5) to ~0.15 (S=100) with overlapping confidence bands. No discrimination between T types at any S. The decrease is driven by Gaussian R dilution: as S grows, E_R[s,a] → 0 for all (s,a) and all T types, so V*≈V^rand everywhere regardless of topology. This is a sample-size artefact of fixed σ=1 Gaussian R, not a structural agenticity signal.
