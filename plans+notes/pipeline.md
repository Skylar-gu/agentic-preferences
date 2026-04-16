Research Pipeline
Reward Function Agenticity Classification

CORPUS — Sources to Read
Read the following sources in order of priority. For each, extract every explicitly stated reward function. Mark inferred formulas with [inferred].

Tier 1 — Classical RL Benchmarks
Reward formulas fully specified in source material.

1.	Sutton & Barto (2018), RL: An Introduction, Ch. 3-5
Gridworld (sparse terminal and step-penalty variants), cliff-walking, cart-pole. These are ground-truth tabular cases — use for anchor/calibration.
2.	Bellemare et al. (2013), The Arcade Learning Environment
Raw clipped game score across Atari games. Note which games are dense (Breakout: reward per brick) vs sparse (Montezuma's Revenge: reward only on key/room transitions). Reward structure varies wildly — treat each game as a separate entry.
3.	Tassa et al. (2020), dm_control: Software and Tasks for Continuous Control
DeepMind Control Suite. Read source code reward functions directly — they are fully transparent and use a structured tolerance() function with explicit sigma and margin parameters. Environments: walker/walk, cheetah/run, hopper/hop, reacher/easy, cartpole/balance.
4.	Gymnasium / OpenAI Gym documentation (current)
HalfCheetah, Ant, Hopper, Humanoid: explicit decomposition into forward velocity + control cost penalty + survival bonus. R = v_x - alpha|u|^2 + beta*1[alive]. Each component has different agenticity profile — treat composite and components separately.
5.	Ng, Harada, Russell (1999), Policy Invariance Under Reward Transformations
Canonical shaping examples. Also the theoretical basis for the invariance checks (Definitions D-E) you will apply later.

Tier 2 — RLHF and Learned Reward Functions

6.	Christiano et al. (2017), Deep RL from Human Preferences
Preference-model reward in continuous control domain. Reward is output of a neural network trained on human comparisons of trajectory segments — not a formula, but a learned function over trajectory windows.
7.	Ziegler et al. (2019), Fine-Tuning Language Models from Human Preferences
Scalar preference model on text completions. First application to language.
8.	Ouyang et al. (2022), InstructGPT
R = r_phi(x,y) - beta * log(pi(y|x) / pi_ref(y|x)). Critical: the KL term is a running per-token secondary reward, not just a constraint. This makes the composite reward structurally different from a pure outcome reward — flag this for Definition A.
9.	Lightman et al. (2023), Let's Verify Step by Step
Process reward model (per-step feedback) vs outcome reward model (terminal only). Structurally critical for your agenticity taxonomy — PRM and ORM are qualitatively different in temporality even when trained to judge the same underlying task.

Tier 3 — Verifiable / Rule-Based Rewards (2024-2025)

10.	Guo et al. (2025), DeepSeek-R1
Two reward types: (a) accuracy reward: R = 1[answer = gold], binary terminal; (b) format reward: R = 1[output in correct XML tags], also terminal. Both sparse and episodic. No neural reward model used by design — authors explicitly chose verifiable rewards to avoid reward hacking at scale.
11.	Lambert et al. (2024), RLVR
Verifiable rewards via code execution: R = 1[unit tests pass]. Pass@k structure. Reward is determined by a symbolic verifier (compiler), not a learned model.
12.	Shao et al. (2024), DeepSeekMath / GRPO
Same verifiable accuracy reward on math. Introduces group-relative baseline (GRPO): reward is normalised within a group of rollouts on the same prompt. Note: this affects the training algorithm, not the reward function itself — keep these separate in your analysis.

Tier 4 — AI Safety / Power-Seeking

13.	Turner et al. (2021), Optimal Policies Tend to Seek Power (NeurIPS)
CRITICAL — read carefully. Defines POWER(s) = average V*(s) across reward functions. Proves that most reward functions (measure-theoretic sense over the simplex) make power-seeking optimal under certain MDP graph symmetries. This is the closest existing formal work to what Joar's project targets. Turner's POWER is a property of states; your project characterises reward functions. Flag this distinction explicitly when applying Definition F.
14.	Turner & Tadepalli (2022), Parametrically Retargetable Decision-Makers Tend to Seek Power (NeurIPS)
Extends (13) beyond optimal policies to a broader class of decision-making procedures. More directly applicable to learned policies.
15.	Pan et al. (2023), Do the Rewards Justify the Means? MACHIAVELLI (ICML)
134 text-based environments with 500,000+ social decision-making scenarios. LLMs fine-tuned on game rewards empirically exhibit power-seeking and Machiavellian behaviour. Directly usable as an empirical test case for your definitions — check whether the game reward functions in MACHIAVELLI score as agentic under your framework.

Extraction Format
For each reward function found, record the following fields in a structured table.

Field	Notes
id	Short slug, e.g. mujoco_halfcheetah
source	Paper + year
domain	tabular / continuous-control / Atari / LLM-preference / LLM-verifiable / text-game
R_formula	Mathematical expression if explicit; mark [inferred] otherwise
temporality	immediate / terminal / episodic-mixed
sparsity	sparse / dense / mixed
composites	yes/no — is this reward a sum of sub-terms? If yes, list components

Classification Definitions
Apply all six definitions to every reward function in your corpus. Be explicit about your reasoning for each. Use a continuous [0,1] scale where 1 = maximally agentic.

Definition A — Q-Value Dominance Ratio
Estimate the ratio: alpha(R,T) = E[gamma * V(s')] / E[R(s,a,s') + gamma * V(s')]

•	If alpha ~ 1: future value dominates → agentic
•	If alpha ~ 0: immediate reward dominates → non-agentic

Since you cannot run simulations, reason qualitatively: does the bulk of the Q-value come from the immediate signal or from downstream states? Assign a score in [0,1] and explain your reasoning.
INVARIANCE WARNING: This measure is NOT invariant to potential shaping (per Joar's footnote). Flag this explicitly whenever it affects your classification. This is a known limitation of Definition A.

Definition B — Recovery Time (Policy Perturbation Sensitivity)
If the agent takes one random action instead of the optimal action, how many steps until expected discounted return recovers to near-optimal?

•	Short recovery (1-3 steps): non-agentic
•	Long recovery (many steps / never): agentic

Reason from the structure of the reward: does a single mistake compound? Is the state space forgiving? Does the reward function create narrow paths through state space that are hard to recover from?

Definition C — Sparsity
Compute: sparsity(R) = |{(s,a,s') : R(s,a,s') = 0}| / |S x A x S'|

•	High sparsity: reward signal is rare, agent must actively seek it → agentic
•	Low sparsity: agent receives constant feedback → non-agentic

KNOWN FAILURE MODE — THE LOTTERY PROBLEM: A random lottery is sparse (reward only on one outcome) but not agentic in any meaningful sense — the agent cannot plan to win it. Flag every application of this definition against this counterexample. Note whether the reward triggers this failure mode and explain whether it is a genuine false positive.

Definition D — State-Space Coverage Under Optimal Policy
Under the optimal policy pi*, what fraction of the state space is visited in expectation over trajectories?

•	High coverage (agent must range widely across states): agentic
•	Low coverage (agent stays near a fixed region or terminates quickly): non-agentic

Reason from the reward structure: does maximising this reward require the agent to actively expand its reach, or does it incentivise staying put / terminating?
KNOWN FAILURE MODE: Some non-agentic rewards (e.g. constant reward, random walk) also produce high coverage. Flag if applicable and explain whether coverage is driven by the reward or by the environment dynamics.

Definition E — Potential-Shaping Invariance Check
For each classification above, ask: if I add a potential function Phi(s) - gamma*Phi(s') to R, does my agenticity score change? It should not for a well-behaved definition.

For each definition, record:
•	invariant: yes / no / partially
•	If no: note which definition breaks and explain why

This is a formal sanity check, not a classification. Definitions that fail invariance are not wrong, but they are measuring something environment-entangled rather than intrinsic to R.
Do not skip this check. Joar's notes flag it explicitly. Definitions A and B are most likely to fail it.

Definition F — Turner POWER Alignment
Using Turner et al. (2021): POWER(s) = average V*(s) across reward functions, i.e. average over the simplex of all possible reward functions.

Ask: does maximising this reward function cause the optimal policy to navigate toward high-POWER states (states with many reachable future states, options preserved)? Or does it terminate / localise?

•	Navigates toward high-POWER states: agentic
•	Terminates or localises: non-agentic

CRITICAL DISTINCTION: Turner's POWER is a property of states, not reward functions. Your project characterises reward functions. You are asking whether a reward incentivises power-seeking, not whether it constitutes power. Flag this distinction explicitly in every application of Definition F.

Phase 3 — Structured Output Table
Produce a table with one row per reward function and the following columns:

Column	Description
id	Short slug
source	Paper + year
domain	Environment domain
R_formula	Mathematical formula
temporality	immediate / terminal / episodic-mixed
sparsity	sparse / dense / mixed
Def_A	Q-value dominance ratio [0,1]
Def_B	Recovery time sensitivity [0,1]
Def_C	Sparsity score [0,1]
Def_D	State-space coverage [0,1]
Def_F	Turner POWER alignment [0,1]
overall	Majority vote across A-D-F, with override if justified
inv_flag	Which definitions failed invariance check (Definition E)
confidence	[low / medium / high] for overall classification
notes	Any overrides, failure modes triggered, or caveats

For overall_agentic: use majority vote across A, D, F (skip C if lottery problem applies, skip A if shaping non-invariance invalidates it). Override is allowed but must be stated with explicit justification in the notes column.

Phase 4 — Meta-Analysis
After completing the table, answer the following questions in prose (1-3 paragraphs each).

Question 1 — Empirical Base Rate
Of the reward functions in your corpus, what fraction score above 0.5 on overall agenticity? Does this vary by domain (tabular vs continuous control vs LLM-preference vs LLM-verifiable)?
Connect your finding to Turner et al.'s theoretical result that most reward functions (in a measure-theoretic sense) make power-seeking optimal. Does your empirical base rate match, exceed, or fall below what Turner's theorem would predict? Explain any discrepancy.

Question 2 — Definition Disagreements
Where do the definitions most sharply disagree? Which pairs of definitions most often give conflicting scores? What does each disagreement reveal about what that definition is actually measuring, as opposed to what it was intended to measure?

Question 3 — False Positives / False Negatives
Identify: (a) the clearest false positive — a reward that scores high but intuitively seems non-agentic; (b) the clearest false negative — a reward that scores low but intuitively seems agentic. Propose a minimal modification to the offending definition that corrects the case without breaking others.

Question 4 — Invariance Failures
Which definitions failed the potential-shaping invariance check (Definition E), and for which reward functions? What does this imply for the use of these definitions in a regularisation context — e.g., penalising agentic rewards during RLHF training?

Question 5 — Proposed Refinement
Based on your full analysis, propose one refined or composite definition that you believe best captures agenticity across the corpus. State it mathematically if possible. Identify its remaining weaknesses and the cases it still fails.

Question 6 — Relationship to Turner et al.
Turner proves that most reward functions make power-seeking optimal. Does this imply that most reward functions are agentic under your definitions? If not, explain what your definition captures that Turner's framework does not, or vice versa. Is your definition a refinement, a weakening, or an orthogonal characterisation of the same underlying concept?

Question 7 — Verifiable Rewards as a Special Class
DeepSeek-R1 and RLVR use purely terminal, binary, sparse rewards (correct/incorrect). By Definition C, these score as highly agentic. But intuitively, 'answer this maths problem correctly' feels like a bounded, non-agentic objective. Is this a genuine failure of Definition C, or is the intuition wrong? Argue both sides, then take a position.

Question 8 — Compositionality of Agenticity
The KL penalty in InstructGPT's reward acts as a dense, per-token regulariser penalising deviation from the reference policy. Does this term make the composite RLHF reward more or less agentic? What does this reveal about whether agenticity is compositionally analysable — i.e., is the agenticity of R_1 + R_2 a function of the agenticity of R_1 and R_2 separately? Provide a formal argument or counterexample.

Constraints and Conduct Rules
Strictly follow these rules throughout the pipeline.

16.	Never hallucinate formulas. If a paper does not state R explicitly, say so and use the closest explicit approximation with a [inferred] flag. Do not invent formulas.
17.	Be explicit about uncertainty. Use [low confidence], [medium confidence], [high confidence] tags on every overall classification.
18.	Do not collapse definitions prematurely. Even if two definitions seem to agree on most cases, keep them separate until Phase 4.
19.	Flag the lottery counterexample whenever Definition C (sparsity) is applied. It is a known failure mode and must be explicitly addressed for every sparse reward.
20.	Shaping invariance (Definition E) is non-negotiable. Do not skip it for any reward function.
21.	Keep hand-specified and learned reward functions in separate analytical categories. Gridworld and MuJoCo rewards are clean test cases for formal definitions. RLHF reward models introduce additional complexity from misspecification (see Skalse & Abate 2023) — treat them as a separate stratum.
22.	For composite rewards (e.g. InstructGPT KL penalty, MuJoCo multi-term rewards), classify each component separately before classifying the composite.

Deliverable Format
Return the following, in order:

23.	The structured extraction table (Phase 3) — one row per reward function, all columns filled
24.	The meta-analysis (Phase 4) — eight numbered prose sections, 1-3 paragraphs each
25.	A final one-paragraph recommendation for which definition(s) Joar's project should carry forward into formal axiomatic development, with explicit justification and remaining open questions

Caveats for the Researcher
These are honest limitations to bear in mind when interpreting the LLM's output.

What This Pipeline Will Do Well
•	Force systematic cross-domain comparison across a curated corpus
•	Surface the lottery/sparsity problem immediately and consistently
•	Flag invariance failures that reveal definition weaknesses
•	Produce a structured dataset of hypotheses to test computationally

What Will Be Difficult
Definitions A and B require running actual MDPs to compute rigorously. The LLM will reason qualitatively, which introduces noise. Treat Phase 3 scores as hypotheses to verify computationally, not as ground truth.
Learned reward functions (RLHF) are not closed-form — the LLM can only reason about their structural properties (outcome vs process, trajectory vs per-step), not their actual numerical values. Flag all RLHF classifications as [low confidence] unless structural reasoning is clearly sufficient.

Recommended Follow-Up
Once you have the table, run actual Q-value computations on the tabular cases (gridworld, cliff-walk) where this is tractable. These become your ground-truth anchors for calibrating the qualitative reasoning (on which definition it’s logical to rely on depending on the environment) on the harder continuous-control and LLM cases.
The MACHIAVELLI benchmark (Pan et al. 2023) provides pre-labelled power-seeking behaviour in text-based environments. Use it as an external validation set — check whether the reward functions in MACHIAVELLI that empirically produce power-seeking also score as agentic under your definitions.
