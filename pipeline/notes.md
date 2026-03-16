



2. The PRM/ORM counterexample (Lightman 2023) involves two reward function, one dense per-step (PRM), one sparse terminal (ORM) which produce near-identical optimal behaviour on math tasks. If they get different agenticity scores, we need an explaination of why two rewards inducing the same optimal policy should differ in agenticity. This leads to the question: should agenticity be policy-invariant? The counterexample points toward no, agenticity is a property of the reward structure, not the induced policy, but this needs to be argued more explicitly


 The KL penalty looks like potential-based shaping, but it isn't — the reference policy π_ref is fixed while the policy π changes during training, making the KL term non-stationary
 - so we dont have the invariance result where set of optimal policies is unchanged, for any environment dynamics

 Test on Machievelli:  134 environments with ground-truth observations of power-seeking behaviour.


For RLHF: 
If the true human preference is non-Markovian but the learned reward is forced Markovian, then the reward function you're trying to classify for agenticity is already a *lossy projection* of the actual objective. Whether we should classify the learned rθ​ or the underlying human preference is a choice and they could differ in agenticity.


Some interesting counterexamples: 
For chess or GO: if the reward function is 1_{win game}, is this function agentic? Or is the environment doing too much heavy lifting by enforcing things like planning because in a different environment this would not be agentic 

