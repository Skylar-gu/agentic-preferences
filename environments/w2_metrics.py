"""
w2_metrics.py — Week 2 shaping-invariant metrics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core import MDP, value_iteration, policy_evaluation, discounted_occupancy


# ---------------------------------------------------------------------------
# Week 2 Metrics (shaping-invariant, use d0)
# ---------------------------------------------------------------------------

def control_advantage(mdp: MDP, pi_baseline):
    """A_ctrl = E_{s~d0}[ V*(s) - V^{pi0}(s) ]."""
    V_star, _, _ = value_iteration(mdp)
    V0 = policy_evaluation(mdp, pi_baseline)
    return float(mdp.d0 @ (V_star - V0))


def one_step_recovery(mdp: MDP, pi_baseline):
    """
    RecAdv = E_{s~d*} E_{a~Uniform(A)} E_{s'~T(s,a,·)}[ V*(s') - V^{pi0}(s') ].
    Random action taken from s, then recovery advantage measured at landing state s'.
    Invariant: both values shift by -Phi(s'), which cancels.
    """
    _, _, pi_star = value_iteration(mdp)
    d_star = discounted_occupancy(mdp, pi_star)

    V_star = policy_evaluation(mdp, pi_star)
    V0 = policy_evaluation(mdp, pi_baseline)

    # uniform random action kernel: P(s'|s) = (1/A) * sum_a T(s,a,s')
    rand_kernel = mdp.T.mean(axis=1)  # (S, S)
    dist_S0 = d_star @ rand_kernel
    return float(dist_S0 @ (V_star - V0))
