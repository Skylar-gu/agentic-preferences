"""
Finite MDP Lab — entry point.
Import everything from submodules; run as `python environments/mdp.py` for demo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import *         # noqa: F401,F403
from w2_metrics import *   # noqa: F401,F403
from pams import *         # noqa: F401,F403
from envs import *         # noqa: F401,F403
from runners import *      # noqa: F401,F403

if __name__ == '__main__':
    import numpy as np

    rng = np.random.default_rng(42)

    # --- Week 2: shaping invariance verification ---
    print("=" * 70)
    print("WEEK 2 — SHAPING INVARIANCE VERIFICATION (5x5 gridworld, slip=0.1)")
    print("=" * 70)

    gw = gridworld(5, 5, goal_xy=(4, 4), step_cost=-1.0, slip=0.1, gamma=0.99)
    pi0 = np.ones((gw.S, gw.A)) / gw.A

    Actrl = control_advantage(gw, pi0)
    R1 = one_step_recovery(gw, pi0)
    print(f"  A_ctrl = {Actrl:.6f}")
    print(f"  R1     = {R1:.6f}")

    Phi = np.random.default_rng(0).normal(size=gw.S)
    gw2 = add_potential_shaping(gw, Phi)
    Actrl2 = control_advantage(gw2, pi0)
    R1_2 = one_step_recovery(gw2, pi0)
    print(f"  diff A_ctrl = {Actrl2 - Actrl:.2e}  (should be ~0)")
    print(f"  diff R1     = {R1_2 - R1:.2e}  (should be ~0)")

    # --- Week 3: agenticity proxies ---
    print("\n" + "=" * 70)
    print("WEEK 3 — AGENTICITY PROXIES (shaping-invariant)")
    print("=" * 70)

    cases = [
        ("Chain — Terminal",   make_chain_mdp(10, reward_type='terminal')),
        ("Chain — Dense",      make_chain_mdp(10, reward_type='dense')),
        ("Chain — Lottery",    make_chain_mdp(10, reward_type='lottery')),
        ("Chain — Progress",   make_chain_mdp(10, reward_type='progress')),
        ("Grid  — Goal",      make_grid_mdp(5, 5, reward_type='goal')),
        ("Grid  — Local",     make_grid_mdp(5, 5, reward_type='local')),
        ("Grid  — Cliff",     make_grid_mdp(5, 5, reward_type='cliff')),
    ]

    summary = []
    for name, mdp in cases:
        print(f"\n>>> {name}")
        r = agenticity_score(mdp, verbose=True, rng=rng)
        summary.append((name, r['composite'], r['adv_gap_norm'],
                        r['vstar_var_norm'], r['mi_diff'], r['adv_sparsity']))

    print("\n" + "=" * 70)
    print("SUMMARY (descending composite)")
    print("=" * 70)
    print(f"{'MDP':<25} {'Comp':>6} {'AdvGp':>6} {'V*var':>6} {'MIdif':>6} {'ASpar':>6}")
    print("-" * 70)
    for row in sorted(summary, key=lambda x: -x[1]):
        name, comp, ag, vv, mi, asp = row
        print(f"{name:<25} {comp:>6.4f} {ag:>6.4f} {vv:>6.4f} {mi:>+6.4f} {asp:>6.4f}")

    # --- MCE demo ---
    print("\n" + "=" * 70)
    print("MCE DEMO — Grid-Cliff")
    print("=" * 70)

    cliff_mdp = make_grid_mdp(5, 5, reward_type='cliff')
    _, _, pi_hard = value_iteration(cliff_mdp)

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        V_soft, Q_soft, pi_mce = soft_value_iteration(cliff_mdp, alpha=alpha)
        J = mce_objective(cliff_mdp, Q_soft, alpha=alpha)
        hard_onehot = np.zeros_like(pi_mce)
        hard_onehot[np.arange(cliff_mdp.S), pi_hard] = 1.0
        dist = np.abs(pi_mce - hard_onehot).sum(axis=1).mean()
        print(f"  alpha={alpha:<6g}  J_MCE={J:+.4f}  L1(pi_mce,pi*)={dist:.4f}")

    print("  (alpha->0: hard optimal | alpha->inf: uniform)")

    # --- Week 4: PAM experiments (Q1/Q2/Q3) ---
    print("\n" + "=" * 70)
    print("WEEK 4 — PAM BATCH EXPERIMENT (random MDPs, n=50 per cell)")
    print("=" * 70)
    run_pam_experiment(
        n_random_mdps=50,
        S_values=[5, 10],
        A=4,
        gamma=0.95,
        k=3,
        R_types=['gaussian', 'uniform', 'bernoulli'],
        n_fixed_T=5,
        rng_seed=42,
        verbose=True,
        include_mi=False,
    )

    print("\n" + "=" * 70)
    print("MCE DEMO — Chain-Dense")

    print("=" * 70)

    dense_mdp = make_chain_mdp(10, reward_type='dense')
    _, _, pi_hard_dense = value_iteration(dense_mdp)

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        V_soft, Q_soft, pi_mce = soft_value_iteration(dense_mdp, alpha=alpha)
        J = mce_objective(dense_mdp, Q_soft, alpha=alpha)
        hard_onehot = np.zeros_like(pi_mce)
        hard_onehot[np.arange(dense_mdp.S), pi_hard_dense] = 1.0
        dist = np.abs(pi_mce - hard_onehot).sum(axis=1).mean()
        print(f"  alpha={alpha:<6g}  J_MCE={J:+.4f}  L1(pi_mce,pi*)={dist:.4f}")

    print("  (alpha->0: hard optimal | alpha->inf: uniform)")
