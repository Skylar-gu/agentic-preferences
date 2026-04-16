"""
01_build.py — Run Q1/Q2/Q3 PAM experiment and R-scale sweep. Save all results to disk.

Usage:
  python 01_build.py            # full run (~2 min)
  python 01_build.py --fast     # smoke test (~10 sec)

Output: results/pam_experiment.pkl
"""
import sys, os, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

import numpy as np
from runners import run_pam_experiment, random_mdp
from core import MDP
from pams import agenticity_score


def _run_rscale_sweep(n_mdps: int, rng_seed: int = 99) -> dict:
    """Fix T, sweep Gaussian R_scale ∈ {0.1, 0.5, 1.0, 2.0, 5.0}. Returns {scale: (n, 5) array}."""
    R_SCALES = [0.1, 0.5, 1.0, 2.0, 5.0]
    S, A, gamma = 10, 4, 0.95
    rng = np.random.default_rng(rng_seed)

    canonical = random_mdp(S, A, gamma=gamma, k=3, R_type='gaussian',
                           terminal_states=1, rng=np.random.default_rng(7))
    T_fixed, term_fixed, d0_fixed = canonical.T, canonical.terminal, canonical.d0

    scale_data = {}
    for R_scale in R_SCALES:
        rows = []
        for _ in range(n_mdps):
            tmp = random_mdp(S, A, gamma=gamma, k=3, R_type='gaussian',
                             R_scale=R_scale, terminal_states=1,
                             rng=np.random.default_rng(int(rng.integers(0, 2**31))))
            mdp = MDP(S=S, A=A, T=T_fixed, R=tmp.R, gamma=gamma,
                      terminal=term_fixed, d0=d0_fixed.copy())
            r = agenticity_score(mdp, verbose=False, compute_mi=False,
                                 rng=np.random.default_rng(int(rng.integers(0, 2**31))))
            rows.append([r['adv_gap_norm'], r['vstar_var_norm'],
                         r['mi_diff']          if r['mi_diff']          is not None else float('nan'),
                         r['H_eps_norm'],
                         r['mce_entropy_norm'] if r['mce_entropy_norm'] is not None else float('nan')])
        scale_data[R_scale] = np.array(rows)
    return scale_data


def main():
    parser = argparse.ArgumentParser(description='Run Q1/Q2/Q3 PAM experiment.')
    parser.add_argument('--fast', action='store_true',
                        help='Smoke test: small n, fewer R_types and T_structures.')
    args = parser.parse_args()

    if args.fast:
        exp_kwargs = dict(
            n_random_mdps=5,
            S_values=[5],
            A=4, gamma=0.95, k=3,
            R_types=['gaussian', 'potential'],
            n_fixed_T=2,
            T_structures=['dirichlet_1.0'],
            rng_seed=42,
            verbose=False,
            include_mi=False,
        )
        n_scale = 5
        print("Running in --fast mode (smoke test).")
    else:
        exp_kwargs = dict(
            n_random_mdps=50,
            S_values=[5, 10],
            A=4, gamma=0.95, k=3,
            R_types=['gaussian', 'uniform', 'bernoulli', 'potential', 'goal'],
            n_fixed_T=5,
            T_structures=['dirichlet_0.1', 'dirichlet_1.0', 'dirichlet_10.0'],
            rng_seed=42,
            verbose=False,
            include_mi=False,
        )
        n_scale = 50

    print("Running PAM experiment (Q1/Q2/Q3)...")
    results = run_pam_experiment(**exp_kwargs)
    print(f"  Q1: {len(results['q1'])} cells")
    print(f"  Q2: {len(results['q2'])} cells")
    print(f"  Q3: {len(results['q3'])} MDPs")

    print("Running R-scale sweep (for Plot 08)...")
    results['rscale_sweep'] = _run_rscale_sweep(n_mdps=n_scale)
    print(f"  Scales: {sorted(results['rscale_sweep'].keys())}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'pam_experiment.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    size_kb = os.path.getsize(out) / 1024
    print(f"Saved → {out}  ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
