"""
01_sweep.py — MCE entropy alpha calibration sweep.

Samples N_T canonical T matrices once (fixed seeds), then sweeps N_R reward
matrices across each T for every (R_type, alpha) combination.

This gives:
  - Cleaner distribution shapes (~500 samples per cell)
  - Direct T-sensitivity check (do histograms shift between T₁…T₄?)
  - Separation of T-variance from R-variance in MCE entropy scores

Usage:
  python 01_sweep.py           # full run (~3 min)
  python 01_sweep.py --fast    # smoke test (~15 sec)

Output: results/mce_alpha_sweep.pkl
"""
import sys, os, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

import numpy as np
from dataclasses import replace as dc_replace
from core import MDP, soft_value_iteration
from runners import random_mdp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

S, A, GAMMA, K = 10, 4, 0.95, 3

R_TYPES  = ['gaussian', 'uniform', 'bernoulli', 'spike_slab', 'goal', 'potential']
R_SCALES = {'bernoulli': 0.5, 'spike_slab': 0.1}  # overrides; others default to 1.0

ALPHAS   = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


# ---------------------------------------------------------------------------
# MCE helper (normalised rewards, variable alpha)
# ---------------------------------------------------------------------------

def _mce_norm(mdp: MDP, alpha: float) -> float:
    """1 - H(pi_MCE)/log(A)  with reward normalisation. High = agentic."""
    r_std = float(mdp.R.std())
    mdp_vi = dc_replace(mdp, R=mdp.R / r_std) if r_std > 1e-8 else mdp
    _, _, pi_mce = soft_value_iteration(mdp_vi, alpha=alpha)

    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return float('nan')

    eps = 1e-12
    H_s = -np.sum(pi_mce * np.log(pi_mce + eps), axis=1)
    d0_nt = mdp.d0[non_terminal]
    d0_nt = d0_nt / d0_nt.sum()
    H_mean = float(np.dot(d0_nt, H_s[non_terminal]))
    return float(np.clip(1.0 - H_mean / np.log(A), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    N_T = 3 if args.fast else 5   # canonical T matrices
    N_R = 20 if args.fast else 500  # R samples per (T, R_type)

    print(f"MCE alpha calibration sweep  (N_T={N_T}, N_R={N_R}, fast={args.fast})")
    print(f"  S={S}, A={A}, gamma={GAMMA}, k={K}")
    print(f"  R_types: {R_TYPES}")
    print(f"  Alphas:  {ALPHAS}")

    master_rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # 1. Sample N_T canonical T matrices (fixed once)
    # ------------------------------------------------------------------
    terminal = set(range(S - 1, S))
    non_terminal = [s for s in range(S) if s not in terminal]
    d0_base = np.zeros(S)
    d0_base[non_terminal] = 1.0 / len(non_terminal)

    print(f"\nSampling {N_T} canonical T matrices...")
    T_matrices = []
    T_seeds = []
    for i in range(N_T):
        seed = int(master_rng.integers(0, 2**31))
        T_seeds.append(seed)
        proto = random_mdp(S, A, gamma=GAMMA, k=K, R_type='gaussian',
                           terminal_states=1,
                           rng=np.random.default_rng(seed))
        T_matrices.append(proto.T.copy())
        print(f"  T{i+1} (seed={seed}): sampled")

    # ------------------------------------------------------------------
    # 2. Sweep (T, R_type, alpha)
    # ------------------------------------------------------------------
    # data[T_idx][R_type][alpha] = np.array of N_R scores
    data = {i: {rt: {a: [] for a in ALPHAS} for rt in R_TYPES}
            for i in range(N_T)}

    total_cells = N_T * len(R_TYPES)
    cell = 0
    for t_idx, T_mat in enumerate(T_matrices):
        for rt in R_TYPES:
            cell += 1
            rs = R_SCALES.get(rt, 1.0)
            print(f"  [{cell}/{total_cells}] T{t_idx+1} × {rt}...", end='', flush=True)

            for _ in range(N_R):
                r_seed = int(master_rng.integers(0, 2**31))
                # Sample R only (T is fixed)
                proto = random_mdp(S, A, gamma=GAMMA, k=K,
                                   R_type=rt, R_scale=rs,
                                   terminal_states=1,
                                   rng=np.random.default_rng(r_seed))
                mdp = MDP(S=S, A=A, T=T_mat, R=proto.R, gamma=GAMMA,
                          terminal=terminal, d0=d0_base.copy())

                for alpha in ALPHAS:
                    data[t_idx][rt][alpha].append(_mce_norm(mdp, alpha))

            # Convert to arrays
            for alpha in ALPHAS:
                data[t_idx][rt][alpha] = np.array(data[t_idx][rt][alpha])

            # Quick summary at alpha=0.25
            scores_025 = data[t_idx][rt][0.25]
            print(f"  mean@0.25={scores_025.mean():.3f} ±{scores_025.std():.3f}")

    # ------------------------------------------------------------------
    # 3. Between-type variance summary
    # ------------------------------------------------------------------
    print("\nBetween-type variance (non-potential) per alpha per T:")
    non_pot = [rt for rt in R_TYPES if rt != 'potential']
    hdr = f"  {'':8s}" + "".join(f"  a={a:<5g}" for a in ALPHAS)
    print(hdr)
    for t_idx in range(N_T):
        row = []
        for alpha in ALPHAS:
            type_means = [data[t_idx][rt][alpha].mean() for rt in non_pot]
            row.append(f"{np.var(type_means):.4f}")
        print(f"  T{t_idx+1}      " + "".join(f"  {v:9s}" for v in row))

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out = {
        'data':       data,
        'T_matrices': T_matrices,
        'T_seeds':    T_seeds,
        'params': {
            'S': S, 'A': A, 'gamma': GAMMA, 'k': K,
            'N_T': N_T, 'N_R': N_R,
            'R_types': R_TYPES, 'R_scales': R_SCALES,
            'alphas': ALPHAS,
        },
    }
    out_path = os.path.join(_SCRIPT_DIR, 'results', 'mce_alpha_sweep.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved → {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
