"""
01_build.py — Run Group 1/2/3 sampling sweep experiments. Save all results to disk.

Groups:
  Group 1 (plots 09-10): reward sparsity p-sweep for Bernoulli and spike-and-slab R
  Group 2 (plot 12):     T-sensitivity — same R scored under Uniform vs Deterministic T
  Group 3a (plot 11):    γ sweep — PAM scores vs discount factor
  Group 3b (plot 13):    S sweep — PAM scores vs state-space size × T type

Usage:
  python 01_build.py            # full run (~3-5 min)
  python 01_build.py --fast     # smoke test (~15 sec)

Output: results/sweep_results.pkl
"""
import sys, os, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

from runners import run_p_sweep, run_gamma_sweep, run_t_sensitivity, run_s_sweep


def main():
    parser = argparse.ArgumentParser(description='Run Group 1/2/3 sampling sweep experiments.')
    parser.add_argument('--fast', action='store_true',
                        help='Smoke test: small n and fewer parameter values.')
    args = parser.parse_args()

    if args.fast:
        P_VALUES       = [0.05, 0.5]
        GAMMAS         = [0.7, 0.95]
        R_CONDITIONS   = [('spike_slab', 0.1)]
        N_MDPS         = 5
        N_R            = 10
        S_VALUES       = [5, 10]
        T_TYPES_SENS   = ['uniform', 'deterministic']
        T_TYPES_SWEEP  = ['uniform', 'dirichlet']
        print("Running in --fast mode (smoke test).")
    else:
        P_VALUES       = [0.01, 0.05, 0.2, 0.5, 1.0]
        GAMMAS         = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        R_CONDITIONS   = [('spike_slab', 0.1), ('gaussian', 1.0)]
        N_MDPS         = 50
        N_R            = 100
        S_VALUES       = [5, 10, 20, 50, 100]
        T_TYPES_SENS   = ['uniform', 'dirichlet', 'deterministic']
        T_TYPES_SWEEP  = ['uniform', 'dirichlet', 'deterministic']

    results = {}

    # ------------------------------------------------------------------
    # Group 1: p-sweep — Bernoulli (plot 09)
    # ------------------------------------------------------------------
    print("Group 1a: Bernoulli p-sweep...")
    results['bern_data'] = run_p_sweep('bernoulli', P_VALUES, n_mdps=N_MDPS)
    print(f"  p values: {P_VALUES}")

    # ------------------------------------------------------------------
    # Group 1: p-sweep — spike-and-slab (plot 10)
    # ------------------------------------------------------------------
    print("Group 1b: spike-and-slab p-sweep...")
    results['ss_data'] = run_p_sweep('spike_slab', P_VALUES, n_mdps=N_MDPS)

    # ------------------------------------------------------------------
    # Group 3a: γ sweep (plot 11)
    # ------------------------------------------------------------------
    print("Group 3a: γ sweep...")
    results['gamma_data']    = run_gamma_sweep(gammas=GAMMAS, R_conditions=R_CONDITIONS,
                                               n_mdps=N_MDPS)
    results['gamma_params']  = {'gammas': GAMMAS, 'R_conditions': R_CONDITIONS}
    print(f"  γ values: {GAMMAS}")

    # ------------------------------------------------------------------
    # Group 2: T-sensitivity (plot 12)
    # ------------------------------------------------------------------
    print("Group 2: T-sensitivity...")
    results['t_data']       = run_t_sensitivity(n_R=N_R, R_conditions=R_CONDITIONS,
                                                T_types=T_TYPES_SENS)
    results['t_params']     = {'T_types': T_TYPES_SENS, 'R_conditions': R_CONDITIONS}
    print(f"  T types: {T_TYPES_SENS}, n_R={N_R}")

    # ------------------------------------------------------------------
    # Group 3b: S sweep (plot 13)
    # ------------------------------------------------------------------
    print("Group 3b: S sweep...")
    results['s_data']       = run_s_sweep(S_values=S_VALUES, T_types=T_TYPES_SWEEP,
                                          n_mdps=N_MDPS)
    results['s_params']     = {'S_values': S_VALUES, 'T_types': T_TYPES_SWEEP}
    print(f"  S values: {S_VALUES}")

    results['p_values']     = P_VALUES
    results['n_mdps']       = N_MDPS

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'sweep_results.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    size_kb = os.path.getsize(out) / 1024
    print(f"\nSaved → {out}  ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
