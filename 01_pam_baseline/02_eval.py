"""
02_eval.py — Load Q1/Q2/Q3 results and print summary tables.

Usage:  python 02_eval.py
Requires: results/pam_experiment.pkl  (from 01_build.py)
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

from runners import _print_pam_results


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', 'pam_experiment.pkl')
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run 01_build.py first.")
        sys.exit(1)

    with open(path, 'rb') as f:
        results = pickle.load(f)

    meta = results.get('meta', {})
    print(f"Loaded: S_values={meta.get('S_values')}, "
          f"n={meta.get('n_random_mdps')}, "
          f"MI={'included' if meta.get('include_mi') else 'excluded'}")

    _print_pam_results(results)


if __name__ == '__main__':
    main()
