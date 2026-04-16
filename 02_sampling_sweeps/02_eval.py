"""
02_eval.py — Load sweep results and print per-group summary statistics.

Usage:  python 02_eval.py
Requires: results/sweep_results.pkl  (from 01_build.py)
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

import numpy as np

METRICS = [
    ('adv_gap_norm',    'Advantage Gap (norm)'),
    ('vstar_var_norm',  'V*−V^rand Variance (norm)'),
    ('H_eps_norm',      'Planning Horizon H_eps (norm)'),
    ('mce_entropy_norm','MCE Entropy (1−norm)'),
]
HIGH_AGENCY_THRESHOLD = 0.5


def _summarise(score_dicts, label=''):
    """Print mean ± std and % high-agency for each metric."""
    for key, title in METRICS:
        vals = np.array([r[key] for r in score_dicts if r.get(key) is not None])
        if len(vals) == 0:
            continue
        frac = (vals >= HIGH_AGENCY_THRESHOLD).mean()
        print(f"    {title:<38s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"high={frac:.0%}")


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', 'sweep_results.pkl')
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run 01_build.py first.")
        sys.exit(1)

    with open(path, 'rb') as f:
        results = pickle.load(f)

    p_values    = results.get('p_values', [])
    n_mdps      = results.get('n_mdps', '?')
    bern_data   = results.get('bern_data', {})
    ss_data     = results.get('ss_data', {})
    gamma_data  = results.get('gamma_data', {})
    gamma_p     = results.get('gamma_params', {})
    t_data      = results.get('t_data', {})
    t_params    = results.get('t_params', {})
    s_data      = results.get('s_data', {})
    s_params    = results.get('s_params', {})

    sep = '=' * 70

    # ------------------------------------------------------------------
    # Group 1: p-sweep
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("GROUP 1 — Bernoulli p-sweep  (S=10, T fixed)")
    print(sep)
    for p in p_values:
        scores = bern_data.get(p, [])
        print(f"  p={p}  (n={len(scores)})")
        _summarise(scores)

    print(f"\n{sep}")
    print("GROUP 1 — Spike-and-slab p-sweep  (S=10, T fixed)")
    print(sep)
    for p in p_values:
        scores = ss_data.get(p, [])
        print(f"  p={p}  (n={len(scores)})")
        _summarise(scores)

    # ------------------------------------------------------------------
    # Group 3a: γ sweep
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("GROUP 3a — γ sweep")
    print(sep)
    for (R_type, R_scale, gamma), scores in sorted(gamma_data.items(),
                                                    key=lambda x: (x[0][0], x[0][2])):
        print(f"  R={R_type}(scale={R_scale}), γ={gamma}  (n={len(scores)})")
        _summarise(scores)

    # ------------------------------------------------------------------
    # Group 2: T-sensitivity
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("GROUP 2 — T-sensitivity (same R scored under multiple T structures)")
    print(sep)
    T_types = t_params.get('T_types', [])
    for (R_type, R_scale), paired in t_data.items():
        print(f"\n  R={R_type}(scale={R_scale})")
        for T_type in T_types:
            scores = paired.get(T_type, [])
            print(f"    T_type={T_type}  (n={len(scores)})")
            _summarise(scores)

        # Pearson correlation between T types for adv_gap
        if len(T_types) >= 2:
            print(f"    Pearson r (adv_gap_norm) between T types:")
            for i, t1 in enumerate(T_types):
                for t2 in T_types[i+1:]:
                    x = np.array([r['adv_gap_norm'] for r in paired.get(t1, [])])
                    y = np.array([r['adv_gap_norm'] for r in paired.get(t2, [])])
                    if len(x) > 1 and len(y) > 1:
                        r_val = float(np.corrcoef(x, y)[0, 1])
                        print(f"      {t1} vs {t2}: r={r_val:.3f}")

    # ------------------------------------------------------------------
    # Group 3b: S sweep
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("GROUP 3b — S sweep  (Gaussian R σ=1, A=4, γ=0.95)")
    print(sep)
    S_values = s_params.get('S_values', [])
    T_types_s = s_params.get('T_types', [])
    for T_type in T_types_s:
        print(f"\n  T_type={T_type}")
        for S in S_values:
            scores = s_data.get((S, T_type), [])
            means = {k: np.mean([r[k] for r in scores]) for k, _ in METRICS}
            print(f"    S={S:>4d}  adv_gap={means['adv_gap_norm']:.3f}  "
                  f"vstar_var={means['vstar_var_norm']:.3f}  "
                  f"H_eps={means['H_eps_norm']:.3f}")


if __name__ == '__main__':
    main()
