"""
03_plot.py — Generate sampling sweep figures 09–13 from pre-computed results.

Usage:  python 03_plot.py
Requires: results/sweep_results.pkl  (from 01_build.py)
Saves:    figures/09_bernoulli_p_sweep.pdf ... figures/13_s_sweep.pdf
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

HIGH_AGENCY_THRESHOLD = 0.5
VIRIDIS = plt.cm.viridis

METRICS = [
    ('adv_gap_norm',   'Advantage Gap (norm)'),
    ('vstar_var_norm', 'V*−V^rand Variance (norm)'),
    ('H_eps_norm',     'Planning Horizon H_eps (norm)'),
]


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


def plot_p_sweep(data, p_values, title, filename):
    colors = [VIRIDIS(i / max(len(p_values) - 1, 1)) for i in range(len(p_values))]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    for ax, (key, metric_title) in zip(axes, METRICS):
        for p, color in zip(p_values, colors):
            vals = np.array([r[key] for r in data.get(p, [])])
            if len(vals) == 0:
                continue
            frac = (vals >= HIGH_AGENCY_THRESHOLD).mean()
            ax.hist(vals, bins=20, range=(0, 1), alpha=0.45, color=color, density=True,
                    label=f'p={p} ({frac:.0%})')
        ax.axvline(HIGH_AGENCY_THRESHOLD, color='black', lw=1.4, ls='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalised score', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(metric_title, fontsize=9)
        ax.legend(fontsize=7, title='p (% high)', title_fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, filename)


def plot_09_bernoulli(bern_data, p_values):
    print("Plot 09: Bernoulli p-sweep")
    plot_p_sweep(
        bern_data, p_values,
        'Group 1: Bernoulli p sweep  (S=10, T fixed)',
        '09_bernoulli_p_sweep.pdf',
    )


def plot_10_spike_slab(ss_data, p_values):
    print("Plot 10: spike-and-slab p-sweep")
    plot_p_sweep(
        ss_data, p_values,
        'Group 1: spike-and-slab p sweep  (S=10, T fixed)',
        '10_spike_slab_p_sweep.pdf',
    )


def plot_11_gamma_sweep(gamma_data, gammas, R_conditions):
    print("Plot 11: γ sweep")
    R_COLORS = {'spike_slab': '#4C72B0', 'gaussian': '#DD8452'}
    R_LABELS = {'spike_slab': 'spike_slab (p=0.1)', 'gaussian': 'gaussian (σ=1)'}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    for ax, (key, metric_title) in zip(axes, METRICS):
        for R_type, R_scale in R_conditions:
            means = []
            stds  = []
            for g in gammas:
                scores = gamma_data.get((R_type, R_scale, g), [])
                vals   = np.array([r[key] for r in scores])
                means.append(vals.mean() if len(vals) > 0 else float('nan'))
                stds.append(vals.std()   if len(vals) > 0 else float('nan'))
            means = np.array(means)
            stds  = np.array(stds)
            color = R_COLORS.get(R_type, '#555')
            ax.plot(gammas, means, 'o-', color=color,
                    label=R_LABELS.get(R_type, R_type))
            ax.fill_between(gammas, means - stds, means + stds, color=color, alpha=0.15)
        ax.axhline(HIGH_AGENCY_THRESHOLD, color='grey', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel('γ', fontsize=9)
        ax.set_ylabel('Mean score (±1 std)', fontsize=9)
        ax.set_title(metric_title, fontsize=9)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)

    fig.suptitle('Group 3a: PAM scores vs γ  (S=20, A=4, T=random)', fontsize=10)
    fig.tight_layout()
    save(fig, '11_gamma_sweep.pdf')


def plot_12_t_sensitivity(t_data, R_conditions, T_types):
    print("Plot 12: T-sensitivity")
    T_PAIR = (T_types[0], T_types[-1])   # most extreme comparison
    T_COLORS_R = {'gaussian': '#4C72B0', 'spike_slab': '#55A868'}
    S_METRICS_T = METRICS[:2]

    n_rows = len(R_conditions)
    n_cols = len(S_METRICS_T)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for row, (R_type, R_scale) in enumerate(R_conditions):
        paired = t_data.get((R_type, R_scale), {})
        for col, (key, metric_title) in enumerate(S_METRICS_T):
            ax = axes[row, col]
            x = np.array([r[key] for r in paired.get(T_PAIR[0], [])])
            y = np.array([r[key] for r in paired.get(T_PAIR[1], [])])
            if len(x) == 0 or len(y) == 0:
                ax.set_title(f'{metric_title}\n{R_type} — no data', fontsize=8)
                continue
            ax.scatter(x, y, alpha=0.4, s=18, color=T_COLORS_R.get(R_type, '#555'))
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
            corr = float(np.corrcoef(x, y)[0, 1])
            ax.set_xlabel(f'Score under {T_PAIR[0]} T', fontsize=8)
            ax.set_ylabel(f'Score under {T_PAIR[1]} T', fontsize=8)
            ax.set_title(f'{metric_title}\n{R_type} (scale={R_scale})  r={corr:.2f}',
                         fontsize=8)
            ax.set_xlim(-0.05, 1.1)
            ax.set_ylim(-0.05, 1.1)

    fig.suptitle(
        f'Group 2: T-sensitivity — same R scored under {T_PAIR[0]} vs {T_PAIR[1]} T  (S=20)',
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, '12_t_sensitivity.pdf')


def plot_13_s_sweep(s_data, S_values, T_types):
    print("Plot 13: S sweep")
    T_COLORS_S = {'dirichlet': '#4C72B0', 'deterministic': '#55A868'}
    T_LABELS_S = {
        'dirichlet':     'Dirichlet(α=0.1)',
        'deterministic': 'Deterministic',
    }
    S_METRICS_SWEEP = METRICS[:2]

    fig, axes = plt.subplots(1, len(S_METRICS_SWEEP), figsize=(6 * len(S_METRICS_SWEEP), 5))
    for ax, (key, metric_title) in zip(axes, S_METRICS_SWEEP):
        for T_type in T_types:
            means, stds = [], []
            for S in S_values:
                scores = s_data.get((S, T_type), [])
                vals   = np.array([r[key] for r in scores])
                means.append(vals.mean() if len(vals) > 0 else float('nan'))
                stds.append(vals.std()   if len(vals) > 0 else float('nan'))
            means = np.array(means)
            stds  = np.array(stds)
            color = T_COLORS_S.get(T_type, '#888')
            ax.plot(S_values, means, 'o-', color=color,
                    label=T_LABELS_S.get(T_type, T_type))
            ax.fill_between(S_values, means - stds, means + stds, color=color, alpha=0.15)

        ax.axhline(HIGH_AGENCY_THRESHOLD, color='grey', lw=0.8, ls='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xticks(S_values)
        ax.set_xticklabels(S_values)
        ax.set_xlabel('S (state space size)', fontsize=9)
        ax.set_ylabel('Mean score (±1 std)', fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(metric_title, fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(
        'Group 3b: PAM scores vs S × T type  (Gaussian R σ=1, A=4, γ=0.95)',
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, '13_s_sweep.pdf')


def main():
    pkl_path = os.path.join(_SCRIPT_DIR, 'results', 'sweep_results.pkl')
    if not os.path.exists(pkl_path):
        print(f"ERROR: {pkl_path} not found. Run 01_build.py first.")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    p_values   = results.get('p_values', [0.01, 0.05, 0.2, 0.5, 1.0])
    bern_data  = results['bern_data']
    ss_data    = results['ss_data']
    gamma_data = results['gamma_data']
    gamma_p    = results.get('gamma_params', {})
    t_data     = results['t_data']
    t_params   = results.get('t_params', {})
    s_data     = results['s_data']
    s_params   = results.get('s_params', {})

    gammas       = gamma_p.get('gammas', sorted(set(k[2] for k in gamma_data)))
    R_conditions = gamma_p.get('R_conditions', sorted(set((k[0], k[1]) for k in gamma_data)))
    T_types_sens = t_params.get('T_types', ['uniform', 'dirichlet', 'deterministic'])
    S_values     = s_params.get('S_values', sorted(set(k[0] for k in s_data)))
    T_types_s    = s_params.get('T_types', sorted(set(k[1] for k in s_data)))

    plot_09_bernoulli(bern_data, p_values)
    plot_10_spike_slab(ss_data, p_values)
    plot_11_gamma_sweep(gamma_data, gammas, R_conditions)
    plot_12_t_sensitivity(t_data, R_conditions, T_types_sens)
    plot_13_s_sweep(s_data, S_values, T_types_s)

    print(f"\nAll plots saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
