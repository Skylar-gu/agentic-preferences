"""
02_plot.py — Generate MCE alpha calibration figures from pre-computed sweep.

Usage:  python 02_plot.py
Requires: results/mce_alpha_sweep.pkl  (from 01_sweep.py)

Saves:
  figures/14_mce_alpha_distributions.pdf  — per-alpha histograms (R_type × T overlay)
  figures/15_mce_alpha_curves.pdf         — mean ± std vs alpha (one panel per R_type)
  figures/16_mce_t_sensitivity.pdf        — per-R_type × per-T distribution shift
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

R_COLORS = {
    'gaussian':   '#4C72B0',
    'uniform':    '#DD8452',
    'bernoulli':  '#55A868',
    'spike_slab': '#C44E52',
    'goal':       '#8172B2',
    'potential':  '#777777',
}


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 14 — per-alpha panel: histograms per R_type, T's overlaid
# ---------------------------------------------------------------------------

def plot_14_alpha_distributions(data, params):
    """One column per alpha. Within each panel: one histogram per R_type.
    Each R_type has N_T overlaid transparent histograms (T-sensitivity visible
    as spread within a colour band)."""
    print("Plot 14: per-alpha distributions (R_type histograms, T overlay)")

    alphas   = params['alphas']
    R_types  = params['R_types']
    N_T      = params['N_T']

    n_cols = len(alphas)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 5), sharey=True)

    for col, alpha in enumerate(alphas):
        ax = axes[col]
        for rt in R_types:
            color = R_COLORS.get(rt, '#888')
            for t_idx in range(N_T):
                vals = data[t_idx][rt][alpha]
                ax.hist(vals, bins=25, range=(0, 1), alpha=0.25,
                        color=color, density=True)
            # Overlay mean line per R_type (across all T's pooled)
            all_vals = np.concatenate([data[t_idx][rt][alpha] for t_idx in range(N_T)])
            ax.axvline(all_vals.mean(), color=color, lw=1.5, ls='--',
                       label=f'{rt} ({all_vals.mean():.2f})')

        ax.set_title(f'α = {alpha}', fontsize=9)
        ax.set_xlabel('MCE entropy norm (1−H/logA)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Density', fontsize=8)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=6, loc='upper left')

    fig.suptitle(
        f'Fig 14: MCE entropy distributions by R_type at each α\n'
        f'(transparent = individual T matrices, dashed = pooled mean;  N_R={params["N_R"]})',
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, '14_mce_alpha_distributions.pdf')


# ---------------------------------------------------------------------------
# Figure 15 — mean ± std vs alpha, one panel per R_type, lines = T's
# ---------------------------------------------------------------------------

def plot_15_alpha_curves(data, params):
    """Shows the alpha-response curve per R_type, with one line per T.
    Thick line = mean across T's; shaded band = T-to-T spread."""
    print("Plot 15: mean ± std vs alpha, per R_type")

    alphas  = params['alphas']
    R_types = params['R_types']
    N_T     = params['N_T']

    n_cols = 3
    n_rows = int(np.ceil(len(R_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for idx, rt in enumerate(R_types):
        ax = axes[idx // n_cols][idx % n_cols]
        color = R_COLORS.get(rt, '#888')

        # One thin line per T
        t_means = []
        for t_idx in range(N_T):
            means = [data[t_idx][rt][a].mean() for a in alphas]
            ax.plot(alphas, means, color=color, lw=0.8, alpha=0.5,
                    label=f'T{t_idx+1}' if idx == 0 else None)
            t_means.append(means)

        # Grand mean ± between-T std
        t_means = np.array(t_means)           # (N_T, n_alphas)
        grand_mean = t_means.mean(axis=0)
        grand_std  = t_means.std(axis=0)
        ax.plot(alphas, grand_mean, color=color, lw=2.2, label='mean')
        ax.fill_between(alphas,
                        grand_mean - grand_std,
                        grand_mean + grand_std,
                        color=color, alpha=0.15)

        # Mark chosen alpha
        ax.axvline(0.25, color='black', lw=1.0, ls=':', alpha=0.6)

        ax.set_xscale('log')
        ax.set_xticks(alphas)
        ax.set_xticklabels([str(a) for a in alphas], fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(rt, fontsize=9)
        ax.set_xlabel('α (log scale)', fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel('Mean MCE norm', fontsize=8)

    # Remove empty subplots
    for idx in range(len(R_types), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        'Fig 15: MCE entropy (normalised) vs α per R_type\n'
        '(thin lines = individual T matrices, thick = mean, band = between-T std;  dotted = α=0.25)',
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, '15_mce_alpha_curves.pdf')


# ---------------------------------------------------------------------------
# Figure 16 — T-sensitivity: at chosen alpha, distribution per T per R_type
# ---------------------------------------------------------------------------

def plot_16_t_sensitivity(data, params, chosen_alpha=0.25):
    """At the chosen alpha, shows distributions per R_type side-by-side for
    each T matrix.  Lets you see how much T shifts the score distribution."""
    print(f"Plot 16: T-sensitivity at α={chosen_alpha}")

    R_types = params['R_types']
    N_T     = params['N_T']
    N_R     = params['N_R']

    n_cols = N_T
    n_rows = len(R_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows),
                             squeeze=False, sharex=True)

    for row, rt in enumerate(R_types):
        color = R_COLORS.get(rt, '#888')
        all_vals = np.concatenate([data[t][rt][chosen_alpha] for t in range(N_T)])
        grand_mean = all_vals.mean()

        for col in range(N_T):
            ax = axes[row][col]
            vals = data[col][rt][chosen_alpha]
            ax.hist(vals, bins=25, range=(0, 1), color=color, alpha=0.6, density=True)
            ax.axvline(vals.mean(), color='black', lw=1.2, ls='--')
            ax.axvline(grand_mean, color='red', lw=0.8, ls=':', alpha=0.7)
            ax.set_xlim(0, 1)

            if row == 0:
                ax.set_title(f'T{col+1}', fontsize=9)
            if col == 0:
                ax.set_ylabel(rt, fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel('MCE norm', fontsize=7)

            # Annotate mean ± std
            ax.text(0.97, 0.93, f'{vals.mean():.3f}\n±{vals.std():.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=6.5)

    fig.suptitle(
        f'Fig 16: T-sensitivity at α={chosen_alpha}  (dashed=T mean, dotted=grand mean)\n'
        f'N_R={N_R} per cell',
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, '16_mce_t_sensitivity.pdf')


# ---------------------------------------------------------------------------
# Figure 17 — Between-type variance vs alpha (per T and pooled)
# ---------------------------------------------------------------------------

def plot_17_between_type_var(data, params):
    """Summary: between-type variance (non-potential only) vs alpha.
    One line per T, thick line = mean across T's.  Identifies the sweet-spot alpha."""
    print("Plot 17: between-type variance vs alpha")

    alphas  = params['alphas']
    R_types = params['R_types']
    N_T     = params['N_T']
    non_pot = [rt for rt in R_types if rt != 'potential']

    fig, ax = plt.subplots(figsize=(7, 4))
    T_bvars = []
    for t_idx in range(N_T):
        bvars = []
        for alpha in alphas:
            type_means = [data[t_idx][rt][alpha].mean() for rt in non_pot]
            bvars.append(np.var(type_means))
        ax.plot(alphas, bvars, color='#4C72B0', lw=0.9, alpha=0.5,
                label=f'T{t_idx+1}')
        T_bvars.append(bvars)

    T_bvars = np.array(T_bvars)
    grand = T_bvars.mean(axis=0)
    ax.plot(alphas, grand, color='#4C72B0', lw=2.5, label="mean across T's")
    ax.fill_between(alphas,
                    grand - T_bvars.std(axis=0),
                    grand + T_bvars.std(axis=0),
                    color='#4C72B0', alpha=0.15)

    # Mark chosen alpha
    best_alpha = alphas[int(np.argmax(grand))]
    ax.axvline(0.25, color='black', lw=1.2, ls=':', alpha=0.8,
               label=f'chosen α=0.25')
    ax.axvline(best_alpha, color='red', lw=1.0, ls='--', alpha=0.6,
               label=f'peak α={best_alpha}')

    ax.set_xscale('log')
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel('α (log scale)', fontsize=9)
    ax.set_ylabel('Between-type variance (non-potential)', fontsize=9)
    ax.set_title('Fig 17: discrimination power vs α\n'
                 '(higher = MCE entropy better separates R_types)', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, '17_mce_between_type_var.pdf')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pkl_path = os.path.join(_SCRIPT_DIR, 'results', 'mce_alpha_sweep.pkl')
    if not os.path.exists(pkl_path):
        print(f"ERROR: {pkl_path} not found. Run 01_sweep.py first.")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    data   = results['data']
    params = results['params']

    print(f"Loaded: N_T={params['N_T']}, N_R={params['N_R']}, "
          f"alphas={params['alphas']}")

    plot_14_alpha_distributions(data, params)
    plot_15_alpha_curves(data, params)
    plot_16_t_sensitivity(data, params, chosen_alpha=0.25)
    plot_17_between_type_var(data, params)

    print(f"\nAll plots saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
