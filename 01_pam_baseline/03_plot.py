"""
03_plot.py — Generate PAM analysis figures 01–08 from pre-computed results.

Usage:  python 03_plot.py
Requires: results/pam_experiment.pkl  (from 01_build.py)
Saves:    figures/01_q3_heatmap.pdf ... figures/08_rscale_sweep_agenticity.pdf
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from runners import random_mdp
from core import MDP

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    'gaussian':  '#4C72B0',
    'uniform':   '#DD8452',
    'bernoulli': '#55A868',
    'potential': '#C44E52',
    'goal':      '#8172B2',
}

HIGH_AGENCY_THRESHOLD = 0.5


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


def plot_01_q3_heatmap(q3):
    print("Plot 01: Q3 heatmap")
    PAM_DISPLAY = ['adv_gap_norm', 'vstar_var_norm', 'H_eps_norm',
                   'mce_entropy_norm', 'ctrl_adv_norm', 'one_step_norm', 'composite']
    PAM_LABELS  = ['Adv\nGap', 'V*\nVar', 'H\nEps', 'MCE\nEnt',
                   'Ctrl\nAdv', 'OneStep\nRec', 'Composite']

    mdp_order = sorted(q3.keys(), key=lambda n: -q3[n]['composite'])
    matrix = np.array([[q3[n].get(p, 0.0) or 0.0 for p in PAM_DISPLAY] for n in mdp_order])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(PAM_LABELS)))
    ax.set_xticklabels(PAM_LABELS, fontsize=9)
    ax.set_yticks(range(len(mdp_order)))
    ax.set_yticklabels(mdp_order, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=7.5, color='black' if v < 0.7 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.03, label='Normalised score')
    ax.set_title('Q3 — Human-made MDPs: normalised PAM scores', fontsize=11)
    ax.axvline(len(PAM_DISPLAY) - 1.5, color='white', lw=2)
    fig.tight_layout()
    save(fig, '01_q3_heatmap.pdf')


def plot_02_q3_grouped_bar(q3):
    print("Plot 02: Q3 grouped bar")
    metrics_bar = ['adv_gap_norm', 'vstar_var_norm', 'H_eps_norm', 'ctrl_adv_norm', 'composite']
    labels_bar  = ['Adv Gap', 'V* Var', 'H Eps', 'Ctrl Adv', 'Composite']
    mdp_order = sorted(q3.keys(), key=lambda n: -q3[n]['composite'])
    n_mdp = len(mdp_order)
    n_met = len(metrics_bar)
    x = np.arange(n_mdp)
    width = 0.14
    offsets = np.linspace(-(n_met-1)/2, (n_met-1)/2, n_met) * width

    fig, ax = plt.subplots(figsize=(13, 5))
    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#333333']
    for idx, (met, lab, col) in enumerate(zip(metrics_bar, labels_bar, bar_colors)):
        vals = [q3[n].get(met, 0.0) or 0.0 for n in mdp_order]
        ax.bar(x + offsets[idx], vals, width, label=lab, color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(mdp_order, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Normalised score')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Q3 — Human-made MDPs: PAM comparison by environment', fontsize=10)
    ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5)
    fig.tight_layout()
    save(fig, '02_q3_grouped_bar.pdf')


def plot_03_q1_boxplots(q1):
    print("Plot 03: Q1 box plots by R_type")
    R_TYPES    = ['gaussian', 'uniform', 'bernoulli', 'potential', 'goal']
    pam_indices = [0, 1, 3]
    pam_titles  = ['Advantage Gap (norm)', 'V*−Vrand Variance (norm)',
                   'Planning Horizon H_eps (norm)']
    S_show = max(set(k[0] for k in q1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, pi, title in zip(axes, pam_indices, pam_titles):
        data = [q1.get((S_show, rt), np.full((1, 5), np.nan))[:, pi] for rt in R_TYPES]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color='black', lw=1.5),
                        whiskerprops=dict(lw=1), capprops=dict(lw=1),
                        showfliers=True, flierprops=dict(marker='o', ms=3, alpha=0.5))
        for patch, rt in zip(bp['boxes'], R_TYPES):
            patch.set_facecolor(PALETTE[rt])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(R_TYPES)+1))
        ax.set_xticklabels(R_TYPES, rotation=20, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(0.5, color='grey', lw=0.7, ls='--', alpha=0.5)

    fig.suptitle(f'Q1 — PAM distributions by reward type (S={S_show}, T fixed, n=50)',
                 fontsize=10)
    fig.tight_layout()
    save(fig, '03_q1_boxplots_by_Rtype.pdf')


def plot_04_q2_variance_decomp(q2):
    print("Plot 04: Q2 variance decomposition")
    structures  = sorted(set(k[0] for k in q2))
    S_vals_q2   = sorted(set(k[1] for k in q2))
    R_types_q2  = sorted(set(k[2] for k in q2))

    def _struct_label(s):
        if s.startswith('dirichlet_'):
            return f'Dir(α={s.split("_",1)[1]})'
        return s

    n_s      = len(S_vals_q2)
    bar_w    = 0.22
    colors_q2 = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, n_s, figsize=(5 * n_s, 5), sharey=True)
    if n_s == 1:
        axes = [axes]

    for ax, S in zip(axes, S_vals_q2):
        x_pos = np.arange(len(R_types_q2))
        for si, (struct, color) in enumerate(zip(structures, colors_q2)):
            ratios = [q2.get((struct, S, rt), {}).get('ratio', float('nan'))
                      for rt in R_types_q2]
            offset = (si - (len(structures) - 1) / 2) * bar_w
            ax.bar(x_pos + offset, ratios, bar_w, label=_struct_label(struct),
                   color=color, alpha=0.85)

        ax.axhline(0.5, color='red', lw=1.2, ls='--', label='50% threshold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(R_types_q2, rotation=20, ha='right', fontsize=8)
        ax.set_title(f'S={S}', fontsize=10)
        ax.set_ylabel('T share of total variance (ratio)')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7)

    fig.suptitle('Q2 — Fraction of composite variance explained by T (vs R)', fontsize=10)
    fig.tight_layout()
    save(fig, '04_q2_variance_decomp.pdf')


def plot_05_q1_scatter(q1):
    print("Plot 05: Q1 adv_gap vs vstar_var scatter")
    R_TYPES = ['gaussian', 'uniform', 'bernoulli', 'potential', 'goal']
    S_vals = sorted(set(k[0] for k in q1))
    fig, axes = plt.subplots(1, max(len(S_vals), 1), figsize=(6 * max(len(S_vals), 1), 5))
    if len(S_vals) == 1:
        axes = [axes]
    for ax, S in zip(axes, S_vals):
        handles = []
        for rt in R_TYPES:
            mat = q1.get((S, rt))
            if mat is None:
                continue
            ax.scatter(mat[:, 0], mat[:, 1], alpha=0.55, s=25,
                       color=PALETTE[rt], label=rt)
            handles.append(mpatches.Patch(color=PALETTE[rt], label=rt))
        ax.set_xlabel('adv_gap (norm)', fontsize=9)
        ax.set_ylabel('vstar_var (norm)', fontsize=9)
        ax.set_title(f'S={S}', fontsize=9)
        ax.legend(handles=handles, fontsize=7)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.axline((0, 0), (1, 1), color='grey', lw=0.8, ls='--', alpha=0.5)

    fig.suptitle('Q1 — adv_gap vs vstar_var by R_type', fontsize=10)
    fig.tight_layout()
    save(fig, '05_q1_scatter_gap_vs_vstarvar.pdf')


def plot_06_q1_radar(q1):
    print("Plot 06: Q1 radar chart — mean PAM profile by R_type")
    R_TYPES    = ['gaussian', 'uniform', 'bernoulli', 'potential', 'goal']
    PAMS_RADAR = ['adv_gap', 'vstar_var', 'H_eps']
    RADAR_IDX  = [0, 1, 3]
    N = len(PAMS_RADAR)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    S_vals = sorted(set(k[0] for k in q1))
    n_cols = max(len(S_vals), 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), subplot_kw=dict(polar=True))
    if n_cols == 1:
        axes = [axes]
    for ax, S in zip(axes, S_vals):
        for rt in R_TYPES:
            mat = q1.get((S, rt))
            if mat is None:
                continue
            means = [np.nanmean(mat[:, i]) for i in RADAR_IDX]
            means += means[:1]
            ax.plot(angles, means, color=PALETTE[rt], lw=2, label=rt)
            ax.fill(angles, means, color=PALETTE[rt], alpha=0.08)
        ax.set_thetagrids(np.degrees(angles[:-1]), PAMS_RADAR, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'S={S}', fontsize=10, pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

    fig.suptitle('Q1 — Mean normalised PAM profile by R_type', fontsize=10)
    fig.tight_layout()
    save(fig, '06_q1_radar_by_Rtype.pdf')


def plot_07_reward_dist(q1, n_random_mdps):
    print("Plot 07: reward distribution agenticity")
    R_TYPES = ['gaussian', 'uniform', 'bernoulli', 'potential', 'goal']
    DIST_METRICS = [
        (0, 'Advantage Gap (norm)'),
        (1, 'V*−V^rand Variance (norm)'),
        (3, 'Planning Horizon H_eps (norm)'),
    ]
    S_dist = max(set(k[0] for k in q1))

    fig, axes = plt.subplots(1, len(DIST_METRICS), figsize=(5 * len(DIST_METRICS), 5))
    if len(DIST_METRICS) == 1:
        axes = [axes]
    fig.suptitle(
        f'Agenticity distributions by reward type  (S={S_dist}, T fixed, n={n_random_mdps})',
        fontsize=10,
    )
    for ax, (col_idx, title) in zip(axes, DIST_METRICS):
        for rt in R_TYPES:
            mat = q1.get((S_dist, rt))
            if mat is None:
                continue
            vals = mat[:, col_idx]
            vals = vals[~np.isnan(vals)]
            frac_high = (vals >= HIGH_AGENCY_THRESHOLD).mean()
            ax.hist(vals, bins=20, range=(0, 1), alpha=0.45, color=PALETTE[rt], density=True,
                    label=f'{rt} ({frac_high:.0%})')
        ax.axvline(HIGH_AGENCY_THRESHOLD, color='black', lw=1.4, ls='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalised score', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, title='R type (% high)', title_fontsize=7)

    fig.tight_layout()
    save(fig, '07_reward_dist_agenticity.pdf')


def plot_08_rscale_sweep(scale_data, n_random_mdps):
    print("Plot 08: R_scale sweep for Gaussian rewards")
    R_SCALES = sorted(scale_data.keys())
    SCALE_CMAP   = plt.cm.viridis
    scale_colors = [SCALE_CMAP(i / max(len(R_SCALES) - 1, 1)) for i in range(len(R_SCALES))]

    SCALE_METRICS = [
        (0, 'Advantage Gap (norm)'),
        (1, 'V*−V^rand Variance (norm)'),
        (3, 'Planning Horizon H_eps (norm)'),
    ]

    fig, axes = plt.subplots(1, len(SCALE_METRICS), figsize=(5 * len(SCALE_METRICS), 5))
    if len(SCALE_METRICS) == 1:
        axes = [axes]
    fig.suptitle(
        f'Agenticity vs reward scale  (Gaussian, S=10, T fixed, n={n_random_mdps})',
        fontsize=10,
    )
    for ax, (col_idx, title) in zip(axes, SCALE_METRICS):
        for R_scale, color in zip(R_SCALES, scale_colors):
            vals = scale_data[R_scale][:, col_idx]
            vals = vals[~np.isnan(vals)]
            frac = (vals >= HIGH_AGENCY_THRESHOLD).mean()
            ax.hist(vals, bins=20, range=(0, 1), alpha=0.45, color=color, density=True,
                    label=f'σ={R_scale} ({frac:.0%})')
        ax.axvline(HIGH_AGENCY_THRESHOLD, color='black', lw=1.4, ls='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalised score', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, title='R_scale (% high)', title_fontsize=7)

    fig.tight_layout()
    save(fig, '08_rscale_sweep_agenticity.pdf')


def main():
    pkl_path = os.path.join(_SCRIPT_DIR, 'results', 'pam_experiment.pkl')
    if not os.path.exists(pkl_path):
        print(f"ERROR: {pkl_path} not found. Run 01_build.py first.")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    q1          = results['q1']
    q1_raw      = results['q1_raw']
    q2          = results['q2']
    q3          = results['q3']
    scale_data  = results.get('rscale_sweep', {})
    n           = results.get('meta', {}).get('n_random_mdps', 50)

    plot_01_q3_heatmap(q3)
    plot_02_q3_grouped_bar(q3)
    plot_03_q1_boxplots(q1)
    plot_04_q2_variance_decomp(q2)
    plot_05_q1_scatter(q1)
    plot_06_q1_radar(q1)
    plot_07_reward_dist(q1, n)
    if scale_data:
        plot_08_rscale_sweep(scale_data, n)
    else:
        print("Plot 08: skipped (no rscale_sweep in results — re-run 01_build.py)")

    print(f"\nAll plots saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
