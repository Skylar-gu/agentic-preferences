"""
plot_pams.py — Visualise PAM results from run_pam_experiment.

Plots:
  1. Q3 heatmap     — human-made MDPs × PAMs (normalised)
  2. Q3 grouped bar — composite + key metrics by MDP
  3. Q1 box plots   — PAM distributions by R_type (adv_gap, vstar_var, mce_entropy)
  4. Q2 variance    — T vs R share of variance by (S, R_type)
  5. Q1 corr grid   — adv_gap vs mce_entropy scatter, coloured by R_type

Run:  python environments/plot_pams.py
Saves figures to environments/figures/*.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from runners import run_pam_experiment, random_mdp
from core import MDP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    'gaussian':  '#4C72B0',
    'uniform':   '#DD8452',
    'bernoulli': '#55A868',
    'potential': '#C44E52',
    'goal':      '#8172B2',
}
MDP_COLORS = {
    'Chain-Terminal':  '#a8c8e8',
    'Chain-Dense':     '#2171b5',
    'Chain-Lottery':   '#c6dbef',
    'Chain-Progress':  '#6baed6',
    'Grid-Goal':       '#fd8d3c',
    'Grid-Local':      '#d94801',
    'Grid-Cliff':      '#fdae6b',
}

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


# ===========================================================================
# Main Execution Block
# ===========================================================================
if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # Run experiment
    # ---------------------------------------------------------------------------
    print("Running experiment (n=50 per cell, no MI)...")
    results = run_pam_experiment(
        n_random_mdps=50,
        S_values=[5, 10],
        A=4,
        gamma=0.95,
        k=3,
        R_types=['gaussian', 'uniform', 'bernoulli', 'potential', 'goal'],
        n_fixed_T=5,
        T_structures=['dirichlet_0.1', 'dirichlet_1.0', 'dirichlet_10.0'],
        rng_seed=42,
        verbose=True,
        include_mi=False,
    )
    q1, q1_raw, q2, q3 = results['q1'], results['q1_raw'], results['q2'], results['q3']
    print("Done.\n")

    # ===========================================================================
    # Plot 1 — Q3 Heatmap: human-made MDPs × normalised PAMs
    # ===========================================================================
    print("Plot 1: Q3 heatmap")

    # UPDATED: Added empowerment_norm and starc_clarity to display
    PAM_DISPLAY = ['adv_gap_norm', 'vstar_var_norm', 'H_eps_norm',
                   'mce_entropy_norm', 'empowerment_norm', 'starc_clarity', 'composite']
    PAM_LABELS  = ['Adv\nGap', 'V*\nVar', 'H\nEps', 'MCE\nEnt', 'Empow\nNorm', 'STARC\nClar', 'Composite']

    mdp_order = sorted(q3.keys(), key=lambda n: -q3[n]['composite'])
    matrix = np.array([[q3[n].get(p, 0.0) or 0.0 for p in PAM_DISPLAY] for n in mdp_order])

    fig, ax = plt.subplots(figsize=(12, 5))
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
    ax.set_title('Q3 — Human-made MDPs: normalised PAM scores\n(sorted by composite, descending)', fontsize=11)
    ax.axvline(len(PAM_DISPLAY) - 1.5, color='white', lw=2)  # separate composite
    fig.tight_layout()
    save(fig, '01_q3_heatmap.png')

    # ===========================================================================
    # Plot 2 — Q3 Grouped bar: top metrics side-by-side per MDP
    # ===========================================================================
    print("Plot 2: Q3 grouped bar")

    # UPDATED: Added empowerment_norm to the bars
    metrics_bar = ['adv_gap_norm', 'vstar_var_norm', 'empowerment_norm', 'composite']
    labels_bar  = ['Adv Gap', 'V* Var', 'Empowerment', 'Composite']
    n_mdp = len(mdp_order)
    n_met = len(metrics_bar)
    x = np.arange(n_mdp)
    width = 0.18
    offsets = np.linspace(-(n_met-1)/2, (n_met-1)/2, n_met) * width

    fig, ax = plt.subplots(figsize=(13, 5))
    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#333333']
    for idx, (met, lab, col) in enumerate(zip(metrics_bar, labels_bar, bar_colors)):
        vals = [q3[n].get(met, 0.0) or 0.0 for n in mdp_order]
        ax.bar(x + offsets[idx], vals, width, label=lab, color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(mdp_order, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Normalised score')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Q3 — Human-made MDPs: PAM comparison by environment\n'
                 '(Showing balance between planning gap and empowerment)',
                 fontsize=10)
    ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5)
    fig.tight_layout()
    save(fig, '02_q3_grouped_bar.png')

    # ===========================================================================
    # Plot 3 — Q1 Box plots: PAM distributions by R_type (S=10 only)
    # ===========================================================================
    print("Plot 3: Q1 box plots by R_type")

    R_TYPES = ['gaussian', 'uniform', 'bernoulli', 'potential', 'goal']
    pam_indices = [0, 1, 3, 4]
    pam_names   = ['adv_gap_norm', 'vstar_var_norm', 'H_eps_norm', 'mce_entropy_norm']
    pam_titles  = ['Advantage Gap (norm)', 'V*−Vrand Variance (norm)',
                   'Planning Horizon H_eps (norm)', 'MCE Policy Entropy (norm)']
    S_show = 10

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, pi, title in zip(axes, pam_indices, pam_titles):
        data  = [q1.get((S_show, rt), np.full((1, 5), np.nan))[:, pi] for rt in R_TYPES]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='black', lw=1.5),
                        whiskerprops=dict(lw=1), capprops=dict(lw=1), showfliers=True,
                        flierprops=dict(marker='o', ms=3, alpha=0.5))
        for patch, rt in zip(bp['boxes'], R_TYPES):
            patch.set_facecolor(PALETTE[rt])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(R_TYPES)+1))
        ax.set_xticklabels(R_TYPES, rotation=20, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(0.5, color='grey', lw=0.7, ls='--', alpha=0.5)

    fig.suptitle(f'Q1 — PAM distributions by reward type (S={S_show}, T fixed, n=50)\n'
                 f'potential = non-agentic control; should cluster low on adv_gap/vstar_var',
                 fontsize=10)
    fig.tight_layout()
    save(fig, '03_q1_boxplots_by_Rtype.png')

    # ===========================================================================
    # Plot 4 — Q2 Variance decomposition: T share vs R_type
    # ===========================================================================
    print("Plot 4: Q2 variance decomposition")

    structures = sorted(set(k[0] for k in q2))
    S_vals_q2  = sorted(set(k[1] for k in q2))
    R_types_q2 = sorted(set(k[2] for k in q2))

    def _struct_label(s):
        if s.startswith('dirichlet_'):
            alpha = s.split('_', 1)[1]
            return f'Dir(α={alpha})'
        return s

    n_s = len(S_vals_q2)
    n_r = len(R_types_q2)
    bar_w = 0.22
    colors_q2 = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, n_s, figsize=(5 * n_s, 5), sharey=True)
    if n_s == 1:
        axes = [axes]

    for ax, S in zip(axes, S_vals_q2):
        x_pos = np.arange(n_r)
        for si, (struct, color) in enumerate(zip(structures, colors_q2)):
            ratios = []
            for rt in R_types_q2:
                key = (struct, S, rt)
                v   = q2.get(key, {})
                ratios.append(v.get('ratio', float('nan')))

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

    fig.suptitle('Q2 — Fraction of composite variance explained by T (vs R)\n'
                 'Dir(α=0.1)=near-deterministic  α=1.0=balanced  α=10.0=near-uniform\n'
                 'Low ratio → R drives agenticity; high ratio → T topology dominates',
                 fontsize=10)
    fig.tight_layout()
    save(fig, '04_q2_variance_decomp.png')

    # ===========================================================================
    # Plot 5 — Q1 Scatter: adv_gap vs mce_entropy coloured by R_type (S=10)
    # ===========================================================================
    print("Plot 5: Q1 adv_gap vs mce_entropy scatter")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, S in zip(axes, [5, 10]):
        handles = []
        for rt in R_TYPES:
            mat = q1.get((S, rt))
            if mat is None:
                continue
            ax.scatter(mat[:, 0], mat[:, 4], alpha=0.55, s=25,
                       color=PALETTE[rt], label=rt)
            handles.append(mpatches.Patch(color=PALETTE[rt], label=rt))
        ax.set_xlabel('adv_gap (norm)', fontsize=9)
        ax.set_ylabel('mce_entropy (norm)', fontsize=9)
        ax.set_title(f'S={S}: adv_gap vs MCE entropy\n(strong negative corr expected — high gap → low entropy)',
                     fontsize=9)
        ax.legend(handles=handles, fontsize=7)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.axline((0, 1), (1, 0), color='grey', lw=0.8, ls='--', alpha=0.5)

    fig.suptitle('Q1 — adv_gap ↔ MCE entropy trade-off by R_type\n'
                 'potential rewards should cluster top-left (low gap, high entropy)',
                 fontsize=10)
    fig.tight_layout()
    save(fig, '05_q1_scatter_gap_vs_entropy.png')

    # ===========================================================================
    # Plot 6 — Q1 Mean PAM profile per R_type (radar / spider chart)
    # ===========================================================================
    print("Plot 6: Q1 radar chart — mean PAM profile by R_type")

    PAMS_RADAR = ['adv_gap', 'vstar_var', 'H_eps', 'mce_entropy']
    RADAR_IDX  = [0, 1, 3, 4]
    N = len(PAMS_RADAR)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))
    for ax, S in zip(axes, [5, 10]):
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

    fig.suptitle('Q1 — Mean normalised PAM profile by R_type\n'
                 '(potential should show low adv_gap + low vstar_var; goal should be intermediate)',
                 fontsize=10)
    fig.tight_layout()
    save(fig, '06_q1_radar_by_Rtype.png')

    # ===========================================================================
    # Plot 7 — Reward-distribution agenticity: histograms + high-agency fraction
    # ===========================================================================
    print("Plot 7: reward distribution agenticity (histograms + high-agency fraction)")

    HIGH_AGENCY_THRESHOLD = 0.5

    DIST_METRICS = [
        (0, 'adv_gap_norm',    'Advantage Gap (norm)'),
        (1, 'vstar_var_norm',  'V*−V^rand Variance (norm)'),
        (3, 'H_eps_norm',      'Planning Horizon H_eps (norm)'),
        (4, 'mce_entropy_norm','MCE Policy Entropy (norm)'),
    ]
    n_metrics = len(DIST_METRICS)
    S_dist = 10

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    fig.suptitle(
        f'Agenticity distributions by reward type  (S={S_dist}, T fixed, n={results["meta"]["n_random_mdps"]})\n'
        f'Dashed line = high-agency threshold ({HIGH_AGENCY_THRESHOLD}); '
        f'fraction label = proportion above threshold',
        fontsize=10,
    )

    for ax, (col_idx, _, title) in zip(axes, DIST_METRICS):
        for rt in R_TYPES:
            mat = q1.get((S_dist, rt))
            if mat is None:
                continue
            vals = mat[:, col_idx]
            vals = vals[~np.isnan(vals)]
            frac_high = (vals >= HIGH_AGENCY_THRESHOLD).mean()
            color = PALETTE[rt]
            ax.hist(vals, bins=20, range=(0, 1), alpha=0.45, color=color,
                    label=f'{rt} ({frac_high:.0%})', density=True)

        ax.axvline(HIGH_AGENCY_THRESHOLD, color='black', lw=1.4, ls='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalised score', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, title='R type (% high)', title_fontsize=7)

    fig.tight_layout()
    save(fig, '07_reward_dist_agenticity.png')

    # ===========================================================================
    # Plot 8 — R_scale sweep: how reward variance drives agenticity (Gaussian)
    # ===========================================================================
    print("Plot 8: R_scale sweep for Gaussian rewards")

    R_SCALES   = [0.1, 0.5, 1.0, 2.0, 5.0]
    N_SCALE    = results['meta']['n_random_mdps']
    S_scale    = 10
    A_scale    = 4
    gamma_sc   = 0.95
    rng_sc     = np.random.default_rng(99)

    canonical_sc = random_mdp(S_scale, A_scale, gamma=gamma_sc, k=3,
                               R_type='gaussian', terminal_states=1,
                               rng=np.random.default_rng(7))
    T_sc       = canonical_sc.T
    term_sc    = canonical_sc.terminal
    d0_sc      = canonical_sc.d0

    SCALE_CMAP = plt.cm.viridis
    scale_colors = [SCALE_CMAP(i / (len(R_SCALES) - 1)) for i in range(len(R_SCALES))]

    SCALE_METRICS = [
        (0, 'Advantage Gap (norm)'),
        (1, 'V*−V^rand Variance (norm)'),
        (3, 'Planning Horizon H_eps (norm)'),
        (4, 'MCE Policy Entropy (norm)'),
    ]

    scale_data = {}
    for R_scale in R_SCALES:
        rows = []
        for _ in range(N_SCALE):
            tmp = random_mdp(S_scale, A_scale, gamma=gamma_sc, k=3,
                             R_type='gaussian', R_scale=R_scale, terminal_states=1,
                             rng=np.random.default_rng(int(rng_sc.integers(0, 2**31))))
            mdp_sc = MDP(S=S_scale, A=A_scale, T=T_sc, R=tmp.R,
                         gamma=gamma_sc, terminal=term_sc, d0=d0_sc.copy())
            from pams import agenticity_score as _agscore
            r = _agscore(mdp_sc, verbose=False, compute_mi=False,
                         rng=np.random.default_rng(int(rng_sc.integers(0, 2**31))))
            rows.append([r['adv_gap_norm'], r['vstar_var_norm'],
                         r['mi_diff'] if r['mi_diff'] is not None else float('nan'),
                         r['H_eps_norm'], r['mce_entropy_norm']])
        scale_data[R_scale] = np.array(rows)

    fig, axes = plt.subplots(1, len(SCALE_METRICS), figsize=(5 * len(SCALE_METRICS), 5))
    fig.suptitle(
        f'Agenticity vs reward scale  (Gaussian, S={S_scale}, T fixed, n={N_SCALE})\n'
        f'Lower R_scale = flatter reward landscape → planning should matter less',
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
    save(fig, '08_rscale_sweep_agenticity.png')

    print("\nAll plots saved to environments/figures/")