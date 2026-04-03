"""
plot_sampling.py — pipeline2.md Group 1/2/3 experiment plots.

Plots:
  09 — Group 1: Bernoulli (spike-and-slab-const) p sweep
  10 — Group 1: spike-and-slab (mask × N(0,1)) p sweep
  11 — Group 3a: γ sweep — PAM scores vs discount factor
  12 — Group 2: T-sensitivity — same R scored under Uniform vs Deterministic T
  13 — Group 3b: S sweep — adv_gap and V* var vs state space size per T type

Run:  python environments/plot_sampling.py
Saves figures to environments/figures/*.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import MDP
from runners import (random_mdp, run_p_sweep, run_gamma_sweep,
                     run_t_sensitivity, run_s_sweep)
from pams import agenticity_score

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)

HIGH_AGENCY_THRESHOLD = 0.5
VIRIDIS = plt.cm.viridis

# Metrics available in every agenticity_score dict
METRICS = [
    ('adv_gap_norm',    'Advantage Gap (norm)'),
    ('vstar_var_norm',  'V*−V^rand Variance (norm)'),
    ('H_eps_norm',      'Planning Horizon H_eps (norm)'),
    ('mce_entropy_norm','MCE Entropy (norm)'),
]


# ---------------------------------------------------------------------------
# Shared: p-sweep histogram plot
# ---------------------------------------------------------------------------
def plot_p_sweep(data, p_values, title, filename):
    n_m = len(METRICS)
    colors = [VIRIDIS(i / max(len(p_values) - 1, 1)) for i in range(len(p_values))]
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5))
    for ax, (key, metric_title) in zip(axes, METRICS):
        for p, color in zip(p_values, colors):
            vals = np.array([r[key] for r in data[p]])
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


# ===========================================================================
# Plot 09 — Bernoulli p sweep
# ===========================================================================
print("Plot 09: Bernoulli (spike-and-slab-const) p sweep")
P_VALUES = [0.01, 0.05, 0.2, 0.5, 1.0]
bern_data = run_p_sweep('bernoulli', P_VALUES, n_mdps=50)
plot_p_sweep(
    bern_data, P_VALUES,
    'Group 1: Bernoulli (spike-and-slab-const) p sweep  (S=10, T fixed, n=50)\n'
    'R_scale=p = fraction non-zero entries; magnitude always 1',
    '09_bernoulli_p_sweep.png',
)


# ===========================================================================
# Plot 10 — spike-and-slab p sweep
# ===========================================================================
print("Plot 10: spike-and-slab p sweep")
ss_data = run_p_sweep('spike_slab', P_VALUES, n_mdps=50)
plot_p_sweep(
    ss_data, P_VALUES,
    'Group 1: spike-and-slab (mask × N(0,1)) p sweep  (S=10, T fixed, n=50)\n'
    'R_scale=p = sparsity probability; non-zero magnitude ~ N(0,1)',
    '10_spike_slab_p_sweep.png',
)


# ===========================================================================
# Plot 11 — γ sweep
# ===========================================================================
print("Plot 11: γ sweep")
GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
R_CONDITIONS = [('spike_slab', 0.1), ('gaussian', 1.0)]
gamma_data = run_gamma_sweep(gammas=GAMMAS, R_conditions=R_CONDITIONS, n_mdps=50)

R_COLORS  = {'spike_slab': '#4C72B0', 'gaussian': '#DD8452'}
R_LABELS  = {'spike_slab': 'spike_slab (p=0.1)', 'gaussian': 'gaussian (σ=1)'}

n_m = len(METRICS)
fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5))
for ax, (key, metric_title) in zip(axes, METRICS):
    for R_type, R_scale in R_CONDITIONS:
        means = np.array([np.mean([r[key] for r in gamma_data[(R_type, R_scale, g)]]) for g in GAMMAS])
        stds  = np.array([np.std( [r[key] for r in gamma_data[(R_type, R_scale, g)]]) for g in GAMMAS])
        color = R_COLORS[R_type]
        ax.plot(GAMMAS, means, 'o-', color=color, label=R_LABELS[R_type])
        ax.fill_between(GAMMAS, means - stds, means + stds, color=color, alpha=0.15)
    ax.axhline(HIGH_AGENCY_THRESHOLD, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('γ', fontsize=9)
    ax.set_ylabel('Mean score (±1 std)', fontsize=9)
    ax.set_title(metric_title, fontsize=9)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7)
fig.suptitle(
    'Group 3a: PAM scores vs γ  (S=20, A=4, T=random, n=50)\n'
    'Predicted: monotone increase for spike_slab (structured); flatter for gaussian',
    fontsize=10,
)
fig.tight_layout()
save(fig, '11_gamma_sweep.png')


# ===========================================================================
# Plot 12 — T-sensitivity scatter
# ===========================================================================
print("Plot 12: T-sensitivity")
t_data = run_t_sensitivity(n_R=100, R_conditions=R_CONDITIONS)

T_PAIR = ('uniform', 'deterministic')   # most extreme comparison
T_COLORS_R = {'gaussian': '#4C72B0', 'spike_slab': '#55A868'}
S_METRICS_T = METRICS[:2]              # adv_gap and vstar_var

n_rows = len(R_CONDITIONS)
n_cols = len(S_METRICS_T)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

for row, (R_type, R_scale) in enumerate(R_CONDITIONS):
    paired = t_data[(R_type, R_scale)]
    for col, (key, metric_title) in enumerate(S_METRICS_T):
        ax = axes[row, col]
        x = np.array([r[key] for r in paired[T_PAIR[0]]])
        y = np.array([r[key] for r in paired[T_PAIR[1]]])
        ax.scatter(x, y, alpha=0.4, s=18, color=T_COLORS_R.get(R_type, '#555'))
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.set_xlabel(f'Score under Uniform T', fontsize=8)
        ax.set_ylabel(f'Score under Deterministic T', fontsize=8)
        ax.set_title(f'{metric_title}\n{R_type} (scale={R_scale})  r={corr:.2f}', fontsize=8)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)

fig.suptitle(
    'Group 2: T-sensitivity — same R, scored under Uniform vs Deterministic T  (S=20, n=100)\n'
    'Diagonal = T-invariant PAM; scatter above diagonal = Deterministic T inflates score',
    fontsize=10,
)
fig.tight_layout()
save(fig, '12_t_sensitivity.png')


# ===========================================================================
# Plot 13 — S sweep
# ===========================================================================
print("Plot 13: S sweep")
S_VALUES = [5, 10, 20, 50, 100]
T_TYPES  = ['uniform', 'dirichlet', 'deterministic']
s_data = run_s_sweep(S_values=S_VALUES, T_types=T_TYPES, n_mdps=50)

T_COLORS_S = {'uniform': '#C44E52', 'dirichlet': '#4C72B0', 'deterministic': '#55A868'}
T_LABELS_S = {
    'uniform':     'Uniform T (1/S)',
    'dirichlet':   'Dirichlet(α=0.1)',
    'deterministic': 'Deterministic',
}
S_METRICS_SWEEP = METRICS[:2]   # adv_gap, vstar_var — predicted to collapse under Uniform T

fig, axes = plt.subplots(1, len(S_METRICS_SWEEP), figsize=(6 * len(S_METRICS_SWEEP), 5))
for ax, (key, metric_title) in zip(axes, S_METRICS_SWEEP):
    for T_type in T_TYPES:
        means = np.array([np.mean([r[key] for r in s_data[(S, T_type)]]) for S in S_VALUES])
        stds  = np.array([np.std( [r[key] for r in s_data[(S, T_type)]]) for S in S_VALUES])
        color = T_COLORS_S[T_type]
        ax.plot(S_VALUES, means, 'o-', color=color, label=T_LABELS_S[T_type])
        ax.fill_between(S_VALUES, means - stds, means + stds, color=color, alpha=0.15)
    ax.axhline(HIGH_AGENCY_THRESHOLD, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks(S_VALUES)
    ax.set_xticklabels(S_VALUES)
    ax.set_xlabel('S (state space size)', fontsize=9)
    ax.set_ylabel('Mean score (±1 std)', fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(metric_title, fontsize=9)
    ax.legend(fontsize=8)
fig.suptitle(
    'Group 3b: PAM scores vs S × T type  (Gaussian R σ=1, A=4, γ=0.95, n=50)\n'
    'Predicted: Uniform T → score collapses as S → ∞ (averaging suppresses Q-spread)',
    fontsize=10,
)
fig.tight_layout()
save(fig, '13_s_sweep.png')


print("\nAll sampling plots saved to environments/figures/")
