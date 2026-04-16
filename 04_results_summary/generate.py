"""
generate.py — Results summary document + metric-organised figures.

Loads pre-computed results from experiments 1 and 2, then produces:
  results/summary.md     — tables organised by metric then test type
  figures/A_adv_gap.pdf  — adv_gap across all test conditions
  figures/B_vstar_var.pdf
  figures/C_mce_entropy.pdf
  figures/D_h_eps.pdf
  figures/E_composite.pdf

Each figure has 6 panels (2 rows × 3 cols):
  Row 1: Q3 human MDPs (bar) | Q1 distributions by R_type | Gaussian σ sweep
  Row 2: p-sweep (Bernoulli + spike_slab) | γ-sweep | S-sweep

Where a metric is missing (e.g. mce_entropy in old builds), panels show "no data".
Plot references (e.g. ← Plot 03) are included in figure titles.

Usage:
  python generate.py
Requires:
  ../01_pam_baseline/results/pam_experiment.pkl
  ../02_sampling_sweeps/results/sweep_results.pkl
"""
import sys, os, pickle, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environments'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR     = os.path.join(_SCRIPT_DIR, 'figures')
RES_DIR     = os.path.join(_SCRIPT_DIR, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# (key in score dict,  column index in Q1 matrix,  display label,  figure letter)
METRICS = [
    ('adv_gap_norm',      0, 'Advantage Gap (norm)',         'A'),
    ('vstar_var_norm',    1, 'V*−V^rand Variance (norm)',    'B'),
    ('mce_entropy_norm',  4, 'MCE Entropy norm (1−H/logA)', 'C'),
    ('H_eps_norm',        3, 'Planning Horizon H_eps (norm)','D'),
    ('composite',        -1, 'Composite Score',              'E'),
]

R_COLORS = {
    'gaussian':   '#4C72B0',
    'uniform':    '#DD8452',
    'bernoulli':  '#55A868',
    'spike_slab': '#C44E52',
    'goal':       '#8172B2',
    'potential':  '#777777',
}
T_COLORS = {
    'uniform':       '#C44E52',
    'dirichlet':     '#4C72B0',
    'deterministic': '#55A868',
}
MDP_COLOR = '#5A9BD5'
HIGH_AGENCY = 0.5

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _vals(scores, key):
    """Extract float array for a metric key from a list of score dicts."""
    arr = np.array([s.get(key, float('nan')) for s in scores], dtype=float)
    return arr[~np.isnan(arr)]


def _stats(arr):
    """Return (mean, std, pct_above_0.5) or (nan, nan, nan) for empty array."""
    if len(arr) == 0:
        return float('nan'), float('nan'), float('nan')
    return float(arr.mean()), float(arr.std()), float((arr >= HIGH_AGENCY).mean())


def _q1_col(mat, col_idx):
    """Extract a column from a Q1 matrix, dropping NaNs."""
    if col_idx < 0 or col_idx >= mat.shape[1]:
        return np.array([])
    col = mat[:, col_idx].astype(float)
    return col[~np.isnan(col)]


def _no_data_ax(ax, msg='no data'):
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=8, color='#999')
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    p1_path = os.path.join(_SCRIPT_DIR, '..', '01_pam_baseline',
                           'results', 'pam_experiment.pkl')
    p2_path = os.path.join(_SCRIPT_DIR, '..', '02_sampling_sweeps',
                           'results', 'sweep_results.pkl')
    p1, p2 = None, None
    if os.path.exists(p1_path):
        with open(p1_path, 'rb') as f:
            p1 = pickle.load(f)
        print(f"  loaded {p1_path}")
    else:
        print(f"  MISSING: {p1_path}")
    if os.path.exists(p2_path):
        with open(p2_path, 'rb') as f:
            p2 = pickle.load(f)
        print(f"  loaded {p2_path}")
    else:
        print(f"  MISSING: {p2_path}")
    return p1, p2


# ---------------------------------------------------------------------------
# Panel drawing functions
# ---------------------------------------------------------------------------

def panel_q3(ax, p1, key, col_idx, title):
    """Horizontal bar chart: one bar per human MDP, sorted descending by this metric."""
    if p1 is None:
        return _no_data_ax(ax, 'experiment 1 not built')
    q3 = p1['q3']
    names, vals = [], []
    for name, r in q3.items():
        v = r.get(key, float('nan')) if key != 'composite' else r.get('composite', float('nan'))
        names.append(name)
        vals.append(float(v) if v is not None else float('nan'))
    # Sort descending
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals  = [vals[i]  for i in order]
    colors = [MDP_COLOR if v >= HIGH_AGENCY else '#AAAAAA' for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.6)
    ax.axvline(HIGH_AGENCY, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Score', fontsize=7)
    ax.set_title(f'Q3 Human MDPs  ← Plot 01/02', fontsize=7)
    ax.invert_yaxis()


def panel_q1(ax, p1, key, col_idx, title):
    """Box/violin distributions by R_type at S=10."""
    if p1 is None:
        return _no_data_ax(ax, 'experiment 1 not built')
    q1 = p1['q1']
    S_use = 10
    # Collect R_types at S=10
    r_types = sorted(set(rt for (S, rt) in q1.keys() if S == S_use))
    data, labels, colors = [], [], []
    for rt in r_types:
        mat = q1.get((S_use, rt))
        if mat is None:
            continue
        col = _q1_col(mat, col_idx) if col_idx >= 0 else _vals(
            [{'composite': float(v)} for v in mat[:, 0]], 'composite')
        if col_idx < 0:
            # composite not in Q1 matrix — skip
            continue
        data.append(col)
        labels.append(rt)
        colors.append(R_COLORS.get(rt, '#888'))
    if not data:
        return _no_data_ax(ax, 'no Q1 data for this metric')
    bp = ax.boxplot(data, patch_artist=True, vert=True, widths=0.5,
                    medianprops=dict(color='black', lw=1.2),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    flierprops=dict(marker='o', markersize=2, alpha=0.4))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(HIGH_AGENCY, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel('Score', fontsize=7)
    ax.set_title(f'Q1 by R_type  (S={S_use})  ← Plot 03/07', fontsize=7)


def panel_rscale(ax, p1, key, col_idx, title):
    """Gaussian σ sweep (R_scale sweep)."""
    if p1 is None or 'rscale_sweep' not in p1:
        return _no_data_ax(ax, 'no R_scale sweep data')
    sweep = p1['rscale_sweep']
    scales = sorted(sweep.keys())
    means, stds = [], []
    for sc in scales:
        mat = sweep[sc]
        col = _q1_col(mat, col_idx) if col_idx >= 0 else np.array([])
        m, s, _ = _stats(col)
        means.append(m); stds.append(s)
    means, stds = np.array(means), np.array(stds)
    valid = ~np.isnan(means)
    if not valid.any():
        return _no_data_ax(ax, 'all NaN (metric not computed in this build)')
    sc_arr = np.array(scales)
    ax.plot(sc_arr[valid], means[valid], 'o-', color='#4C72B0', lw=1.5)
    ax.fill_between(sc_arr[valid],
                    (means - stds)[valid], (means + stds)[valid],
                    color='#4C72B0', alpha=0.15)
    ax.axhline(HIGH_AGENCY, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xscale('log'); ax.set_xticks(scales); ax.set_xticklabels(scales, fontsize=6.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('σ (Gaussian R_scale)', fontsize=7)
    ax.set_ylabel('Mean ± std', fontsize=7)
    ax.set_title('Gaussian σ sweep  ← Plot 08', fontsize=7)


def panel_psweep(ax, p2, key, title):
    """Bernoulli + spike_slab p-sweep on the same axes."""
    if p2 is None:
        return _no_data_ax(ax, 'experiment 2 not built')
    datasets = [
        ('bernoulli',  p2.get('bern_data', {}), '#4C72B0', 'Bernoulli'),
        ('spike_slab', p2.get('ss_data',   {}), '#C44E52', 'Spike-slab'),
    ]
    plotted = False
    for _, data, color, label in datasets:
        if not data:
            continue
        ps = sorted(data.keys())
        means, stds = [], []
        for p in ps:
            arr = _vals(data[p], key)
            m, s, _ = _stats(arr)
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        valid = ~np.isnan(means)
        if not valid.any():
            continue
        p_arr = np.array(ps)
        ax.plot(p_arr[valid], means[valid], 'o-', color=color, lw=1.5, label=label)
        ax.fill_between(p_arr[valid],
                        (means-stds)[valid], (means+stds)[valid],
                        color=color, alpha=0.15)
        plotted = True
    if not plotted:
        return _no_data_ax(ax, 'all NaN')
    ax.axhline(HIGH_AGENCY, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('p (sparsity)', fontsize=7)
    ax.set_ylabel('Mean ± std', fontsize=7)
    ax.legend(fontsize=6)
    ax.set_title('p-sweep: Bernoulli & spike_slab  ← Plots 09/10', fontsize=7)


def panel_gamma(ax, p2, key, title):
    """γ-sweep: one line per R_condition."""
    if p2 is None or 'gamma_data' not in p2:
        return _no_data_ax(ax, 'experiment 2 not built')
    gdata  = p2['gamma_data']
    gparams = p2.get('gamma_params', {})
    gammas = gparams.get('gammas', sorted(set(k[2] for k in gdata)))
    r_conds = gparams.get('R_conditions', sorted(set((k[0], k[1]) for k in gdata)))
    R_COND_COLORS = {'spike_slab': '#C44E52', 'gaussian': '#4C72B0'}
    plotted = False
    for R_type, R_scale in r_conds:
        means, stds = [], []
        for g in gammas:
            arr = _vals(gdata.get((R_type, R_scale, g), []), key)
            m, s, _ = _stats(arr)
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        valid = ~np.isnan(means)
        if not valid.any():
            continue
        g_arr = np.array(gammas)
        color = R_COND_COLORS.get(R_type, '#888')
        ax.plot(g_arr[valid], means[valid], 'o-', color=color, lw=1.5,
                label=f'{R_type}')
        ax.fill_between(g_arr[valid],
                        (means-stds)[valid], (means+stds)[valid],
                        color=color, alpha=0.15)
        plotted = True
    if not plotted:
        return _no_data_ax(ax, 'all NaN')
    ax.axhline(HIGH_AGENCY, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('γ', fontsize=7)
    ax.set_ylabel('Mean ± std', fontsize=7)
    ax.legend(fontsize=6)
    ax.set_title('γ-sweep  ← Plot 11', fontsize=7)


def panel_ssweep(ax, p2, key, title):
    """S-sweep: one line per T_type (uniform excluded from default, but shown if present)."""
    if p2 is None or 's_data' not in p2:
        return _no_data_ax(ax, 'experiment 2 not built')
    sdata   = p2['s_data']
    sparams = p2.get('s_params', {})
    S_vals  = sparams.get('S_values', sorted(set(k[0] for k in sdata)))
    T_types = sparams.get('T_types',  sorted(set(k[1] for k in sdata)))
    T_LABELS = {
        'uniform':       'Uniform (1/S) [uninformative]',
        'dirichlet':     'Dirichlet(α=0.1)',
        'deterministic': 'Deterministic',
    }
    T_STYLES = {'uniform': '--', 'dirichlet': '-', 'deterministic': '-'}
    plotted = False
    for T_type in T_types:
        means, stds = [], []
        for S in S_vals:
            arr = _vals(sdata.get((S, T_type), []), key)
            m, s, _ = _stats(arr)
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        valid = ~np.isnan(means)
        if not valid.any():
            continue
        S_arr = np.array(S_vals)
        color = T_COLORS.get(T_type, '#888')
        ls = T_STYLES.get(T_type, '-')
        ax.plot(S_arr[valid], means[valid], f'o{ls}', color=color, lw=1.2,
                label=T_LABELS.get(T_type, T_type), alpha=0.7 if T_type == 'uniform' else 1.0)
        ax.fill_between(S_arr[valid],
                        (means-stds)[valid], (means+stds)[valid],
                        color=color, alpha=0.1)
        plotted = True
    if not plotted:
        return _no_data_ax(ax, 'all NaN')
    ax.axhline(HIGH_AGENCY, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks(S_vals); ax.set_xticklabels(S_vals, fontsize=6.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('S (state space size)', fontsize=7)
    ax.set_ylabel('Mean ± std', fontsize=7)
    ax.legend(fontsize=5.5)
    ax.set_title('S-sweep  ← Plot 13', fontsize=7)


# ---------------------------------------------------------------------------
# Build one figure per metric
# ---------------------------------------------------------------------------

def make_figure(metric_key, col_idx, metric_label, letter, p1, p2):
    fig = plt.figure(figsize=(15, 8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_q3    = fig.add_subplot(gs[0, 0])
    ax_q1    = fig.add_subplot(gs[0, 1])
    ax_rscl  = fig.add_subplot(gs[0, 2])
    ax_psw   = fig.add_subplot(gs[1, 0])
    ax_gamma = fig.add_subplot(gs[1, 1])
    ax_ssw   = fig.add_subplot(gs[1, 2])

    panel_q3(ax_q3,    p1, metric_key, col_idx, metric_label)
    panel_q1(ax_q1,    p1, metric_key, col_idx, metric_label)
    panel_rscale(ax_rscl, p1, metric_key, col_idx, metric_label)
    panel_psweep(ax_psw,  p2, metric_key, metric_label)
    panel_gamma(ax_gamma, p2, metric_key, metric_label)
    panel_ssweep(ax_ssw,  p2, metric_key, metric_label)

    fig.suptitle(f'{letter}. {metric_label}', fontsize=12, fontweight='bold', y=1.01)

    path = os.path.join(FIG_DIR, f'{letter}_{metric_key}.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Build summary markdown document
# ---------------------------------------------------------------------------

def fmt(v, decimals=3):
    return 'N/A' if (v is None or np.isnan(v)) else f'{v:.{decimals}f}'


def table_row(*cells):
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


def table_header(*cols):
    hdr = table_row(*cols)
    sep = '| ' + ' | '.join('---' for _ in cols) + ' |'
    return hdr + '\n' + sep


def build_doc(p1, p2):
    lines = []

    lines.append('# Results Summary')
    lines.append('')
    lines.append('Organised by metric. Each section covers the same metric across all test')
    lines.append('conditions. Plot references in parentheses indicate the corresponding figure')
    lines.append('in the experiment outputs.')
    lines.append('')
    lines.append(f'Threshold for "high agenticity": ≥ {HIGH_AGENCY}')
    lines.append('')

    for metric_key, col_idx, metric_label, letter in METRICS:
        lines.append(f'---')
        lines.append('')
        lines.append(f'## {letter}. {metric_label}')
        lines.append('')

        # ---- Q3 Human MDPs ----
        lines.append('### Q3 — Human MDPs  (← Plots 01/02)')
        lines.append('')
        if p1 is not None:
            q3 = p1['q3']
            rows = []
            for name, r in q3.items():
                v = r.get(metric_key)
                rows.append((name, float(v) if v is not None else float('nan')))
            rows.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -999)
            lines.append(table_header('MDP', metric_label, f'≥ {HIGH_AGENCY}?'))
            for name, v in rows:
                lines.append(table_row(name, fmt(v), '✓' if not np.isnan(v) and v >= HIGH_AGENCY else ''))
        else:
            lines.append('*Experiment 1 not built.*')
        lines.append('')

        # ---- Q1 baseline ----
        lines.append('### Q1 — Baseline by R_type  (← Plot 03/07, S=10, T fixed)')
        lines.append('')
        if p1 is not None and col_idx >= 0:
            q1 = p1['q1']
            S_use = 10
            r_types = sorted(set(rt for (S, rt) in q1.keys() if S == S_use))
            lines.append(table_header('R_type', 'mean', 'std', f'% ≥ {HIGH_AGENCY}'))
            for rt in r_types:
                mat = q1.get((S_use, rt))
                if mat is None:
                    continue
                col = _q1_col(mat, col_idx)
                m, s, pct = _stats(col)
                lines.append(table_row(rt, fmt(m), fmt(s), fmt(pct*100, 1) + '%'))
        else:
            lines.append('*Not available for this metric.*')
        lines.append('')

        # ---- R_scale sweep ----
        lines.append('### Gaussian σ sweep  (← Plot 08)')
        lines.append('')
        if p1 is not None and 'rscale_sweep' in p1 and col_idx >= 0:
            sweep = p1['rscale_sweep']
            lines.append(table_header('σ', 'mean', 'std', f'% ≥ {HIGH_AGENCY}'))
            for sc in sorted(sweep.keys()):
                col = _q1_col(sweep[sc], col_idx)
                m, s, pct = _stats(col)
                lines.append(table_row(sc, fmt(m), fmt(s), fmt(pct*100, 1) + '%'))
        else:
            lines.append('*Not available.*')
        lines.append('')

        # ---- p-sweep Bernoulli ----
        lines.append('### p-sweep — Bernoulli  (← Plot 09)')
        lines.append('')
        if p2 is not None and 'bern_data' in p2:
            bdata = p2['bern_data']
            lines.append(table_header('p', 'mean', 'std', f'% ≥ {HIGH_AGENCY}'))
            for p in sorted(bdata.keys()):
                arr = _vals(bdata[p], metric_key)
                m, s, pct = _stats(arr)
                lines.append(table_row(p, fmt(m), fmt(s), fmt(pct*100, 1) + '%'))
        else:
            lines.append('*Experiment 2 not built.*')
        lines.append('')

        # ---- p-sweep spike_slab ----
        lines.append('### p-sweep — Spike-and-slab  (← Plot 10)')
        lines.append('')
        if p2 is not None and 'ss_data' in p2:
            ssdata = p2['ss_data']
            lines.append(table_header('p', 'mean', 'std', f'% ≥ {HIGH_AGENCY}'))
            for p in sorted(ssdata.keys()):
                arr = _vals(ssdata[p], metric_key)
                m, s, pct = _stats(arr)
                lines.append(table_row(p, fmt(m), fmt(s), fmt(pct*100, 1) + '%'))
        else:
            lines.append('*Experiment 2 not built.*')
        lines.append('')

        # ---- γ-sweep ----
        lines.append('### γ-sweep  (← Plot 11)')
        lines.append('')
        if p2 is not None and 'gamma_data' in p2:
            gdata   = p2['gamma_data']
            gparams = p2.get('gamma_params', {})
            gammas  = gparams.get('gammas', sorted(set(k[2] for k in gdata)))
            r_conds = gparams.get('R_conditions', sorted(set((k[0],k[1]) for k in gdata)))
            col_headers = ['γ'] + [f'{rt} (scale={sc})' for rt, sc in r_conds]
            lines.append(table_header(*col_headers))
            for g in gammas:
                row = [g]
                for R_type, R_scale in r_conds:
                    arr = _vals(gdata.get((R_type, R_scale, g), []), metric_key)
                    m, _, _ = _stats(arr)
                    row.append(fmt(m))
                lines.append(table_row(*row))
        else:
            lines.append('*Experiment 2 not built.*')
        lines.append('')

        # ---- T-sensitivity (correlation summary) ----
        lines.append('### T-sensitivity  (← Plot 12, Pearson r between T types)')
        lines.append('')
        if p2 is not None and 't_data' in p2:
            tdata   = p2['t_data']
            tparams = p2.get('t_params', {})
            T_types = tparams.get('T_types', ['uniform', 'dirichlet', 'deterministic'])
            r_conds = tparams.get('R_conditions', sorted(tdata.keys()))
            col_headers = ['R_type']
            pairs = [(T_types[i], T_types[j])
                     for i in range(len(T_types))
                     for j in range(i+1, len(T_types))]
            col_headers += [f'r({a} vs {b})' for a, b in pairs]
            lines.append(table_header(*col_headers))
            for R_type, R_scale in r_conds:
                paired = tdata.get((R_type, R_scale), {})
                row = [f'{R_type} (scale={R_scale})']
                for Ta, Tb in pairs:
                    xa = _vals(paired.get(Ta, []), metric_key)
                    xb = _vals(paired.get(Tb, []), metric_key)
                    n = min(len(xa), len(xb))
                    if n >= 3:
                        r = float(np.corrcoef(xa[:n], xb[:n])[0, 1])
                        row.append(fmt(r, 2))
                    else:
                        row.append('N/A')
                lines.append(table_row(*row))
        else:
            lines.append('*Experiment 2 not built.*')
        lines.append('')

        # ---- S-sweep ----
        lines.append('### S-sweep  (← Plot 13, Uniform T excluded — see notes)')
        lines.append('')
        if p2 is not None and 's_data' in p2:
            sdata   = p2['s_data']
            sparams = p2.get('s_params', {})
            S_vals  = sparams.get('S_values', sorted(set(k[0] for k in sdata)))
            T_types_s = [t for t in sparams.get('T_types',
                         sorted(set(k[1] for k in sdata)))
                         if t != 'uniform']
            col_headers = ['S'] + T_types_s
            lines.append(table_header(*col_headers))
            for S in S_vals:
                row = [S]
                for T_type in T_types_s:
                    arr = _vals(sdata.get((S, T_type), []), metric_key)
                    m, _, _ = _stats(arr)
                    row.append(fmt(m))
                lines.append(table_row(*row))
        else:
            lines.append('*Experiment 2 not built.*')
        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*Note: "N/A" entries indicate the metric was not computed in the stored build')
    lines.append('(e.g. mce_entropy with compute_entropy=False, or mi_diff with compute_mi=False).*')
    lines.append('*Uniform T excluded from S-sweep tables: range-normalisation makes adv_gap and*')
    lines.append('*vstar_var S-invariant under Uniform T (numerator and denominator both scale as*')
    lines.append('*σ/√S and cancel). Uniform T values are still plotted with a dashed line in figures.*')

    doc_path = os.path.join(RES_DIR, 'summary.md')
    with open(doc_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  saved → {doc_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading data...')
    p1, p2 = load_data()

    print('Generating figures...')
    for metric_key, col_idx, metric_label, letter in METRICS:
        print(f'  Figure {letter}: {metric_label}')
        make_figure(metric_key, col_idx, metric_label, letter, p1, p2)

    print('Generating summary document...')
    build_doc(p1, p2)

    print(f'\nDone.')
    print(f'  Figures  → {FIG_DIR}')
    print(f'  Document → {RES_DIR}/summary.md')


if __name__ == '__main__':
    main()
