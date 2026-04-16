"""
runners.py — Batch experiment runners.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core import MDP
from pams import agenticity_score
from envs import make_chain_mdp, make_grid_mdp
from metrics import control_advantage, one_step_recovery  # noqa: F401 (re-export)


# ---------------------------------------------------------------------------
# Random MDP Generator
# ---------------------------------------------------------------------------

def random_mdp(
    S: int,
    A: int,
    gamma: float = 0.95,
    k: int = 3,
    R_type: str = 'gaussian',
    R_scale: float = 1.0,
    terminal_states: int = 1,
    T_type: str = 'random',
    T_alpha: float = 0.1,
    rng: np.random.Generator = None,
) -> MDP:
    """
    Generate a random tabular MDP.

    Transition types (T_type):
      'random':      for each (s,a), sample k successors from non-terminal states,
                     assign Dirichlet(1,...,1) weights. (default)
      'uniform':     T[s,a,s'] = 1/S for all s' (fully mixing).
      'dirichlet':   like 'random' but uses Dirichlet(T_alpha,...,T_alpha) weights.
                     Low T_alpha (e.g. 0.1) → highly concentrated / near-deterministic.
      'deterministic': each (s,a) transitions to a single randomly chosen successor.

    T_alpha: concentration parameter for 'dirichlet' T_type (default 0.1).

    Reward types (R_type):
      'gaussian':   R[s,a,s'] ~ N(0, R_scale)
      'uniform':    R[s,a,s'] ~ Uniform(0, 1)
      'bernoulli':  R[s,a,s'] = 1 w.p. R_scale, else 0
                    (R_scale is the sparsity; default 0.1 for sparse rewards)
      'spike_slab': R[s,a,s'] = mask * N(0,1), mask ~ Bernoulli(R_scale).
                    R_scale = p = sparsity probability; non-zero magnitude ~ N(0,1).
      'potential':  R(s,a,s') = gamma*Phi(s') - Phi(s), Phi(s) ~ N(0, R_scale).
                    Non-agentic by construction — does not change policy ordering.
                    Useful as a negative control: PAMs should score these low.
      'goal':          R(s,a,s') = 1[s'=g] for a uniformly random non-terminal goal g.
                       Goal-conditioned sparse reward; structurally realistic for
                       navigation domains. Agenticity varies with goal location.
      'uniform_simplex': Dirichlet(1,...,1) over all S×A×S entries — uniform prior
                       over the reward simplex. Non-negative, sums to 1 across all
                       (s,a,s') triples. Same prior as Turner et al. (2021).
                       R_scale is ignored for this type.

    Terminal states: last `terminal_states` indices {S-terminal_states, ..., S-1}.
    d0: uniform over non-terminal states.
    """
    if rng is None:
        rng = np.random.default_rng()

    terminal = set(range(S - terminal_states, S)) if terminal_states > 0 else set()
    non_terminal = [s for s in range(S) if s not in terminal]

    def _build_T():
        T = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                if s in terminal:
                    T[s, a, s] = 1.0
                    continue
                pool = non_terminal if non_terminal else list(range(S))
                if T_type == 'uniform':
                    T[s, a, :] = 1.0 / S
                elif T_type == 'deterministic':
                    T[s, a, int(rng.choice(pool))] = 1.0
                elif T_type == 'dirichlet':
                    k_eff = min(k, len(pool))
                    successors = rng.choice(pool, size=k_eff, replace=False)
                    T[s, a, successors] = rng.dirichlet(np.full(k_eff, T_alpha))
                else:  # 'random'
                    k_eff = min(k, len(pool))
                    successors = rng.choice(pool, size=k_eff, replace=False)
                    T[s, a, successors] = rng.dirichlet(np.ones(k_eff))
        return T

    def _find_reachable_goal(T):
        """BFS backwards from each candidate goal; return first reachable from all non-terminal states, or None."""
        adj = (T > 0).any(axis=1)  # (S, S)
        pool = non_terminal if non_terminal else list(range(S))
        for candidate in rng.permutation(pool):
            can_reach = {int(candidate)}
            frontier = [int(candidate)]
            while frontier:
                nxt = []
                for tgt in frontier:
                    for s in np.where(adj[:, tgt])[0]:
                        if s not in can_reach:
                            can_reach.add(int(s))
                            nxt.append(int(s))
                frontier = nxt
            if all(s in can_reach for s in non_terminal):
                return int(candidate)
        return None

    if R_type == 'goal':
        # Resample T until a goal reachable from all non-terminal states exists.
        for _ in range(100):
            T = _build_T()
            g = _find_reachable_goal(T)
            if g is not None:
                break
        else:
            raise RuntimeError(
                f"random_mdp: could not find a T with a universally reachable goal "
                f"after 100 attempts (S={S}, A={A}, T_type={T_type!r}, k={k}). "
                "Try increasing k or using T_type='random'."
            )
    else:
        T = _build_T()

    if R_type == 'gaussian':
        R = rng.normal(0, R_scale, size=(S, A, S))
    elif R_type == 'uniform':
        R = rng.uniform(0, 1, size=(S, A, S))
    elif R_type == 'bernoulli':
        R = (rng.uniform(0, 1, size=(S, A, S)) < R_scale).astype(float)
    elif R_type == 'potential':
        Phi = rng.normal(0, R_scale, size=S)
        # F[s,a,s'] = gamma*Phi[s'] - Phi[s]; action-independent shaping
        R = gamma * Phi[None, None, :] - Phi[:, None, None]
        R = np.broadcast_to(R, (S, A, S)).copy()
    elif R_type == 'goal':
        # g was selected during T-sampling above (guaranteed reachable from all non-terminal states)
        R = np.zeros((S, A, S))
        R[:, :, g] = 1.0
    elif R_type == 'spike_slab':
        mask = (rng.uniform(0, 1, size=(S, A, S)) < R_scale).astype(float)
        R = mask * rng.normal(0, 1.0, size=(S, A, S))
    elif R_type == 'uniform_simplex':
        # Uniform prior over the reward simplex: Dirichlet(1,...,1) over all S×A×S entries.
        # Non-negative, sums to 1 across all (s,a,s') triples. Same prior as Turner et al. (2021).
        flat = rng.dirichlet(np.ones(S * A * S))
        R = flat.reshape(S, A, S)
    else:
        raise ValueError(
            f"Unknown R_type: {R_type!r}. "
            "Choose 'gaussian', 'uniform', 'bernoulli', 'potential', 'goal', 'spike_slab', "
            "or 'uniform_simplex'."
        )

    for s in terminal:
        R[s, :, :] = 0.0

    d0 = np.zeros(S)
    if non_terminal:
        d0[non_terminal] = 1.0 / len(non_terminal) # ensure you start off in a non-terminal state, if any exist
    else:
        d0[:] = 1.0 / S

    return MDP(S=S, A=A, T=T, R=R, gamma=gamma, terminal=terminal, d0=d0)


# ---------------------------------------------------------------------------
# Baseline metric normalisation utility
# ---------------------------------------------------------------------------

def norm_baseline(ctrl_adv, one_step, scales):
    """Normalise baseline metrics using empirical 95th-pct scales: 1-exp(-x/scale)."""
    def _n(x, s):
        return float(1 - np.exp(-x / s)) if s > 1e-10 else 0.0
    return {
        'ctrl_adv_norm': _n(ctrl_adv, scales.get('ctrl_adv', 1.0)),
        'one_step_norm': _n(one_step, scales.get('one_step', 1.0)),
    }


# ---------------------------------------------------------------------------
# Batch PAM Experiment (Q1/Q2/Q3)
# ---------------------------------------------------------------------------

def run_pam_experiment(
    n_random_mdps: int = 200,
    S_values: list = None,
    A: int = 4,
    gamma: float = 0.95,
    k: int = 3,
    R_types: list = None,
    R_scales: dict = None,
    T_structures: list = None,
    n_fixed_T: int = 10,
    rng_seed: int = 42,
    verbose: bool = False,
    include_mi: bool = False,
) -> dict:
    """
    Batch experiment comparing PAMs across random and human-made MDPs.

    Q1 — PAM correlation (T fixed, R varies):
      For each (S, R_type): fix one canonical T, sample n_random_mdps R's,
      compute all PAMs. q1 stores norm PAM matrix; q1_raw stores raw values.

    Q2 — T-dependence (variance decomposition):
      For each (S, R_type, T_structure): sample n_fixed_T transition matrices
      with the given topology, for each T sample n_R=50 R's and compute
      composite. Reports within-T variance (R's share) and between-T variance
      (T's share). T_structures can include 'random', 'chain', 'grid'.

    Q3 — Human-made vs random:
      Run agenticity_score on the 7 built-in MDPs. Results include n_actions
      so the MCE entropy |A| confound is visible in the printout.

    R_scales: per-type R_scale overrides, e.g. {'bernoulli': 0.5}.
              Default: {'bernoulli': 0.5} to avoid all-zero reward matrices.
    T_structures: list of topology names for Q2. Default: ['random'].
                  'chain' = linear chain (A0=advance, A1=retreat, A2=stay, A3=stochastic)
                  'grid'  = 2D grid with 4-directional actions
    include_mi: if True, compute early_action_mi (slow, n_episodes reduced to 300).
    """
    if S_values is None:
        S_values = [5, 10, 20]
    if R_types is None:
        R_types = ['gaussian', 'uniform', 'bernoulli', 'potential', 'goal']
    if R_scales is None:
        R_scales = {'bernoulli': 0.5}
    if T_structures is None:
        T_structures = ['random']

    rng = np.random.default_rng(rng_seed)
    PAM_NAMES = ['adv_gap', 'vstar_var', 'mi_diff', 'h_eff', 'mce_entropy']

    def _score_mdp(mdp, n_ep=500):
        return agenticity_score(
            mdp, verbose=False, rng=rng,
            compute_mi=include_mi,
            n_episodes=(300 if include_mi else n_ep),
        )

    def _nan(v):
        return float('nan') if v is None else v

    def _pam_vector_norm(r):
        return np.array([
            r['adv_gap_norm'],
            r['vstar_var_norm'],
            _nan(r['mi_diff']),
            r['H_eps_norm'],
            _nan(r['mce_entropy_norm']),
        ], dtype=float)

    def _pam_vector_raw(r):
        return np.array([
            r['adv_gap'],
            r['vstar_var_raw'],
            _nan(r['mi_diff']),
            r['H_eps'],
            _nan(r['mce_entropy_raw']),
            r['ctrl_adv'],
            r['one_step_recovery'],
        ], dtype=float)

    def _r_scale_for(R_type):
        return R_scales.get(R_type, 1.0)

    def _make_structured_T(S, structure, terminal, local_rng):
        """Build a topology-specific transition matrix."""
        non_term = [s for s in range(S) if s not in terminal]
        T = np.zeros((S, A, S))

        # 'dirichlet_{alpha}' — Dirichlet(alpha,...,alpha) weights over k successors.
        # 'random' is an alias for 'dirichlet_1.0'.
        if structure == 'random' or structure.startswith('dirichlet_'):
            alpha = 1.0
            if structure.startswith('dirichlet_'):
                alpha = float(structure.split('_', 1)[1])
            for s in range(S):
                for a in range(A):
                    if s in terminal:
                        T[s, a, s] = 1.0
                        continue
                    pool = non_term if non_term else list(range(S))
                    k_eff = min(k, len(pool))
                    succ = local_rng.choice(pool, size=k_eff, replace=False)
                    T[s, a, succ] = local_rng.dirichlet(np.full(k_eff, alpha))

        elif structure == 'chain':
            # A0=advance, A1=retreat, A2=stay, A3=noisy-advance
            max_nt = max(non_term) if non_term else S - 1
            min_nt = min(non_term) if non_term else 0
            for s in range(S):
                if s in terminal:
                    T[s, :, s] = 1.0
                    continue
                s_fwd  = min(s + 1, max_nt) if (s + 1) not in terminal else s
                s_back = max(s - 1, min_nt) if (s - 1) not in terminal else s
                T[s, 0, s_fwd]  = 1.0   # advance
                T[s, 1, s_back] = 1.0   # retreat
                T[s, 2, s]      = 1.0   # stay
                # noisy advance: if at boundary s_fwd==s, just stay
                if s_fwd != s:
                    T[s, 3, s_fwd] = 0.7
                    T[s, 3, s]     = 0.3
                else:
                    T[s, 3, s] = 1.0

        elif structure == 'grid':
            rows = max(2, int(np.round(np.sqrt(S - len(terminal)))))
            cols = max(2, (S - len(terminal) + rows - 1) // rows)
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up/down/left/right
            for s in non_term:
                r_s, c_s = s // cols, s % cols
                for a, (dr, dc) in enumerate(deltas[:A]):
                    nr = max(0, min(rows - 1, r_s + dr))
                    nc = max(0, min(cols - 1, c_s + dc))
                    s_next = nr * cols + nc
                    if s_next >= S or s_next in terminal:
                        s_next = s  # boundary: stay
                    T[s, a, s_next] += 1.0
            for s in terminal:
                T[s, :, s] = 1.0
        else:
            raise ValueError(f"Unknown T_structure: {structure!r}. "
                             "Choose 'random', 'dirichlet_{{alpha}}', 'chain', or 'grid'.")
        return T

    # ------------------------------------------------------------------
    # Q1 — fix T, vary R
    # ------------------------------------------------------------------
    q1 = {}
    q1_raw = {}
    for S in S_values:
        for R_type in R_types:
            if verbose:
                print(f"Q1: S={S}, R_type={R_type}")

            canonical = random_mdp(S, A, gamma=gamma, k=k,
                                   R_type='gaussian', terminal_states=1,
                                   rng=np.random.default_rng(0))
            T_fixed = canonical.T
            terminal_fixed = canonical.terminal
            d0_fixed = canonical.d0

            pam_norm = []
            pam_raw  = []
            for _ in range(n_random_mdps):
                tmp_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                tmp = random_mdp(S, A, gamma=gamma, k=k,
                                 R_type=R_type,
                                 R_scale=_r_scale_for(R_type),
                                 terminal_states=1,
                                 rng=tmp_rng)
                tmp_mdp = MDP(S=S, A=A, T=T_fixed, R=tmp.R,
                              gamma=gamma, terminal=terminal_fixed,
                              d0=d0_fixed.copy())
                r = _score_mdp(tmp_mdp)
                pam_norm.append(_pam_vector_norm(r))
                pam_raw.append(_pam_vector_raw(r))

            q1[(S, R_type)]     = np.array(pam_norm)
            q1_raw[(S, R_type)] = np.array(pam_raw)

    # ------------------------------------------------------------------
    # Compute empirical 95th-percentile scales from Q1 raw data
    # ------------------------------------------------------------------
    def _compute_baseline_scales(q1_raw_dict, percentile=95):
        pools = [[], []]  # ctrl_adv, one_step
        for mat in q1_raw_dict.values():
            for i in range(5, 7):
                col = mat[:, i]
                pools[i-5].extend(col[~np.isnan(col)].tolist())
        keys = ['ctrl_adv', 'one_step']
        return {k: float(np.percentile(v, percentile)) if v else 1.0
                for k, v in zip(keys, pools)}

    w2_scales = _compute_baseline_scales(q1_raw)

    # ------------------------------------------------------------------
    # Q2 — variance decomposition: T vs R, across T_structures
    # ------------------------------------------------------------------
    q2 = {}
    n_R = 50
    for structure in T_structures:
        for S in S_values:
            for R_type in R_types:
                if verbose:
                    print(f"Q2: structure={structure}, S={S}, R_type={R_type}")

                terminal_q2 = set(range(S - 1, S))
                non_term_q2 = [s for s in range(S) if s not in terminal_q2]
                d0_q2 = np.zeros(S)
                d0_q2[non_term_q2] = 1.0 / len(non_term_q2)

                group_means = []
                group_composites_all = []

                for _ in range(n_fixed_T):
                    t_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                    T_t = _make_structured_T(S, structure, terminal_q2, t_rng)

                    composites = []
                    for _ in range(n_R):
                        r_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                        tmp = random_mdp(S, A, gamma=gamma, k=k,
                                         R_type=R_type,
                                         R_scale=_r_scale_for(R_type),
                                         terminal_states=1,
                                         rng=r_rng)
                        mdp_t = MDP(S=S, A=A, T=T_t, R=tmp.R,
                                    gamma=gamma, terminal=terminal_q2,
                                    d0=d0_q2.copy())
                        r = _score_mdp(mdp_t)
                        composites.append(r['composite'])

                    composites = np.array(composites)
                    group_means.append(composites.mean())
                    group_composites_all.append(composites)

                group_means = np.array(group_means)
                within_var  = float(np.mean([np.var(c) for c in group_composites_all]))
                between_var = float(np.var(group_means))
                total_var   = within_var + between_var
                ratio = between_var / total_var if total_var > 1e-12 else float('nan')

                q2[(structure, S, R_type)] = {
                    'within_var':  within_var,
                    'between_var': between_var,
                    'ratio':       ratio,
                }

    # ------------------------------------------------------------------
    # Q3 — human-made MDPs
    # ------------------------------------------------------------------
    human_cases = [
        ("Chain-Terminal",  make_chain_mdp(10, reward_type='terminal')),
        ("Chain-BkTrack",   make_chain_mdp(10, reward_type='terminal', backtrack=True)),
        ("Chain-Dense",     make_chain_mdp(10, reward_type='dense')),
        ("Chain-Lottery",   make_chain_mdp(10, reward_type='lottery')),
        ("Chain-Progress",  make_chain_mdp(10, reward_type='progress')),
        ("Grid-Goal",       make_grid_mdp(5, 5, reward_type='goal')),
        ("Grid-Local",      make_grid_mdp(5, 5, reward_type='local')),
        ("Grid-Cliff",      make_grid_mdp(5, 5, reward_type='cliff')),
    ]

    q3 = {}
    for name, mdp in human_cases:
        if verbose:
            print(f"  scoring {name}...")
        r = _score_mdp(mdp)
        q3[name] = {**r, **norm_baseline(r['ctrl_adv'], r['one_step_recovery'],
                                          w2_scales),
                    'n_actions': mdp.A}

    meta = {
        'S_values': S_values,
        'n_random_mdps': n_random_mdps,
        'PAM_names': PAM_NAMES,
        'include_mi': include_mi,
        'rng_seed': rng_seed,
        'noisy_mi_estimates': include_mi,
        'R_scales': R_scales,
        'T_structures': T_structures,
        'baseline_empirical_scales': w2_scales,
    }

    results = {'q1': q1, 'q1_raw': q1_raw, 'q2': q2, 'q3': q3, 'meta': meta}

    if verbose:
        _print_pam_results(results)

    return results


def _print_pam_results(results: dict) -> None:
    """Pretty-print Q1/Q2/Q3 results from run_pam_experiment."""
    PAM_COLS = ['adv_gap', 'vstar_var', 'mi_diff', 'H_eps', 'mce_ent']
    PAM_COLS_RAW = ['adv_gap', 'vstar_var', 'mi_diff', 'H_eps', 'mce_ent', 'ctrl_adv', 'one_step']
    W = 9  # column width

    def _fmt(v, w, decimals=4):
        return f"{'N/A':>{w}s}" if (v is None or np.isnan(v)) else f"{v:>{w}.{decimals}f}"

    def _col_stats(mat):
        """Per-column mean and std, skipping NaN, without triggering warnings."""
        means = np.full(mat.shape[1], np.nan)
        stds  = np.full(mat.shape[1], np.nan)
        for i in range(mat.shape[1]):
            valid = mat[:, i][~np.isnan(mat[:, i])]
            if len(valid) > 0:
                means[i] = valid.mean()
                stds[i]  = valid.std()
        return means, stds

    # ------------------------------------------------------------------
    # Q1 — PAM distributions and correlations (norm + raw)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1 — PAM DISTRIBUTIONS (T fixed, R varies)")
    print("    norm = normalised [0,1]  |  raw = original scale")
    print("=" * 70)
    hdr = f"  {'':18s}" + "".join(f"{n:>{W}s}" for n in PAM_COLS)

    for (S, R_type), mat in sorted(results['q1'].items()):
        label = f"S={S} {R_type}"
        norm_means, norm_stds = _col_stats(mat)
        raw_mat = results['q1_raw'].get((S, R_type))

        print(f"\n  {label}")
        print(hdr)
        print(f"  {'norm mean':18s}" + "".join(_fmt(v, W) for v in norm_means))
        print(f"  {'norm std':18s}" + "".join(_fmt(v, W) for v in norm_stds))
        if raw_mat is not None:
            raw_means, raw_stds = _col_stats(raw_mat)
            # Show all 8 raw columns
            hdr_raw = f"  {'':18s}" + "".join(f"{n:>{W}s}" for n in PAM_COLS_RAW)
            print(hdr_raw)
            print(f"  {'raw mean':18s}" + "".join(_fmt(v, W) for v in raw_means))
            print(f"  {'raw std':18s}" + "".join(_fmt(v, W) for v in raw_stds))

        # Pearson correlation on norm matrix (skip all-NaN / zero-variance cols)
        valid_cols = [i for i in range(mat.shape[1])
                      if not np.isnan(norm_stds[i]) and norm_stds[i] > 1e-10]
        if len(valid_cols) >= 2:
            corr = np.corrcoef(mat[:, valid_cols].T)
            col_names = [PAM_COLS[i] for i in valid_cols]
            print(f"  {'Pearson corr':18s}[{', '.join(col_names)}]")
            for i, row in enumerate(corr):
                print(f"    {col_names[i]:<14s}" + "".join(_fmt(v, W, 3) for v in row))

    # ------------------------------------------------------------------
    # Q2 — variance decomposition (grouped by T_structure)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q2 — T vs R VARIANCE DECOMPOSITION")
    print("  ratio = between_T_var / total_var  (>0.5 → T dominates)")
    print("=" * 70)
    print(f"  {'T_struct':<10s} {'S':<5s} {'R_type':<12s} "
          f"{'within_var':>11s} {'between_var':>12s} {'ratio':>7s}  verdict")
    print("  " + "-" * 70)
    for (structure, S, R_type), v in sorted(results['q2'].items()):
        ratio = v['ratio']
        verdict = "T dominates" if (not np.isnan(ratio) and ratio > 0.5) else "R dominates"
        ratio_str = _fmt(ratio, 7, 3)
        print(f"  {structure:<10s} {S:<5d} {R_type:<12s} "
              f"{v['within_var']:>11.5f} {v['between_var']:>12.5f} {ratio_str}  {verdict}")

    # ------------------------------------------------------------------
    # Q3 — human-made MDPs  (note |A| confound for MCE entropy)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q3 — HUMAN-MADE MDPs")
    print("  Note: chains have A=2, grids have A=4. MCE entropy is H/log(A),")
    print("  so grids mechanically score higher — control for |A| before comparing.")
    print("=" * 70)
    score_cols = ['composite', 'adv_gap_norm', 'vstar_var_norm', 'H_eps_norm', 'mce_entropy_norm',
                  'ctrl_adv_norm', 'one_step_norm']
    hdrs3 = ['comp', 'adv_gap', 'v*var', 'h_eff', 'mce_ent', 'ctrl_adv', 'one_step']
    print(f"  {'MDP':<22s} {'|A|':>4s}" + "".join(f"{h:>9s}" for h in hdrs3))
    print("  " + "-" * 86)
    rows = [(name, r) for name, r in results['q3'].items()]
    for name, r in sorted(rows, key=lambda x: -x[1]['composite']):
        a_str = f"{r['n_actions']:>4d}"
        vals = "".join(f"{r[c]:>9.4f}" for c in score_cols)
        print(f"  {name:<22s} {a_str}{vals}")

    print(f"\n  MI column: {'included' if results['meta']['include_mi'] else 'excluded (compute_mi=False)'}.")


# ---------------------------------------------------------------------------
# Group 1: p-sweep helper (bernoulli and spike_slab)
# ---------------------------------------------------------------------------

def run_p_sweep(
    R_type: str,
    p_values: list,
    S: int = 10,
    A: int = 4,
    gamma: float = 0.95,
    k: int = 3,
    n_mdps: int = 50,
    rng_seed: int = 42,
) -> dict:
    """
    Group 1: fix one canonical T, sweep R_scale=p for R_type in p_values.
    Returns {p: [agenticity_score dicts]}.
    """
    rng = np.random.default_rng(rng_seed)
    canonical = random_mdp(S, A, gamma=gamma, k=k, R_type='gaussian',
                           terminal_states=1, rng=np.random.default_rng(0))
    T_fixed, term_fixed, d0_fixed = canonical.T, canonical.terminal, canonical.d0
    results = {}
    for p in p_values:
        scores = []
        for _ in range(n_mdps):
            tmp = random_mdp(S, A, gamma=gamma, k=k, R_type=R_type, R_scale=p,
                             terminal_states=1,
                             rng=np.random.default_rng(int(rng.integers(0, 2**31))))
            mdp = MDP(S=S, A=A, T=T_fixed, R=tmp.R, gamma=gamma,
                      terminal=term_fixed, d0=d0_fixed.copy())
            scores.append(agenticity_score(mdp, verbose=False, compute_mi=False,
                                           rng=np.random.default_rng(int(rng.integers(0, 2**31)))))
        results[p] = scores
    return results


# ---------------------------------------------------------------------------
# Group 3a: γ sweep
# ---------------------------------------------------------------------------

def run_gamma_sweep(
    gammas: list = None,
    S: int = 20,
    A: int = 4,
    k: int = 3,
    R_conditions: list = None,
    n_mdps: int = 50,
    T_type: str = 'random',
    T_alpha: float = 0.1,
    rng_seed: int = 42,
) -> dict:
    """
    Group 3a: for each (R_type, R_scale, gamma) triple, sample n_mdps MDPs and
    compute agenticity scores.
    Returns {(R_type, R_scale, gamma): [score dicts]}.
    """
    if gammas is None:
        gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    if R_conditions is None:
        R_conditions = [('spike_slab', 0.1), ('gaussian', 1.0)]
    rng = np.random.default_rng(rng_seed)
    results = {}
    for R_type, R_scale in R_conditions:
        for gamma in gammas:
            scores = []
            for _ in range(n_mdps):
                mdp = random_mdp(S, A, gamma=gamma, k=k,
                                 R_type=R_type, R_scale=R_scale, terminal_states=1,
                                 T_type=T_type, T_alpha=T_alpha,
                                 rng=np.random.default_rng(int(rng.integers(0, 2**31))))
                scores.append(agenticity_score(mdp, verbose=False, compute_mi=False,
                                               rng=np.random.default_rng(int(rng.integers(0, 2**31)))))
            results[(R_type, R_scale, gamma)] = scores
    return results


# ---------------------------------------------------------------------------
# Group 2: T-sensitivity
# ---------------------------------------------------------------------------

def run_t_sensitivity(
    n_R: int = 100,
    S: int = 20,
    A: int = 4,
    gamma: float = 0.95,
    k: int = 3,
    R_conditions: list = None,
    T_types: list = None,
    T_alpha: float = 0.1,
    rng_seed: int = 42,
) -> dict:
    """
    Group 2: for each R sample, evaluate agenticity under multiple T structures.

    Fixes n_R reward matrices per (R_type, R_scale) then scores each R under
    every T_type independently. Index i in each T_type list corresponds to the
    same R_i, enabling paired scatter plots.

    Returns {(R_type, R_scale): {T_type: [score dicts]}}.
    """
    if R_conditions is None:
        R_conditions = [('gaussian', 1.0), ('spike_slab', 0.05)]
    if T_types is None:
        T_types = ['uniform', 'dirichlet', 'deterministic']
    rng = np.random.default_rng(rng_seed)
    terminal = set(range(S - 1, S))
    non_terminal = [s for s in range(S) if s not in terminal]
    d0 = np.zeros(S)
    d0[non_terminal] = 1.0 / len(non_terminal)

    results = {}
    for R_type, R_scale in R_conditions:
        R_list = []
        for _ in range(n_R):
            tmp = random_mdp(S, A, gamma=gamma, k=k,
                             R_type=R_type, R_scale=R_scale, terminal_states=1,
                             rng=np.random.default_rng(int(rng.integers(0, 2**31))))
            R_list.append(tmp.R.copy())

        paired = {T_type: [] for T_type in T_types}
        for R_mat in R_list:
            for T_type in T_types:
                tmp_t = random_mdp(S, A, gamma=gamma, k=k,
                                   R_type='gaussian', terminal_states=1,
                                   T_type=T_type, T_alpha=T_alpha,
                                   rng=np.random.default_rng(int(rng.integers(0, 2**31))))
                mdp = MDP(S=S, A=A, T=tmp_t.T, R=R_mat, gamma=gamma,
                          terminal=terminal, d0=d0.copy())
                paired[T_type].append(
                    agenticity_score(mdp, verbose=False, compute_mi=False,
                                     rng=np.random.default_rng(int(rng.integers(0, 2**31))))
                )
        results[(R_type, R_scale)] = paired
    return results


# ---------------------------------------------------------------------------
# Group 3b: S sweep
# ---------------------------------------------------------------------------

def run_s_sweep(
    S_values: list = None,
    A: int = 4,
    gamma: float = 0.95,
    k: int = 3,
    R_type: str = 'gaussian',
    R_scale: float = 1.0,
    T_types: list = None,
    T_alpha: float = 0.1,
    n_mdps: int = 50,
    rng_seed: int = 42,
) -> dict:
    """
    Group 3b: sweep S_values × T_types. Returns {(S, T_type): [score dicts]}.
    """
    if S_values is None:
        S_values = [5, 10, 20, 50, 100]
    if T_types is None:
        T_types = ['uniform', 'dirichlet', 'deterministic']
    rng = np.random.default_rng(rng_seed)
    results = {}
    for S in S_values:
        for T_type in T_types:
            scores = []
            for _ in range(n_mdps):
                mdp = random_mdp(S, A, gamma=gamma, k=k,
                                 R_type=R_type, R_scale=R_scale, terminal_states=1,
                                 T_type=T_type, T_alpha=T_alpha,
                                 rng=np.random.default_rng(int(rng.integers(0, 2**31))))
                scores.append(agenticity_score(mdp, verbose=False, compute_mi=False,
                                               rng=np.random.default_rng(int(rng.integers(0, 2**31)))))
            results[(S, T_type)] = scores
    return results


# ---------------------------------------------------------------------------
# Fraction-agentic estimator (Turner et al. 2021 comparison)
# ---------------------------------------------------------------------------

def fraction_agentic(
    prior: str,
    threshold: float = 0.6,
    n_samples: int = 200,
    S: int = 10,
    A: int = 4,
    gamma: float = 0.95,
    k: int = 3,
    R_scale: float = 1.0,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
    **mdp_kwargs,
) -> dict:
    """
    Estimate P(composite > threshold) under a given reward prior, with bootstrap CI.

    Parameters
    ----------
    prior       : R_type string passed to random_mdp — e.g. 'gaussian',
                  'spike_slab', or 'uniform_simplex' (Turner et al. 2021).
    threshold   : composite score cutoff defining "agentic" (default 0.6).
    n_samples   : number of MDPs sampled from the prior.
    S, A, gamma : MDP dimensions and discount factor.
    k           : transition fan-out for random_mdp.
    R_scale     : R_scale argument forwarded to random_mdp (ignored for
                  'uniform_simplex', where the scale is fixed by the simplex).
    n_bootstrap : bootstrap resamples for the 95 % CI.
    rng_seed    : master RNG seed.
    **mdp_kwargs: extra keyword arguments forwarded to random_mdp.

    Returns
    -------
    dict with keys:
        point_estimate  — fraction of samples with composite > threshold
        ci_lower        — 2.5th bootstrap percentile
        ci_upper        — 97.5th bootstrap percentile
        threshold       — threshold used
        prior           — prior name used
        n_samples       — number of MDP samples drawn
        composites      — array of all composite scores (length n_samples)
    """
    rng = np.random.default_rng(rng_seed)
    composites = []

    for _ in range(n_samples):
        mdp = random_mdp(
            S, A, gamma=gamma, k=k,
            R_type=prior, R_scale=R_scale,
            terminal_states=1,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
            **mdp_kwargs,
        )
        r = agenticity_score(
            mdp, verbose=False, compute_mi=False,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )
        composites.append(r['composite'])

    composites = np.array(composites)
    point_est = float((composites > threshold).mean())

    # Non-parametric bootstrap CI
    boot_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    boot_means = np.array([
        (boot_rng.choice(composites, size=n_samples, replace=True) > threshold).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        'point_estimate': round(point_est, 4),
        'ci_lower':       round(ci_lower, 4),
        'ci_upper':       round(ci_upper, 4),
        'threshold':      threshold,
        'prior':          prior,
        'n_samples':      n_samples,
        'composites':     composites,
    }
