# Results Summary

Organised by metric. Each section covers the same metric across all test
conditions. Plot references in parentheses indicate the corresponding figure
in the experiment outputs.

Threshold for "high agenticity": ≥ 0.5

---

## A. Advantage Gap (norm)

### Q3 — Human MDPs  (← Plots 01/02)

| MDP | Advantage Gap (norm) | ≥ 0.5? |
| --- | --- | --- |
| Chain-Dense | 0.371 |  |
| Grid-Goal | 0.331 |  |
| Chain-Terminal | 0.218 |  |
| Chain-Lottery | 0.218 |  |
| Chain-BkTrack | 0.207 |  |
| Chain-Progress | 0.192 |  |
| Grid-Local | 0.186 |  |
| Grid-Cliff | 0.099 |  |

### Q1 — Baseline by R_type  (← Plot 03/07, S=10, T fixed)

| R_type | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| bernoulli | 0.993 | 0.032 | 100.0% |
| gaussian | 0.971 | 0.073 | 100.0% |
| goal | 0.966 | 0.077 | 100.0% |
| potential | 0.000 | 0.000 | 0.0% |
| uniform | 0.962 | 0.088 | 100.0% |

### Gaussian σ sweep  (← Plot 08)

| σ | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.1 | 0.961 | 0.097 | 100.0% |
| 0.5 | 0.961 | 0.101 | 98.0% |
| 1.0 | 0.960 | 0.096 | 100.0% |
| 2.0 | 0.941 | 0.125 | 100.0% |
| 5.0 | 0.995 | 0.017 | 100.0% |

### p-sweep — Bernoulli  (← Plot 09)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.413 | 0.333 | 62.0% |
| 0.05 | 0.797 | 0.149 | 100.0% |
| 0.2 | 0.910 | 0.133 | 100.0% |
| 0.5 | 0.983 | 0.050 | 100.0% |
| 1.0 | 0.000 | 0.000 | 0.0% |

### p-sweep — Spike-and-slab  (← Plot 10)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.384 | 0.320 | 50.0% |
| 0.05 | 0.732 | 0.157 | 96.0% |
| 0.2 | 0.862 | 0.155 | 100.0% |
| 0.5 | 0.930 | 0.112 | 100.0% |
| 1.0 | 0.941 | 0.118 | 100.0% |

### γ-sweep  (← Plot 11)

| γ | spike_slab (scale=0.1) | gaussian (scale=1.0) |
| --- | --- | --- |
| 0.5 | 0.597 | 0.973 |
| 0.7 | 0.598 | 0.952 |
| 0.8 | 0.635 | 0.941 |
| 0.9 | 0.658 | 0.913 |
| 0.95 | 0.670 | 0.911 |
| 0.99 | 0.676 | 0.913 |

### T-sensitivity  (← Plot 12, Pearson r between T types)

| R_type | r(uniform vs dirichlet) | r(uniform vs deterministic) | r(dirichlet vs deterministic) |
| --- | --- | --- | --- |
| spike_slab (scale=0.1) | 0.18 | 0.12 | 0.10 |
| gaussian (scale=1.0) | 0.06 | 0.03 | 0.08 |

### S-sweep  (← Plot 13, Uniform T excluded — see notes)

| S | dirichlet | deterministic |
| --- | --- | --- |
| 5 | 0.991 | 0.990 |
| 10 | 0.983 | 0.942 |
| 20 | 0.913 | 0.924 |
| 50 | 0.781 | 0.824 |
| 100 | 0.680 | 0.674 |

---

## B. V*−V^rand Variance (norm)

### Q3 — Human MDPs  (← Plots 01/02)

| MDP | V*−V^rand Variance (norm) | ≥ 0.5? |
| --- | --- | --- |
| Chain-Dense | 0.426 |  |
| Chain-Terminal | 0.422 |  |
| Chain-Lottery | 0.422 |  |
| Chain-Progress | 0.416 |  |
| Chain-BkTrack | 0.405 |  |
| Grid-Cliff | 0.328 |  |
| Grid-Local | 0.320 |  |
| Grid-Goal | 0.271 |  |

### Q1 — Baseline by R_type  (← Plot 03/07, S=10, T fixed)

| R_type | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| bernoulli | 0.407 | 0.073 | 8.0% |
| gaussian | 0.395 | 0.074 | 10.0% |
| goal | 0.374 | 0.064 | 0.0% |
| potential | 0.000 | 0.000 | 0.0% |
| uniform | 0.355 | 0.054 | 0.0% |

### Gaussian σ sweep  (← Plot 08)

| σ | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.1 | 0.390 | 0.079 | 12.0% |
| 0.5 | 0.388 | 0.073 | 6.0% |
| 1.0 | 0.378 | 0.067 | 6.0% |
| 2.0 | 0.378 | 0.080 | 8.0% |
| 5.0 | 0.397 | 0.076 | 8.0% |

### p-sweep — Bernoulli  (← Plot 09)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.231 | 0.184 | 0.0% |
| 0.05 | 0.393 | 0.071 | 8.0% |
| 0.2 | 0.400 | 0.088 | 12.0% |
| 0.5 | 0.395 | 0.074 | 6.0% |
| 1.0 | 0.000 | 0.000 | 0.0% |

### p-sweep — Spike-and-slab  (← Plot 10)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.223 | 0.178 | 0.0% |
| 0.05 | 0.380 | 0.055 | 4.0% |
| 0.2 | 0.389 | 0.080 | 10.0% |
| 0.5 | 0.374 | 0.072 | 8.0% |
| 1.0 | 0.381 | 0.070 | 4.0% |

### γ-sweep  (← Plot 11)

| γ | spike_slab (scale=0.1) | gaussian (scale=1.0) |
| --- | --- | --- |
| 0.5 | 0.282 | 0.293 |
| 0.7 | 0.268 | 0.289 |
| 0.8 | 0.279 | 0.286 |
| 0.9 | 0.274 | 0.276 |
| 0.95 | 0.274 | 0.283 |
| 0.99 | 0.267 | 0.280 |

### T-sensitivity  (← Plot 12, Pearson r between T types)

| R_type | r(uniform vs dirichlet) | r(uniform vs deterministic) | r(dirichlet vs deterministic) |
| --- | --- | --- | --- |
| spike_slab (scale=0.1) | -0.05 | 0.06 | 0.00 |
| gaussian (scale=1.0) | -0.20 | -0.02 | 0.10 |

### S-sweep  (← Plot 13, Uniform T excluded — see notes)

| S | dirichlet | deterministic |
| --- | --- | --- |
| 5 | 0.622 | 0.607 |
| 10 | 0.410 | 0.406 |
| 20 | 0.282 | 0.303 |
| 50 | 0.206 | 0.209 |
| 100 | 0.166 | 0.163 |

---

## C. MCE Entropy norm (1−H/logA)

### Q3 — Human MDPs  (← Plots 01/02)

| MDP | MCE Entropy norm (1−H/logA) | ≥ 0.5? |
| --- | --- | --- |
| Chain-Terminal | N/A |  |
| Chain-BkTrack | N/A |  |
| Chain-Dense | N/A |  |
| Chain-Lottery | N/A |  |
| Chain-Progress | N/A |  |
| Grid-Goal | N/A |  |
| Grid-Local | N/A |  |
| Grid-Cliff | N/A |  |

### Q1 — Baseline by R_type  (← Plot 03/07, S=10, T fixed)

| R_type | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| bernoulli | N/A | N/A | N/A% |
| gaussian | N/A | N/A | N/A% |
| goal | N/A | N/A | N/A% |
| potential | N/A | N/A | N/A% |
| uniform | N/A | N/A | N/A% |

### Gaussian σ sweep  (← Plot 08)

| σ | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.1 | N/A | N/A | N/A% |
| 0.5 | N/A | N/A | N/A% |
| 1.0 | N/A | N/A | N/A% |
| 2.0 | N/A | N/A | N/A% |
| 5.0 | N/A | N/A | N/A% |

### p-sweep — Bernoulli  (← Plot 09)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | N/A | N/A | N/A% |
| 0.05 | N/A | N/A | N/A% |
| 0.2 | N/A | N/A | N/A% |
| 0.5 | N/A | N/A | N/A% |
| 1.0 | N/A | N/A | N/A% |

### p-sweep — Spike-and-slab  (← Plot 10)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | N/A | N/A | N/A% |
| 0.05 | N/A | N/A | N/A% |
| 0.2 | N/A | N/A | N/A% |
| 0.5 | N/A | N/A | N/A% |
| 1.0 | N/A | N/A | N/A% |

### γ-sweep  (← Plot 11)

| γ | spike_slab (scale=0.1) | gaussian (scale=1.0) |
| --- | --- | --- |
| 0.5 | N/A | N/A |
| 0.7 | N/A | N/A |
| 0.8 | N/A | N/A |
| 0.9 | N/A | N/A |
| 0.95 | N/A | N/A |
| 0.99 | N/A | N/A |

### T-sensitivity  (← Plot 12, Pearson r between T types)

| R_type | r(uniform vs dirichlet) | r(uniform vs deterministic) | r(dirichlet vs deterministic) |
| --- | --- | --- | --- |
| spike_slab (scale=0.1) | N/A | N/A | N/A |
| gaussian (scale=1.0) | N/A | N/A | N/A |

### S-sweep  (← Plot 13, Uniform T excluded — see notes)

| S | dirichlet | deterministic |
| --- | --- | --- |
| 5 | N/A | N/A |
| 10 | N/A | N/A |
| 20 | N/A | N/A |
| 50 | N/A | N/A |
| 100 | N/A | N/A |

---

## D. Planning Horizon H_eps (norm)

### Q3 — Human MDPs  (← Plots 01/02)

| MDP | Planning Horizon H_eps (norm) | ≥ 0.5? |
| --- | --- | --- |
| Chain-Dense | 0.927 | ✓ |
| Chain-Terminal | 0.116 |  |
| Chain-BkTrack | 0.116 |  |
| Chain-Lottery | 0.116 |  |
| Chain-Progress | 0.116 |  |
| Grid-Goal | 0.101 |  |
| Grid-Local | 0.101 |  |
| Grid-Cliff | 0.087 |  |

### Q1 — Baseline by R_type  (← Plot 03/07, S=10, T fixed)

| R_type | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| bernoulli | 0.855 | 0.004 | 100.0% |
| gaussian | 0.856 | 0.005 | 100.0% |
| goal | 0.855 | 0.000 | 100.0% |
| potential | 0.000 | 0.000 | 0.0% |
| uniform | 0.857 | 0.004 | 100.0% |

### Gaussian σ sweep  (← Plot 08)

| σ | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.1 | 0.856 | 0.004 | 100.0% |
| 0.5 | 0.856 | 0.003 | 100.0% |
| 1.0 | 0.856 | 0.003 | 100.0% |
| 2.0 | 0.857 | 0.004 | 100.0% |
| 5.0 | 0.855 | 0.000 | 100.0% |

### p-sweep — Bernoulli  (← Plot 09)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.532 | 0.416 | 62.0% |
| 0.05 | 0.857 | 0.004 | 100.0% |
| 0.2 | 0.860 | 0.007 | 100.0% |
| 0.5 | 0.855 | 0.005 | 100.0% |
| 1.0 | 0.000 | 0.000 | 0.0% |

### p-sweep — Spike-and-slab  (← Plot 10)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.528 | 0.413 | 62.0% |
| 0.05 | 0.856 | 0.006 | 100.0% |
| 0.2 | 0.857 | 0.005 | 100.0% |
| 0.5 | 0.857 | 0.006 | 100.0% |
| 1.0 | 0.857 | 0.006 | 100.0% |

### γ-sweep  (← Plot 11)

| γ | spike_slab (scale=0.1) | gaussian (scale=1.0) |
| --- | --- | --- |
| 0.5 | 0.335 | 0.333 |
| 0.7 | 0.481 | 0.474 |
| 0.8 | 0.588 | 0.586 |
| 0.9 | 0.750 | 0.746 |
| 0.95 | 0.859 | 0.856 |
| 0.99 | 0.996 | 0.996 |

### T-sensitivity  (← Plot 12, Pearson r between T types)

| R_type | r(uniform vs dirichlet) | r(uniform vs deterministic) | r(dirichlet vs deterministic) |
| --- | --- | --- | --- |
| spike_slab (scale=0.1) | -0.00 | 0.00 | 0.16 |
| gaussian (scale=1.0) | -0.00 | 0.00 | 0.04 |

### S-sweep  (← Plot 13, Uniform T excluded — see notes)

| S | dirichlet | deterministic |
| --- | --- | --- |
| 5 | 0.855 | 0.859 |
| 10 | 0.858 | 0.859 |
| 20 | 0.861 | 0.861 |
| 50 | 0.864 | 0.860 |
| 100 | 0.865 | 0.869 |

---

## E. Composite Score

### Q3 — Human MDPs  (← Plots 01/02)

| MDP | Composite Score | ≥ 0.5? |
| --- | --- | --- |
| Chain-Dense | 0.399 |  |
| Chain-Terminal | 0.320 |  |
| Chain-Lottery | 0.320 |  |
| Chain-BkTrack | 0.306 |  |
| Chain-Progress | 0.304 |  |
| Grid-Goal | 0.301 |  |
| Grid-Local | 0.253 |  |
| Grid-Cliff | 0.214 |  |

### Q1 — Baseline by R_type  (← Plot 03/07, S=10, T fixed)

*Not available for this metric.*

### Gaussian σ sweep  (← Plot 08)

*Not available.*

### p-sweep — Bernoulli  (← Plot 09)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.322 | 0.256 | 40.0% |
| 0.05 | 0.595 | 0.098 | 82.0% |
| 0.2 | 0.655 | 0.093 | 88.0% |
| 0.5 | 0.689 | 0.050 | 100.0% |
| 1.0 | 0.000 | 0.000 | 0.0% |

### p-sweep — Spike-and-slab  (← Plot 10)

| p | mean | std | % ≥ 0.5 |
| --- | --- | --- | --- |
| 0.01 | 0.304 | 0.245 | 34.0% |
| 0.05 | 0.556 | 0.097 | 70.0% |
| 0.2 | 0.626 | 0.101 | 86.0% |
| 0.5 | 0.652 | 0.071 | 94.0% |
| 1.0 | 0.661 | 0.079 | 92.0% |

### γ-sweep  (← Plot 11)

| γ | spike_slab (scale=0.1) | gaussian (scale=1.0) |
| --- | --- | --- |
| 0.5 | 0.439 | 0.633 |
| 0.7 | 0.433 | 0.621 |
| 0.8 | 0.457 | 0.614 |
| 0.9 | 0.466 | 0.595 |
| 0.95 | 0.472 | 0.597 |
| 0.99 | 0.471 | 0.597 |

### T-sensitivity  (← Plot 12, Pearson r between T types)

| R_type | r(uniform vs dirichlet) | r(uniform vs deterministic) | r(dirichlet vs deterministic) |
| --- | --- | --- | --- |
| spike_slab (scale=0.1) | 0.14 | 0.10 | 0.08 |
| gaussian (scale=1.0) | 0.02 | 0.05 | 0.10 |

### S-sweep  (← Plot 13, Uniform T excluded — see notes)

| S | dirichlet | deterministic |
| --- | --- | --- |
| 5 | 0.807 | 0.798 |
| 10 | 0.697 | 0.674 |
| 20 | 0.598 | 0.614 |
| 50 | 0.493 | 0.516 |
| 100 | 0.423 | 0.419 |

---

*Note: "N/A" entries indicate the metric was not computed in the stored build
(e.g. mce_entropy with compute_entropy=False, or mi_diff with compute_mi=False).*
*Uniform T excluded from S-sweep tables: range-normalisation makes adv_gap and*
*vstar_var S-invariant under Uniform T (numerator and denominator both scale as*
*σ/√S and cancel). Uniform T values are still plotted with a dashed line in figures.*
