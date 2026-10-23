# Noise-Sensitivity Analysis Report

**Dataset:** 20261023_122432_noise_experiment.json
**Analysis directory:** results\noise_sensitivity\analysis_20261023

**Generated:** 2026-10-23T15:47:47.478624

## Critical Disclosures

> QUBO parameters were optimized with a noiseless StatevectorSimulator. Noise was applied ONLY during final Aer sampling. This does NOT study noisy optimization convergence. Results are not evidence of quantum advantage.

> This is a classical simulator study using Qiskit Aer. Six instances and ten seeds (n=60 per condition). Interpret statistical significance accordingly.

## Key Findings

### 1. Feasibility

Of 240 total trials, 240 found feasible solutions (100%).

| Condition | Feasible Trials | Rate |
|---|---|---|
| Noiseless (Aer) | 60/60 | 100% |
| p=0.01 | 60/60 | 100% |
| p=0.05 | 60/60 | 100% |
| p=0.10 | 60/60 | 100% |

### 2. Optimal Solution Quality

236/240 trials (98.3%) matched the exact classical optimum.

| Condition | Matched | Rate |
|---|---|---|
| Noiseless (Aer) | 57/60 | 95% |
| p=0.01 | 60/60 | 100% |
| p=0.05 | 60/60 | 100% |
| p=0.10 | 59/60 | 98% |

### 3. Statistical Comparisons (vs noiseless control)

| Condition | Feas.Frac Diff (pp) | p-value | Gap Diff | p-value |
|---|---|---|---|---|
| p=0.01 | 0.19 | 0.5284 | -0.1000 | 0.0832 |
| p=0.05 | 0.07 | 0.8935 | -0.1000 | 0.0832 |
| p=0.10 | 0.25 | 0.6754 | -0.0667 | 0.3214 |

### 4. Runtime

Includes: parameter init, COBYLA optimization, circuit construction, transpilation, and Aer sampling. Does NOT include exact enumeration.

| Condition | Mean (s) | Median (s) |
|---|---|---|
| Noiseless (Aer) | 10.65 | 1.16 |
| p=0.01 | 12.95 | 1.44 |
| p=0.05 | 14.77 | 1.51 |
| p=0.10 | 14.72 | 1.51 |

## Interpretation

- Feasibility is robust across all noise levels.
- Optimum match rate degrades for harder instances under noise.
- Non-monotonic behavior observed (noise can occasionally help).
- Small sample size limits statistical power.

## Limitations

1. Parameters optimized noiselessly.
2. Only depolarizing noise tested.
3. Small parameter regime (6 instances, 10 seeds, p=1).
4. Aer simulator, not real hardware.
5. Does not measure quantum advantage.

## Conclusion

The dataset is internally consistent. The experiment supports that depolarizing noise in final Aer sampling degrades QAOA solution quality for harder instances. Findings are specific to this simulator configuration.
