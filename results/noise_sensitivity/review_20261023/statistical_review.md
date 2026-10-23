# Statistical Robustness and Reproducibility Review

**Dataset:** 20261023_122432_noise_experiment.json
**Review directory:** results/noise_sensitivity/review_20261023
**Generated:** 2026-10-23T16:36:08.263382

## Critical Disclosures

> QUBO parameters were optimized with a noiseless StatevectorSimulator. Noise was applied ONLY during final Aer sampling. This does NOT study noisy optimization convergence. These results are not evidence of quantum advantage.

> This is a classical simulator study using Qiskit Aer. Six instances and ten seeds (n=60 per condition) constitute a small study.

## 1. Data Integrity Verification

All 13 validation checks passed. The dataset is internally consistent.

## 2. Recomputed Aggregates

### Feasibility

| Condition | Feasible | Rate | Mean Frac | Std Frac |
|---|---|---|---|---|
| Noiseless | 60/60 | 100% | 0.1415 | 0.0852 |
| p=0.01 | 60/60 | 100% | 0.1434 | 0.0773 |
| p=0.05 | 60/60 | 100% | 0.1421 | 0.0661 |
| p=0.10 | 60/60 | 100% | 0.1440 | 0.0652 |

### Objective Gap

| Condition | Mean Gap | Median Gap | Std Gap |
|---|---|---|---|
| Noiseless | 0.1000 | 0.0000 | 0.4396 |
| p=0.01 | 0.0000 | 0.0000 | 0.0000 |
| p=0.05 | 0.0000 | 0.0000 | 0.0000 |
| p=0.10 | 0.0333 | 0.0000 | 0.2582 |

### Optimum Match Rate

| Condition | Matched | Rate |
|---|---|---|
| Noiseless | 57/60 | 95% |
| p=0.01 | 60/60 | 100% |
| p=0.05 | 60/60 | 100% |
| p=0.10 | 59/60 | 98% |

## 3. Experimental Design Analysis

### Structure
- 6 instances (7-17 qubits, varying difficulty)
- 10 seeds per instance (repeated across conditions)
- 4 noise conditions (noiseless, p=0.01, p=0.05, p=0.10)
- Pairing: identical instance + seed across conditions

### Implications

**Repeated seeds:** Same seed controls COBYLA init and Aer sampling seed. Creates strong pairing between conditions.

**60 paired trials vs 6 independent instances:** 60 observations cluster within 6 instances. Effective independence is n=6.

**Instance difficulty variation:** Hard 17-qubit instances show lower match rates (80-95%) in noiseless control vs 100% for easy instances.

## 4. Paired Differences (recomputed)

| Condition | Frac Diff (pp) | Gap Diff | Match Diff (pp) | Frac 95% CI (pp) | Gap 95% CI |
|---|---|---|---|---|---|
| p=0.01 | 0.20 | -0.0998 | 4.99 | [-0.53, 1.19] | [-0.2333, 0.0000] |
| p=0.05 | 0.06 | -0.0998 | 4.99 | [-0.80, 1.06] | [-0.2333, 0.0000] |
| p=0.10 | 0.25 | -0.0666 | 3.33 | [-0.79, 1.42] | [-0.1333, 0.0000] |

Bootstrap method: 10,000 iterations, resampling 6 instances with replacement.

## 5. Audit of Existing Paired Tests

The existing statistical_comparisons.csv uses paired t-tests on 60 trial-level observations.

**Issues:**
1. No multiple comparison correction (3 tests, Bonferroni alpha=0.0167)
2. Independence violated: 60 trials cluster within 6 instances
3. Nonsignificance != equivalence
4. No effect size CIs reported

**Verdict:** Defensible only as EXPLORATORY trial-level comparisons.

## 6. Objective Gap Claims

Gaps verified: objective_gap = best_feasible_objective - exact_objective
Uses original assignment objective, NOT QUBO energy. All 240 match.

**Nonzero gap trials:** 4 of 240. Control has 3 nonzero; noisy 0-1.
Negative gap difference not evidence noise improves performance; driven by chance in a small sample.

## 7. Reproducibility & Traceability

### Available Metadata
- Package versions: {'qiskit': '2.6.0', 'qiskit-aer': '0.17.2'}
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Noise conditions: [('noiseless_aer', 0.0), ('p=0.01', 0.01), ('p=0.05', 0.05), ('p=0.10', 0.1)]
- QAOA depth: 1
- Optimizer: COBYLA
- Maxiter: 50
- Shots: 512
- Basis gates: ['id', 'rz', 'sx', 'x', 'cx', 'rzz', 'measure']
- Exact optimum asymmetric_3job_3server: 6.0
- Exact optimum cost_asymmetry_2job_2server: 3.0
- Exact optimum medium_2job_2server: 5.0
- Exact optimum tight_capacity_3job_2server: 6.0
- Exact optimum tight_capacity_restricted: 2.0
- Exact optimum tiny_2job_2server: 5.0

### Missing Metadata
- Git commit hash / source code revision
- Numpy random state for StatevectorSampler
- Python version at experiment time
- OS / platform at experiment time
- CPU architecture (affects runtime, not results)
- Aer simulator seed transparency

### Assessment

Deterministic seeds used for COBYLA init and Aer sampling. Bit-for-bit reproducible on same machine (qiskit 2.6.0). Cross-platform NOT guaranteed.

## 8. Conclusions

### 1. Internal Consistency
PASS. All 13 checks pass. Recomputed aggregates match existing.

### 2. Statistical Claims Need Revision
No multiple comparison correction; conflates trial-level with instance-level inference.

### 3. Conclusions Supported by Data
- All 240 trials feasible across all conditions
- 98.3% match exact optimum overall
- Feasible fraction unaffected by noise
- Gaps remain near zero

### 4. Conclusions Remaining Uncertain
- Systematic noise degradation (CIs include zero)
- Hard-instance significance (n=6)
- Generalization to other configs

### 5. Recommended Wording

```
We evaluated QAOA on six Q-Grid assignment instances under depolarizing noise applied during final Aer sampling (p=0.0, 0.01, 0.05, 0.10). Parameters were optimized with a noiseless StatevectorSampler; noise affected only the sampling step. Across 240 trials, all found feasible solutions, and 236/240 (98.3%) matched the exact optimum. Feasible sample fractions showed no significant differences (paired t-tests, all p>0.5; bootstrap 95% CIs include zero). Noiseless control achieved lower match rates (80-95%) on hard 17-qubit instances. With only 6 independent instances, power is limited. Results are simulator-specific and do not constitute evidence of quantum advantage.
```
