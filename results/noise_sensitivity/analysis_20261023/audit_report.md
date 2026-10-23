# Data-Integrity Audit Report

**Dataset:** `results\noise_sensitivity\20261023_122432_noise_experiment.json`

**Audit time:** 2026-10-23T15:47:47.460460
**Raw file unchanged:** YES (read-only)

## Summary

- Total trial records: 240
- Distinct instances: 6
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Noise conditions: ['noiseless_aer', 'p=0.01', 'p=0.05', 'p=0.10']
- QAOA depth: 1
- Shots per circuit: 512
- Technical failures: 0
- Successful trials: 240

## Validation Checks

- **240 trials**: PASS
- **Unique trial IDs**: PASS
- **Six instances**: PASS
- **Ten seeds**: PASS
- **Four noise conditions**: PASS
- **QAOA depth p=1**: PASS
- **512 shots**: PASS
- **All success**: PASS
- **All error_detail=None**: PASS
- **No null replaced with zero**: PASS
- **Exact optima match**: PASS
- **Gap consistency**: PASS
- **10 per group**: PASS

## Per-Instance Exact Optima

| Instance | Recorded | Reference | Match |
|---|---|---|---|
| asymmetric_3job_3server | 6.0 | 6.0 | YES |
| cost_asymmetry_2job_2server | 3.0 | 3.0 | YES |
| medium_2job_2server | 5.0 | 5.0 | YES |
| tight_capacity_3job_2server | 6.0 | 6.0 | YES |
| tight_capacity_restricted | 2.0 | 2.0 | YES |
| tiny_2job_2server | 5.0 | 5.0 | YES |

## Audit Log

```
[INFO] Loaded 240 trial records
[INFO] OK: 240 trial records
[INFO] OK: All 240 trial identifiers are unique
[INFO] OK: Six instances verified
[INFO] OK: Seeds 0-9 present
[INFO] OK: Four noise conditions verified
[INFO] OK: Each instance x noise-condition group has exactly 10 trials
[INFO] OK: QAOA depth p=1 for all trials
[INFO] OK: 512 shots per circuit for all trials
[INFO] OK: noise_p_depol for noiseless_aer = 0.0
[INFO] OK: noise_p_depol for p=0.01 = 0.01
[INFO] OK: noise_p_depol for p=0.05 = 0.05
[INFO] OK: noise_p_depol for p=0.10 = 0.1
[INFO] OK: All trials have status='success'
[INFO] OK: All error_detail fields are None
[INFO] OK: exact_objective for asymmetric_3job_3server matches reference
[INFO] OK: exact_objective for cost_asymmetry_2job_2server matches reference
[INFO] OK: exact_objective for medium_2job_2server matches reference
[INFO] OK: exact_objective for tight_capacity_3job_2server matches reference
[INFO] OK: exact_objective for tight_capacity_restricted matches reference
[INFO] OK: exact_objective for tiny_2job_2server matches reference
[INFO] OK: All objective_gap = best_feasible_objective - exact_objective
[INFO] OK: No null objectives replaced with zero
[INFO] OK: is_feasible consistent with feasible_sample_fraction
[INFO] DataFrame shape: (240, 21)
[INFO] Saved: results\noise_sensitivity\analysis_20261023\01_feasible_sample_fraction.png
[INFO] Saved: results\noise_sensitivity\analysis_20261023\02_feasible_trial_rate.png
[INFO] Saved: results\noise_sensitivity\analysis_20261023\03_optimum_match_rate.png
[INFO] Saved: results\noise_sensitivity\analysis_20261023\04_objective_gap.png
[INFO] Saved: results\noise_sensitivity\analysis_20261023\05_runtime.png
[INFO] Saved: results\noise_sensitivity\analysis_20261023\06_feasible_fraction_heatmap.png
[INFO] Saved CSV: results\noise_sensitivity\analysis_20261023\aggregate_results.csv (24 rows)
[INFO] Saved CSV: results\noise_sensitivity\analysis_20261023\overall_summary.csv (8 rows)
[INFO] Saved CSV: results\noise_sensitivity\analysis_20261023\statistical_comparisons.csv (3 rows)
```

## Anomalies Found

None. All 240 trial records passed every validation check.
