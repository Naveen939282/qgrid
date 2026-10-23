# Provenance Manifest — QAOA Noise-Sensitivity Experiment

## Raw Data

| Field | Value |
|---|---|
| **Filename** | `results/noise_sensitivity/20261023_122432_noise_experiment.json` |
| **File size** | 250,035 bytes |
| **Trial count** | 240 (60 per condition) |
| **Schema version** | Not explicitly versioned in JSON |
| **Trial design** | 6 instances × 4 noise conditions × 10 seeds = 240 trials |

## Execution Environment

| Field | Value | Status |
|---|---|---|
| **Baseline commit** | `7bb2eda06e0f5508527faed7dca21477f62a15bd` | Baseline/last-known commit. The exact execution commit is **unknown** and not recorded in the JSON. |
| **Python version** | 3.12.10 | Verified via `python --version` (not stored in JSON) |
| **OS / Platform** | Windows 10 (win32) | Verified via `platform.platform()` (not stored in JSON) |
| **qiskit** | 2.6.0 | Stored in JSON `package_versions` ✓ |
| **qiskit-aer** | 0.17.2 | Stored in JSON `package_versions` ✓ |

**Unknown execution provenance (not in JSON, not verifiable):**
- Exact code commit used to generate the experiment
- OS version string
- CPU architecture
- BLAS library version (affects cross-platform reproducibility)

## Experiment Configuration

| Parameter | Value | Stored in JSON |
|---|---|---|
| QAOA depth (p) | 1 | ✓ |
| Shots per circuit | 512 | ✓ |
| Optimizer | COBYLA | ✓ |
| Max iterations | 50 | ✓ |
| Seeds | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | ✓ |
| Basis gates | id, rz, sx, x, cx, rzz, measure | ✓ |
| Noise model | Global depolarizing error (1-qubit: rz, sx; 2-qubit: rzz, cx) | Partially — noise_p_depol stored per condition |

## Noise Conditions

| Label | Depolarizing probability (p) | Sampler | Description |
|---|---|---|---|
| `noiseless_aer` | 0.0 | Aer (no noise) | Baseline control |
| `p=0.01` | 0.01 | Aer with depolarizing noise | Low noise |
| `p=0.05` | 0.05 | Aer with depolarizing noise | Medium noise |
| `p=0.10` | 0.10 | Aer with depolarizing noise | High noise |

## Instance Set

| Instance | Exact optimum | QUBO variables | Seeds |
|---|---|---|---|
| tiny_2job_2server | 5.0 | 11 | 10 |
| medium_2job_2server | 5.0 | 10 | 10 |
| tight_capacity_restricted | 2.0 | 7 | 10 |
| asymmetric_3job_3server | 6.0 | 17 | 10 |
| tight_capacity_3job_2server | 6.0 | 12 | 10 |
| cost_asymmetry_2job_2server | 3.0 | 10 | 10 |

Exact optima verified independently via `solve_exact()` from `qgrid.solvers.enumeration`.

## Critical Methodological Disclosure

**Parameters were optimized with a noiseless StatevectorSampler.** Noise was applied ONLY during final Aer sampling. This means:

1. QAOA parameters (γ, β) are identical across all noise conditions for a given instance + seed pair.
2. The only difference between conditions is the noise applied to the final circuit sampling step.
3. This does **NOT** study noisy optimization convergence (i.e., how noise during the optimization loop would affect parameter search).
4. Results reflect simulator behavior, **not** real quantum hardware.
5. Results are **not** evidence of quantum advantage.

This is reiterated in:
- `run_noise_experiment.py` lines 6–8
- JSON `note` field
- `analysis_report.md` Critical Disclosures section

## Known Limitations

1. **Small instance set:** Only 6 instances (7–17 qubits), 10 seeds per instance.
2. **Finite seeds:** Seeds 0–9 only; no seed randomization or seed range exploration.
3. **Simulated noise:** Depolarizing noise models only; real devices have additional error types (relaxation, dephasing, readout errors, crosstalk).
4. **No quantum advantage claim:** All results are from classical simulation (Qiskit Aer).
5. **No cross-platform verification:** Bit-for-bit reproducibility tested on the same machine only.
6. **Limited statistical power:** Bootstrap CIs are wide due to n=6 independent instances.

## Reproduction Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full noise experiment (240 trials)
python run_noise_experiment.py

# 3. Re-run the analysis
python noise_analysis.py

# 4. Run the statistical review
python statistical_review.py
```

## Artifact Provenance

| Artifact | Producing Script | Directory |
|---|---|---|
| Raw experiment JSON | `run_noise_experiment.py` | `results/noise_sensitivity/` |
| Aggregate results CSV | `noise_analysis.py` | `results/noise_sensitivity/analysis_20261023/` |
| Overall summary CSV | `noise_analysis.py` | `results/noise_sensitivity/analysis_20261023/` |
| Statistical comparisons | `noise_analysis.py` | `results/noise_sensitivity/analysis_20261023/` |
| Audit report MD | `noise_analysis.py` | `results/noise_sensitivity/analysis_20261023/` |
| Analysis report MD | `noise_analysis.py` | `results/noise_sensitivity/analysis_20261023/` |
| Plots 01–05 | `noise_analysis.py` | `results/noise_sensitivity/analysis_20261023/` |
| Heatmap 06 | `noise_analysis.py` | `results/noise_sensitivity/analysis_20261023/` |
| Statistical review MD | `statistical_review.py` | `results/noise_sensitivity/review_20261023/` |
| Paired differences CSV | `statistical_review.py` | `results/noise_sensitivity/review_20261023/` |
| Bootstrap summaries | `statistical_review.py` | `results/noise_sensitivity/review_20261023/` |

---
*This manifest was generated post-hoc from the preserved raw data and does not modify the experiment JSON.*
