# Reproducibility Gaps Analysis

**Date:** 2026-10-23
**Dataset:** `results/noise_sensitivity/20261023_122432_noise_experiment.json`
**Review directory:** `results/noise_sensitivity/review_20261023/`

## Available Metadata (in raw JSON)

| Item | Value |
|---|---|
| Experiment type | `noise_sensitivity_full` |
| QAOA depth (p) | 1 |
| Optimizer | COBYLA |
| Max iterations | 50 |
| Shots (n_shots) | 512 |
| Qiskit version | 2.6.0 |
| Qiskit-Aer version | 0.17.2 |
| Seeds | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |
| Noise model | Depolarizing (p = 0.0, 0.01, 0.05, 0.10) |
| Basis gates | id, rz, sx, x, cx, rzz, measure |
| Noise targets | 1-qubit: rz, sx; 2-qubit: rzz, cx |
| Instance names | tiny_2job_2server, medium_2job_2server, tight_capacity_restricted, asymmetric_3job_3server, tight_capacity_3job_2server, cost_asymmetry_2job_2server |
| Exact optima | Verified against reference values (see analysis_report.md) |

## Missing Metadata

### 1. Source code revision hash
**Missing:** Git commit hash for both the experiment runner (`run_noise_experiment.py`) and the QAOA solver (`qgrid/solvers/qaoa.py`).

**Impact:** Cannot verify exact code version used. The experiment may rely on behavior specific to the commit at runtime.

### 2. Python version
**Missing:** The Python interpreter version used at experiment time is not recorded.

**Impact:** Different Python versions may yield different floating-point behavior in COBYLA or Aer.

### 3. Operating system and platform
**Missing:** OS name, version, and architecture (CPU instruction set) are not recorded.

**Impact:** BLAS implementation differences can affect floating-point results. Runtime metrics are platform-dependent.

### 4. StatevectorSampler random state
**Missing:** While trial seeds are recorded for COBYLA init (`np.random.RandomState(seed)`) and Aer sampling (`seed=seed`), the random state for the StatevectorSampler during optimization is not explicitly logged.

**Impact:** If StatevectorSampler uses a global or internally-seeded RNG, reproducibility across runs may not be guaranteed.

### 5. Aer simulator configuration
**Missing:** Aer-specific settings such as method (e.g., "statevector", "density_matrix"), parallelization backend, and memory limits are not recorded.

**Impact:** Different Aer methods produce different (though statistically equivalent) sampling behavior.

### 6. Effective maxiter
**Missing:** The `effective_maxiter` field exists in trial records but its distribution (how often COBYLA hit the iteration cap) is not summarized.

**Impact:** Cannot determine if convergence was achieved or prematurely stopped.

### 7. QUBO formulation details
**Missing:** The exact QUBO matrix configuration (penalty terms, constraint encoding) is defined in code but not captured in the JSON.

**Impact:** Cannot independently verify constraint enforcement or penalty strength.

### 8. Runtime boundaries
**Present but limited:** `runtime_seconds` in trial records includes "parameter init, COBYLA optimization, circuit construction, transpilation, and Aer sampling." It explicitly **excludes** exact optimum enumeration.

**Missing:** Fine-grained breakdown of component runtimes (optimization vs. sampling).

## Reproducibility Assessment

| Aspect | Status |
|---|---|
| Deterministic seeds | YES — `np.random.RandomState(seed)` and `seed=seed` |
| Package versions | YES — qiskit 2.6.0, qiskit-aer 0.17.2 |
| Noise configuration | YES — depolarizing p values and basis gates |
| QAOA settings | YES — depth, shots, optimizer, maxiter |
| Instance definitions | YES — via `qgrid.benchmark` functions |
| Reference optima | YES — `exact_objective` per trial |
| Seeds | YES — [0-9] |
| Git commit hash | NO |
| Python version | NO |
| OS/platform | NO |
| Aer method config | NO |
| StatevectorSampler seed | NO |

## Recommendation

For future experiments, capture the following at experiment start:
- `git rev-parse HEAD` (commit hash of the workspace)
- `sys.version` (Python version string)
- `platform.platform()` (OS and architecture)
- Aer backend method and configuration
- Explicit seed for StatevectorSampler

These are not blocking for re-analysis (the data and code are preserved), but they are necessary for independent reproduction.
