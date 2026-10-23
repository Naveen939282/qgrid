# Q-Grid

**Noise-Aware Quantum-Classical Optimization for Energy-Efficient Data Center Job Assignment**

## Overview

Q-Grid is an open-source research project investigating how QAOA depth, constraint penalty weights, finite-shot sampling, and simulated noise affect feasible-solution rates and objective quality on small job-to-server assignment problems.

This project uses **local simulation only** — no paid APIs or quantum hardware.

## Technical Stack

- **Python 3.12** on Windows
- **Qiskit 2.6 + Qiskit Aer**: Quantum circuit construction and simulation
- **Google OR-Tools CP-SAT**: Classical optimization solver
- **NumPy, SciPy, pandas, matplotlib**: Numerical computing and visualization
- **pytest**: Testing framework

## Project Structure

```
qgrid/
├── qgrid/                     # Main package
│   ├── __init__.py
│   ├── benchmark.py           # Benchmark harness (6 instances, exact/CP-SAT/greedy/QAOA)
│   ├── formulation/
│   │   ├── __init__.py
│   │   ├── penalty.py         # Penalty weight computation
│   │   └── qubo.py            # QUBO encoding for job assignment
│   ├── problem/
│   │   ├── __init__.py
│   │   ├── instance.py        # JobAssignmentInstance dataclass
│   │   └── problem.py         # Feasibility checker & objective evaluator
│   └── solvers/
│       ├── __init__.py
│       ├── cpsat.py           # OR-Tools CP-SAT solver
│       ├── enumeration.py     # Exact brute-force solver
│       ├── greedy.py          # Greedy construction heuristic
│       ├── qaoa.py            # Depth-p QAOA with Aer sampling + noise
│       └── result.py          # Solver result dataclass
├── tests/
│   ├── conftest.py
│   ├── test_formulation/
│   │   └── test_qubo.py       # QUBO encoding tests
│   ├── test_problem/
│   │   └── test_assignment.py # Problem model tests
│   └── test_solvers/
│       ├── conftest.py
│       ├── test_benchmark.py
│       ├── test_cpsat.py
│       ├── test_enumeration.py
│       ├── test_greedy.py
│       ├── test_qaoa.py
│       ├── test_qaoa_experiments.py
│       └── test_solver_integrity.py
├── scripts/
│   └── audit_qubo.py          # Standalone QUBO energy convention auditor
├── docs/
│   └── milestones/
│       └── MILESTONE_6.4.md
├── results/
│   ├── benchmarks/             # Benchmark JSON results
│   ├── benchmarks/multi_seed/  # Multi-seed benchmark results (6 instances)
│   ├── qaoa_experiments/       # Exploratory QAOA experiments (see note)
│   ├── noise_sensitivity/      # Canonical 240-trial noise experiment
│   └── smoke_tests/            # Smoke test outputs
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── README.md
├── multi_seed_benchmark.py     # Multi-seed benchmark runner
├── noise_analysis.py           # Noise experiment analysis pipeline
├── run_benchmark.py            # Single-instance benchmark runner
├── run_noise_experiment.py     # Full 240-trial noise experiment
├── run_qaoa_experiment.py     # Reproducible QAOA demo on tiny instance
├── run_qaoa_finite_shot_experiment.py  # Finite-shot experiment script
├── run_smoke_test.py           # Smoke tests for noise models
└── statistical_review.py       # Independent statistical review
```

## Problem Definition

### Mathematical Formulation

**Decision variables:** $x_{ij} \in \{0, 1\}$ — job $j$ is assigned to server $i$

**Hard constraints:**
1. Every job assigned exactly once: $\sum_i x_{ij} = 1 \; \forall j$
2. Server capacity respected: $\sum_j w_j \cdot x_{ij} \leq C_i \; \forall i$
3. Eligibility: $x_{ij} = 0$ for all ineligible $(i, j)$ pairs

**Objective (minimize total energy cost):**
$$E = \sum_{i,j} c_{ij} \cdot x_{ij}$$

where $c_{ij}$ is the energy cost of assigning job $j$ to server $i$.

## Setup

### Create and activate a virtual environment (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# If blocked by execution policy, allow for this session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Install dependencies

```powershell
# Pinned direct dependencies (recommended for exact reproducibility)
pip install -r requirements.txt

# Or install from pyproject.toml with dev tools (includes pytest-cov for coverage)
pip install -e ".[dev]"
```

## Running Tests

The standard test command is:

```powershell
python -m pytest tests/ -q
```

Expected: 184 passed.

### Coverage (optional)

The project does not include pytest-cov in requirements.txt. To run tests with coverage:

```powershell
# Install the dev extra first (includes pytest-cov)
pip install -e ".[dev]"

# Run tests with coverage
python -m pytest tests/ -v --cov=qgrid --cov-report=term-missing
```

### Running Tests in VS Code

1. Open the project folder in VS Code
2. Select the Python interpreter from .venv (.venv\Scripts\Python)
3. Open the terminal (Ctrl + `)
4. Run: python -m pytest tests/ -v

## Results and Experiments

### Benchmark Results

The repository includes pre-computed benchmark results in results/benchmarks/. These contain exact optima, QUBO formulation metadata, solver comparisons, and per-trial runtime data.

```bash
# Reproduce multi-seed benchmarks
python multi_seed_benchmark.py
```

### Noise-Sensitivity Experiment

The 240-trial noise-sensitivity experiment is documented in results/noise_sensitivity/.

```bash
# Run the full 240-trial experiment
python run_noise_experiment.py
```

**Critical disclosure:** QUBO parameters are optimized with a noiseless StatevectorSampler. Noise is applied only during final Aer sampling. This does not study noisy optimization convergence and is not evidence of quantum advantage.

Across 240 trials (6 instances x 4 noise conditions x 10 seeds), all trials found feasible solutions, and 236/240 (98.3%) matched the exact classical optimum. Noiseless control achieved lower match rates on harder 17-qubit instances (80-95%) vs. 100% for easy instances. Statistical review notes wide confidence intervals due to only 6 independent instances; CIs include zero for noise degradation effects.

### Statistical Review

```bash
# Reproduce the statistical review
python statistical_review.py
```

### Exploratory QAOA Experiments

The results/qaoa_experiments/ directory contains earlier exploratory QAOA experiments (finite-shot, noise comparison, noise distribution checks) from run_qaoa_finite_shot_experiment.py. These are **not** the canonical noise-sensitivity study. The canonical experiment is in results/noise_sensitivity/.

### Finding the Results

| Directory | Contents |
|---|---|
| results/benchmarks/ | Benchmark JSON outputs (exact solver, CP-SAT, greedy, QAOA) |
| results/benchmarks/multi_seed/ | Multi-seed benchmark results (6 instances) |
| results/qaoa_experiments/ | Exploratory QAOA experiments (finite-shot, noise checks) |
| results/noise_sensitivity/ | 240-trial raw experiment JSON + analysis & review |
| results/noise_sensitivity/analysis_20261023/ | Analysis reports, CSVs, and plots |
| results/noise_sensitivity/review_20261023/ | Statistical review deliverables |
| results/noise_sensitivity/PROVENANCE.md | Experiment provenance manifest |
| results/smoke_tests/ | Smoke test outputs |

### Environment

Pinned direct dependencies are listed in requirements.txt. For the active development environment (Python 3.12.10, Qiskit 2.6.0, Qiskit-Aer 0.17.2, see PROVENANCE.md for full details):

```powershell
pip install -r requirements.txt
```

### Reproducibility

| Script | Purpose | Output Directory |
|---|---|---|
| run_noise_experiment.py | 240-trial noise-sensitivity experiment (6 instances x 4 conditions x 10 seeds) | results/noise_sensitivity/ |
| noise_analysis.py | Analysis pipeline: aggregate stats, plots, reports | results/noise_sensitivity/analysis_20261023/ |
| statistical_review.py | Independent statistical review: paired tests, bootstrap CIs | results/noise_sensitivity/review_20261023/ |
| multi_seed_benchmark.py | Multi-seed QAOA benchmark across 6 instances | results/benchmarks/multi_seed/ |
| run_benchmark.py | Single-instance benchmark comparison | results/benchmarks/ |
| run_smoke_test.py | Smoke tests for noise model coverage | results/smoke_tests/ |
| scripts/audit_qubo.py | Standalone QUBO energy convention auditor | Console output only |

## License

MIT License — see LICENSE file for details.