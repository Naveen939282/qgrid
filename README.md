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
- **Streamlit**: Interactive dashboard

## Project Structure

```
qgrid/
├── qgrid/                    # Main package
│   ├── __init__.py
│   └── problem/              # Job-to-server assignment model
│       ├── __init__.py
│       ├── instance.py       # JobAssignmentInstance dataclass
│       └── problem.py        # Feasibility checker & objective evaluator
├── tests/                    # Test suite
│   └── test_problem/
│       ├── conftest.py       # Test fixtures
│       └── test_assignment.py # Problem tests
├── pyproject.toml
├── .gitignore
└── README.md
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

## Running Tests

### Windows PowerShell

From the project root (`D:\quantum`):

```powershell
# Run all tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_problem/test_assignment.py -v

# Run tests with coverage
python -m pytest tests/ --cov=qgrid --cov-report=html -v
```

### Running Tests in VS Code

1. Open the project folder in VS Code
2. Select the Python interpreter from `.venv` (`.venv\Scripts\Python`)
3. Open the terminal (Ctrl + `)
4. Run: `python -m pytest tests/ -v`

## Results and Experiments

### Benchmark Results

The repository includes pre-computed benchmark results in `results/benchmarks/`. These contain exact optima, QUBO formulation metadata, solver comparisons, and per-trial runtime data.

```bash
# Reproduce multi-seed benchmarks
python multi_seed_benchmark.py
```

### Noise-Sensitivity Experiment

The 240-trial noise-sensitivity experiment is documented in `results/noise_sensitivity/`.

```bash
# Run the full 240-trial experiment
python run_noise_experiment.py
```

**Critical disclosure:** QUBO parameters are optimized with a noiseless StatevectorSampler. Noise is applied only during final Aer sampling. This does not study noisy optimization convergence and is not evidence of quantum advantage.

### Statistical Review

```bash
# Reproduce the statistical review
python statistical_review.py
```

### Finding the Results

| Directory | Contents |
|---|---|
| `results/benchmarks/` | Benchmark JSON outputs (exact solver, CP-SAT, greedy, QAOA) |
| `results/benchmarks/multi_seed/` | Multi-seed benchmark results (6 instances) |
| `results/noise_sensitivity/` | 240-trial raw experiment, analysis, and review |
| `results/noise_sensitivity/analysis_20261023/` | Analysis reports, CSVs, and plots |
| `results/noise_sensitivity/review_20261023/` | Statistical review deliverables |
| `results/noise_sensitivity/PROVENANCE.md` | Experiment provenance manifest |
| `results/smoke_tests/` | Smoke test outputs |

### Environment

Install dependencies with the pinned versions:

```bash
pip install -r requirements.txt
```

## License

MIT License — see LICENSE file for details.