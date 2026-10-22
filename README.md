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

## License

MIT License — see LICENSE file for details.