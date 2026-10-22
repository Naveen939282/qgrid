"""Classical solver package for Q-Grid.

Currently includes:
   - Enumeration: exhaustive search for tiny instances (ground truth).
   - CP-SAT: OR-Tools CP-SAT solver for the assignment problem.
   - Greedy: fast heuristic baseline respecting eligibility and capacity.
   - QAOA: minimal QAOA prototype using Qiskit and Qiskit Aer.
"""

from .enumeration import solve_exact
from .cpsat import solve_cpsat
from .greedy import solve_greedy
from .qaoa import (
    QAOAResult,
    solve_qaoa,
    qaoa_result_to_solver_result,
)
from .result import SolverResult

__all__ = [
    "solve_exact",
    "solve_cpsat",
    "solve_greedy",
    "solve_qaoa",
    "QAOAResult",
    "qaoa_result_to_solver_result",
    "SolverResult",
]
