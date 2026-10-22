"""
Shared result type for all solvers.

Every solver returns a SolverResult dataclass that standardizes the
output format across enumeration, CP-SAT, and greedy heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qgrid.problem import AssignmentSolution


@dataclass
class SolverResult:
    """
    Standardized result returned by all solvers.

    Parameters
    ----------
    solution : AssignmentSolution or None
        The best feasible solution found, or None if no feasible
        solution exists.
    objective : float or None
        The objective (total energy cost) value of the solution.
        None if no feasible solution was found.
    status : str
        Human-readable solver status: "optimal", "feasible",
        "infeasible", "unknown", or "error".
    runtime_seconds : float
        Wall-clock time taken by the solver in seconds.
    message : str
        Additional solver-specific information.
    is_optimal : bool
        True if the solver proved this is the global optimum.
        False otherwise (e.g., greedy or timeout).
    notes : str or None
        Optional additional notes, e.g., number of nodes explored.
    """

    solution: Optional[AssignmentSolution]
    objective: Optional[float]
    status: str
    runtime_seconds: float
    message: str = ""
    is_optimal: bool = False
    notes: Optional[str] = None