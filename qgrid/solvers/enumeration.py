"""
Enumeration solver for job-to-server assignment.

Performs exhaustive search over all possible job-to-server assignments.
Limited to small instances where S^J (server_count ^ job_count) is
computationally tractable (typically S^J <= 10^6).

Returns the provably optimal feasible assignment.
"""

from __future__ import annotations

import itertools
import time
from typing import Optional

from qgrid.problem import (
    AssignmentProblem,
    AssignmentSolution,
    JobAssignmentInstance,
)
from .result import SolverResult


def solve_exact(
    problem: AssignmentProblem,
    time_limit: Optional[float] = None,
) -> SolverResult:
    """
    Exhaustively enumerate all possible assignments and return the
    feasible one with the minimum total energy cost.

    Parameters
    ----------
    problem : AssignmentProblem
        The problem instance to solve.
    time_limit : float or None
        Maximum wall-clock seconds. If exceeded, search stops and
        whatever best solution was found is returned with
        is_optimal=False.

    Returns
    -------
    SolverResult
        Contains the optimal solution (or None if infeasible),
        its objective value, status, and timing info.
    """
    start = time.perf_counter()
    instance: JobAssignmentInstance = problem.instance

    n_jobs = instance.n_jobs
    n_servers = instance.n_servers

    # Early check: if total demand exceeds total capacity, no feasible
    # solution exists.
    if not instance.is_feasible_by_capacity():
        elapsed = time.perf_counter() - start
        return SolverResult(
            solution=None,
            objective=None,
            status="infeasible",
            runtime_seconds=elapsed,
            message="Total demand exceeds total capacity; no feasible solution.",
            is_optimal=True,
        )

    best_solution: Optional[AssignmentSolution] = None
    best_cost: float = float("inf")
    found_feasible = False

    # Generate all possible assignments: each job is assigned to
    # one of n_servers possible servers.
    total_combos = n_servers ** n_jobs
    for combo_idx, combo in enumerate(itertools.product(range(n_servers), repeat=n_jobs)):
        assignment = {j: combo[j] for j in range(n_jobs)}
        solution = AssignmentSolution(assignment=assignment)

        report = problem.check_feasibility(solution)
        if not report.is_feasible:
            continue

        cost = problem.evaluate(solution)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
            found_feasible = True

        if time_limit is not None and (time.perf_counter() - start) > time_limit:
            elapsed = time.perf_counter() - start
            return SolverResult(
                solution=best_solution,
                objective=best_cost if found_feasible else None,
                status="feasible",
                runtime_seconds=elapsed,
                message=(
                    f"Enumeration interrupted by time limit after "
                    f"{combo_idx + 1}/{total_combos} combinations."
                ),
                is_optimal=False,
            )

    elapsed = time.perf_counter() - start

    if not found_feasible:
        return SolverResult(
            solution=None,
            objective=None,
            status="infeasible",
            runtime_seconds=elapsed,
            message=f"No feasible solution found among {total_combos} assignments.",
            is_optimal=True,
        )

    return SolverResult(
        solution=best_solution,
        objective=best_cost,
        status="optimal",
        runtime_seconds=elapsed,
        message=f"Optimal solution found after exhaustive search over {total_combos} assignments.",
        is_optimal=True,
        notes=f"Total combinations examined: {total_combos}",
    )