"""
Greedy baseline solver for job-to-server assignment.

Assigns each job (in order) to the eligible server with the lowest energy
cost that still has sufficient remaining capacity.

If no feasible assignment can be found, reports failure rather than
returning an infeasible solution.
"""

from __future__ import annotations

import time
from typing import Dict, List

from qgrid.problem import (
    AssignmentProblem,
    AssignmentSolution,
    JobAssignmentInstance,
)
from .result import SolverResult


def solve_greedy(
    problem: AssignmentProblem,
    job_order: str = "index",
) -> SolverResult:
    """
    Solve the assignment problem using a greedy heuristic.

    Strategy
    --------
    For each job in the chosen order, assign it to the eligible server
    with the lowest energy cost that has sufficient remaining capacity.
    If no eligible server has capacity, the problem is infeasible for
    this job ordering.

    Parameters
    ----------
    problem : AssignmentProblem
        The problem instance to solve.
    job_order : str, default "index"
        Order in which to process jobs:
        - "index": jobs 0, 1, 2, ...
        - "demand_asc": smallest demand first
        - "demand_desc": largest demand first

    Returns
    -------
    SolverResult
        Contains the feasible solution found (or None if infeasible),
        its objective value, status, and timing info.
    """
    start = time.perf_counter()
    instance: JobAssignmentInstance = problem.instance

    n_jobs = instance.n_jobs
    n_servers = instance.n_servers

    # Early pruning
    if not instance.is_feasible_by_capacity():
        elapsed = time.perf_counter() - start
        return SolverResult(
            solution=None,
            objective=None,
            status="infeasible",
            runtime_seconds=elapsed,
            message="Total demand exceeds total capacity.",
            is_optimal=False,
        )

    # Determine job processing order
    if job_order == "index":
        order: List[int] = list(range(n_jobs))
    elif job_order == "demand_asc":
        order = sorted(range(n_jobs), key=lambda j: instance.job_demands[j])
    elif job_order == "demand_desc":
        order = sorted(
            range(n_jobs), key=lambda j: instance.job_demands[j], reverse=True
        )
    else:
        elapsed = time.perf_counter() - start
        return SolverResult(
            solution=None,
            objective=None,
            status="error",
            runtime_seconds=elapsed,
            message=f"Unknown job_order: {job_order}",
            is_optimal=False,
        )

    # Track remaining capacity per server
    remaining_cap: Dict[int, float] = {
        s: instance.server_capacities[s] for s in range(n_servers)
    }

    # Assignment result
    assignment: Dict[int, int] = {}

    for j in order:
        eligible = instance.eligible_servers(j)
        if not eligible:
            elapsed = time.perf_counter() - start
            return SolverResult(
                solution=None,
                objective=None,
                status="infeasible",
                runtime_seconds=elapsed,
                message=f"Job {j} has no eligible servers.",
                is_optimal=False,
            )

        # Sort eligible servers by energy cost, then by remaining capacity
        eligible_sorted = sorted(
            eligible,
            key=lambda s: (
                instance.energy_costs.get((s, j), float("inf")),
                -remaining_cap[s],  # prefer servers with more remaining capacity
            ),
        )

        assigned = False
        demand = instance.job_demands[j]
        for s in eligible_sorted:
            if demand <= remaining_cap[s] + 1e-9:
                assignment[j] = s
                remaining_cap[s] -= demand
                assigned = True
                break

        if not assigned:
            elapsed = time.perf_counter() - start
            return SolverResult(
                solution=None,
                objective=None,
                status="infeasible",
                runtime_seconds=elapsed,
                message=(
                    f"No eligible server with sufficient capacity for "
                    f"job {j} (demand={demand})."
                ),
                is_optimal=False,
            )

    # Construct solution and validate
    solution = AssignmentSolution(assignment=assignment)
    report = problem.check_feasibility(solution)
    elapsed = time.perf_counter() - start

    if not report.is_feasible:
        return SolverResult(
            solution=None,
            objective=None,
            status="error",
            runtime_seconds=elapsed,
            message=f"Greedy produced infeasible solution: {report}",
            is_optimal=False,
        )

    cost = problem.evaluate(solution)
    return SolverResult(
        solution=solution,
        objective=cost,
        status="feasible",
        runtime_seconds=elapsed,
        message=f"Greedy solution found using job order: {job_order}.",
        is_optimal=False,
        notes=f"Assignment: {assignment}",
    )