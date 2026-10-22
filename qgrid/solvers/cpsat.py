"""
CP-SAT solver for job-to-server assignment using OR-Tools.

Encodes the same constraints and objective as the original problem model.
The CP-SAT model is defined directly with OR-Tools variables and
constraints, and the resulting solution is validated through the
existing AssignmentProblem.check_feasibility() interface.

Since CP-SAT requires integer coefficients, all floating-point values
(demands, capacities, costs) are scaled to integers using a multiplier.
"""

from __future__ import annotations

import time
from typing import Dict, List

from ortools.sat.python import cp_model

from qgrid.problem import (
    AssignmentProblem,
    AssignmentSolution,
    JobAssignmentInstance,
)
from .result import SolverResult

# Scaling factor to convert floating-point values to integers for CP-SAT.
SCALE = 1000


def _to_scaled_int(value: float) -> int:
    """Convert a float demand/capacity/cost to a scaled integer."""
    return int(round(value * SCALE))


def solve_cpsat(
    problem: AssignmentProblem,
    time_limit: float = 60.0,
    num_workers: int = 8,
) -> SolverResult:
    """
    Solve the job-to-server assignment problem using OR-Tools CP-SAT.

    Parameters
    ----------
    problem : AssignmentProblem
        The problem instance to solve.
    time_limit : float
        Maximum solving time in seconds.
    num_workers : int
        Number of parallel workers for the CP-SAT solver.

    Returns
    -------
    SolverResult
        Contains the best solution found (or None), objective value,
        status, and timing info.
    """
    start = time.perf_counter()
    instance: JobAssignmentInstance = problem.instance

    n_jobs = instance.n_jobs
    n_servers = instance.n_servers

    # Early pruning: total demand must not exceed total capacity
    if not instance.is_feasible_by_capacity():
        elapsed = time.perf_counter() - start
        return SolverResult(
            solution=None,
            objective=None,
            status="infeasible",
            runtime_seconds=elapsed,
            message="Total demand exceeds total capacity.",
            is_optimal=True,
        )

    # Scale demands, capacities, and costs to integers
    scaled_demands: Dict[int, int] = {
        j: _to_scaled_int(instance.job_demands[j]) for j in range(n_jobs)
    }
    scaled_capacities: Dict[int, int] = {
        s: _to_scaled_int(instance.server_capacities[s])
        for s in range(n_servers)
    }

    model = cp_model.CpModel()

    # Decision variables: x[j] = server index assigned to job j
    x_vars: List[cp_model.IntVar] = []
    for j in range(n_jobs):
        eligible = instance.eligible_servers(j)
        if not eligible:
            elapsed = time.perf_counter() - start
            return SolverResult(
                solution=None,
                objective=None,
                status="infeasible",
                runtime_seconds=elapsed,
                message=f"Job {j} has no eligible servers.",
                is_optimal=True,
            )
        var = model.new_int_var(eligible[0], eligible[-1], f"x_{j}")
        if len(eligible) > 1:
            model.add_allowed_assignments([var], [[s] for s in eligible])
        else:
            model.add(var == eligible[0])
        x_vars.append(var)

    # Capacity constraints: for each server, sum of demands
    # of jobs assigned to it must not exceed its capacity.
    for s in range(n_servers):
        eligible_jobs_on_server = [
            j for j in range(n_jobs) if s in instance.eligible_servers(j)
        ]
        if not eligible_jobs_on_server:
            continue

        # Boolean indicators: is job j assigned to server s?
        b_vars: List[cp_model.BoolVar] = []
        for j in eligible_jobs_on_server:
            b = model.new_bool_var(f"b_{j}_{s}")
            model.add(x_vars[j] == s).only_enforce_if(b)
            model.add(x_vars[j] != s).only_enforce_if(~b)
            b_vars.append(b)

        # Build capacity constraint with scaled integer demands
        total_demand_expr = sum(
            scaled_demands[j] * b_vars[idx]
            for idx, j in enumerate(eligible_jobs_on_server)
        )
        model.add(total_demand_expr <= scaled_capacities[s])

    # Objective: minimize total energy cost (scaled to integers)
    # We use boolean indicator variables for each (job, server) pair
    # to build a linear objective over integer coefficients.
    obj_terms: List[cp_model.IntVar] = []
    for j in range(n_jobs):
        eligible = instance.eligible_servers(j)
        for s in eligible:
            cost = instance.energy_costs.get((s, j), 0.0)
            scaled_cost = _to_scaled_int(cost)
            # Create a boolean indicator for (job j assigned to server s)
            b = model.new_bool_var(f"obj_{j}_{s}")
            model.add(x_vars[j] == s).only_enforce_if(b)
            model.add(x_vars[j] != s).only_enforce_if(~b)
            obj_terms.append(scaled_cost * b)

    model.minimize(sum(obj_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)

    elapsed = time.perf_counter() - start

    # Process results
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignment: Dict[int, int] = {
            j: int(solver.value(x_vars[j])) for j in range(n_jobs)
        }
        solution = AssignmentSolution(assignment=assignment)

        # Validate through the problem's own feasibility checker
        report = problem.check_feasibility(solution)
        if not report.is_feasible:
            return SolverResult(
                solution=None,
                objective=None,
                status="error",
                runtime_seconds=elapsed,
                message=f"CP-SAT returned infeasible solution: {report}",
                is_optimal=False,
            )

        # Compute objective from the original (unscaled) problem model
        obj_val = problem.evaluate(solution)
        is_optimal = status == cp_model.OPTIMAL
        status_str = "optimal" if is_optimal else "feasible"

        return SolverResult(
            solution=solution,
            objective=obj_val,
            status=status_str,
            runtime_seconds=elapsed,
            message=f"CP-SAT status: {solver.status_name(status)}",
            is_optimal=is_optimal,
            notes=f"CP-SAT scaled objective: {solver.objective_value / SCALE:.4f}",
        )

    elif status == cp_model.INFEASIBLE:
        return SolverResult(
            solution=None,
            objective=None,
            status="infeasible",
            runtime_seconds=elapsed,
            message="CP-SAT determined the problem is infeasible.",
            is_optimal=True,
        )

    elif status == cp_model.UNKNOWN:
        return SolverResult(
            solution=None,
            objective=None,
            status="unknown",
            runtime_seconds=elapsed,
            message="CP-SAT could not determine a solution within the time limit.",
            is_optimal=False,
        )

    else:
        return SolverResult(
            solution=None,
            objective=None,
            status="error",
            runtime_seconds=elapsed,
            message=f"CP-SAT solver returned unexpected status: {status}",
            is_optimal=False,
        )
