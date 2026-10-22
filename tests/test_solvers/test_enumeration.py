"""
Tests for the enumeration (exact) solver.

Tests cover:
  - Optimal solution found on tiny instance
  - Objective value correctness
  - Infeasible instances (capacity violation)
  - Edge cases (eligibility, single server)
  - Timing behavior (time limit)
"""

import pytest

from qgrid.problem import AssignmentProblem, JobAssignmentInstance
from qgrid.solvers import SolverResult
from qgrid.solvers.enumeration import solve_exact


class TestEnumerationSolver:
    """Tests for the exhaustive enumeration solver."""

    def test_finds_optimal(self, tiny_problem: AssignmentProblem) -> None:
        """Enumeration should find the optimal solution (cost=5.0)."""
        result = solve_exact(tiny_problem)
        assert result.status == "optimal"
        assert result.is_optimal is True
        assert result.solution is not None
        assert result.objective == pytest.approx(5.0)
        # Optimal assignment: {0: 1, 1: 0}
        assert result.solution.assignment == {0: 1, 1: 0}

    def test_solution_is_feasible(self, tiny_problem: AssignmentProblem) -> None:
        """The returned solution must pass the feasibility checker."""
        result = solve_exact(tiny_problem)
        assert result.solution is not None
        report = tiny_problem.check_feasibility(result.solution)
        assert report.is_feasible

    def test_infeasible_instance(
        self, infeasible_capacity_problem: AssignmentProblem
    ) -> None:
        """Enumeration should report infeasible when no solution exists."""
        result = solve_exact(infeasible_capacity_problem)
        assert result.status == "infeasible"
        assert result.solution is None
        assert result.objective is None

    def test_tight_capacity_feasible(
        self, tight_capacity_problem: AssignmentProblem
    ) -> None:
        """Tight capacity instance should be feasible (sum of demands = capacity)."""
        result = solve_exact(tight_capacity_problem)
        assert result.status == "optimal"
        assert result.solution is not None
        report = tight_capacity_problem.check_feasibility(result.solution)
        assert report.is_feasible

    def test_eligibility_constraint(
        self, eligibility_problem: AssignmentProblem
    ) -> None:
        """Enumeration should respect eligibility constraints."""
        result = solve_exact(eligibility_problem)
        assert result.solution is not None
        report = eligibility_problem.check_feasibility(result.solution)
        assert report.is_feasible
        assert len(report.ineligible_assignments) == 0

    def test_eligible_instance_optimal(
        self, eligible_problem: AssignmentProblem
    ) -> None:
        """On the 3x3 instance, enumeration finds the global optimum."""
        result = solve_exact(eligible_problem)
        assert result.status == "optimal"
        assert result.solution is not None
        report = eligible_problem.check_feasibility(result.solution)
        assert report.is_feasible
        assert result.is_optimal is True

        # Verify by brute force
        instance = eligible_problem.instance
        best_cost = float("inf")
        from itertools import product
        for combo in product(range(instance.n_servers), repeat=instance.n_jobs):
            sol = {j: combo[j] for j in range(instance.n_jobs)}
            from qgrid.problem import AssignmentSolution
            solution_obj = AssignmentSolution(assignment=sol)
            if eligible_problem.check_feasibility(solution_obj).is_feasible:
                cost = eligible_problem.evaluate(solution_obj)
                best_cost = min(best_cost, cost)
        assert result.objective == pytest.approx(best_cost)

    def test_time_limit_returns_feasible(
        self,
        tiny_problem: AssignmentProblem,
    ) -> None:
        """With a very short time limit, should still return a result."""
        result = solve_exact(tiny_problem, time_limit=0.001)
        assert result.status in ("optimal", "feasible")
        assert result.solution is not None
        assert result.is_optimal in (True, False)

    def test_runtime_positive(self, tiny_problem: AssignmentProblem) -> None:
        """Runtime should be non-negative."""
        result = solve_exact(tiny_problem)
        assert result.runtime_seconds >= 0.0