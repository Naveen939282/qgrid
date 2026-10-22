"""
Tests for the greedy heuristic solver.

Tests cover:
  - Finds feasible solution on standard instances
  - Correctly reports infeasible when capacity cannot be met
  - Respects eligibility constraints
  - Does not return infeasible solutions
  - Multiple job ordering strategies
"""

import pytest

from qgrid.problem import AssignmentProblem
from qgrid.solvers import solve_greedy


class TestGreedySolver:
    """Tests for the greedy heuristic solver."""

    def test_finds_feasible(self, tiny_problem: AssignmentProblem) -> None:
        """Greedy should find a feasible solution on the tiny instance."""
        result = solve_greedy(tiny_problem)
        assert result.status == "feasible"
        assert result.solution is not None
        report = tiny_problem.check_feasibility(result.solution)
        assert report.is_feasible

    def test_objective_not_worse_than_worst(
        self, tiny_problem: AssignmentProblem
    ) -> None:
        """
        Greedy cost should not exceed the worst feasible solution.
        Worst on tiny instance is {0:0, 1:1} = 9.0.
        """
        result = solve_greedy(tiny_problem)
        assert result.objective is not None
        assert result.objective <= 9.0

    def test_infeasible_capacity(
        self, infeasible_capacity_problem: AssignmentProblem
    ) -> None:
        """Greedy should report infeasible when demand > capacity."""
        result = solve_greedy(infeasible_capacity_problem)
        assert result.status == "infeasible"
        assert result.solution is None

    def test_eligibility_respected(
        self, eligibility_problem: AssignmentProblem
    ) -> None:
        """Greedy must not assign jobs to ineligible servers."""
        result = solve_greedy(eligibility_problem)
        assert result.solution is not None
        report = eligibility_problem.check_feasibility(result.solution)
        assert report.is_feasible
        assert len(report.ineligible_assignments) == 0

    def test_tight_capacity_feasible(
        self, tight_capacity_problem: AssignmentProblem
    ) -> None:
        """Greedy should handle tight capacity (sum = capacity)."""
        result = solve_greedy(tight_capacity_problem)
        # With single server, greedy has no choice
        assert result.status == "feasible"
        report = tight_capacity_problem.check_feasibility(result.solution)
        assert report.is_feasible

    def test_demand_desc_order(self, tiny_problem: AssignmentProblem) -> None:
        """Greedy with demand_desc ordering should find a feasible solution."""
        result = solve_greedy(tiny_problem, job_order="demand_desc")
        assert result.status == "feasible"
        report = tiny_problem.check_feasibility(result.solution)
        assert report.is_feasible

    def test_demand_asc_order(self, tiny_problem: AssignmentProblem) -> None:
        """Greedy with demand_asc ordering should find a feasible solution."""
        result = solve_greedy(tiny_problem, job_order="demand_asc")
        assert result.status == "feasible"
        report = tiny_problem.check_feasibility(result.solution)
        assert report.is_feasible

    def test_no_fabricated_results(
        self, tiny_problem: AssignmentProblem
    ) -> None:
        """
        Verify greedy solution cost matches what evaluate() computes.
        No fabricated or hidden objective values.
        """
        result = solve_greedy(tiny_problem)
        assert result.solution is not None
        cost = tiny_problem.evaluate(result.solution)
        assert result.objective == pytest.approx(cost)

    def test_runtime_positive(self, tiny_problem: AssignmentProblem) -> None:
        """Runtime should be non-negative."""
        result = solve_greedy(tiny_problem)
        assert result.runtime_seconds >= 0.0