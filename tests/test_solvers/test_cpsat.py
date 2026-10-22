"""
Tests for the CP-SAT solver.

Tests cover:
  - Optimal solution matches enumeration on tiny instances
  - Correctness of objective value
  - Infeasible instances
  - Eligibility constraints
  - Solver status correctness
"""

import pytest

from qgrid.problem import AssignmentProblem, JobAssignmentInstance
from qgrid.solvers import solve_cpsat, solve_exact


class TestCPSatSolver:
    """Tests for the OR-Tools CP-SAT solver."""

    def test_finds_optimal(self, tiny_problem: AssignmentProblem) -> None:
        """CP-SAT should find the optimal solution (cost=5.0)."""
        result = solve_cpsat(tiny_problem, time_limit=30)
        assert result.status == "optimal"
        assert result.is_optimal is True
        assert result.solution is not None
        assert result.objective == pytest.approx(5.0)

    def test_matches_enumeration(
        self, tiny_problem: AssignmentProblem
    ) -> None:
        """CP-SAT and enumeration should agree on the optimal cost."""
        exact_result = solve_exact(tiny_problem)
        cpsat_result = solve_cpsat(tiny_problem, time_limit=30)
        assert cpsat_result.objective == pytest.approx(exact_result.objective)

    def test_solution_is_feasible(self, tiny_problem: AssignmentProblem) -> None:
        """The returned solution must pass the feasibility checker."""
        result = solve_cpsat(tiny_problem, time_limit=30)
        assert result.solution is not None
        report = tiny_problem.check_feasibility(result.solution)
        assert report.is_feasible

    def test_infeasible_instance(
        self, infeasible_capacity_problem: AssignmentProblem
    ) -> None:
        """CP-SAT should report infeasible when no solution exists."""
        result = solve_cpsat(infeasible_capacity_problem, time_limit=10)
        assert result.status == "infeasible"
        assert result.solution is None
        assert result.objective is None

    def test_eligibility_constraint(
        self, eligibility_problem: AssignmentProblem
    ) -> None:
        """CP-SAT should respect eligibility constraints."""
        result = solve_cpsat(eligibility_problem, time_limit=30)
        assert result.solution is not None
        report = eligibility_problem.check_feasibility(result.solution)
        assert report.is_feasible
        assert len(report.ineligible_assignments) == 0

    def test_eligible_instance_matches_enumeration(
        self, eligible_problem: AssignmentProblem
    ) -> None:
        """CP-SAT and enumeration should match on the 3x3 instance."""
        exact_result = solve_exact(eligible_problem)
        cpsat_result = solve_cpsat(eligible_problem, time_limit=30)
        assert cpsat_result.objective == pytest.approx(exact_result.objective)

    def test_proven_optimal_flag(
        self, tiny_problem: AssignmentProblem
    ) -> None:
        """On a small instance, CP-SAT should return is_optimal=True."""
        result = solve_cpsat(tiny_problem, time_limit=10)
        if result.status == "optimal":
            assert result.is_optimal is True

    def test_runtime_positive(self, tiny_problem: AssignmentProblem) -> None:
        """Runtime should be non-negative."""
        result = solve_cpsat(tiny_problem, time_limit=10)
        assert result.runtime_seconds >= 0.0