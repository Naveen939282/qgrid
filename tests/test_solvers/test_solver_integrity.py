"""
Integration tests comparing solvers.

Verifies that:
  1. Enumeration and CP-SAT agree on optimal objective for tiny instances.
  2. Greedy produces feasible solutions consistent with the problem model.
  3. All solvers report "infeasible" for genuinely infeasible instances.
  4. No solver fabricates feasibility or objective values.
"""

import pytest

from qgrid.problem import AssignmentProblem
from qgrid.solvers import solve_cpsat, solve_exact, solve_greedy


class TestSolverAgreement:
    """Cross-solver consistency tests."""

    def test_enumeration_and_cpsat_agree(
        self, tiny_problem: AssignmentProblem
    ) -> None:
        """CP-SAT and enumeration must agree on optimal cost."""
        exact = solve_exact(tiny_problem)
        cpsat = solve_cpsat(tiny_problem, time_limit=30)
        assert exact.objective == pytest.approx(cpsat.objective)

    def test_enumeration_and_cpsat_agree_eligible(
        self, eligible_problem: AssignmentProblem
    ) -> None:
        """CP-SAT and enumeration must agree on the 3x3 instance."""
        exact = solve_exact(eligible_problem)
        cpsat = solve_cpsat(eligible_problem, time_limit=30)
        assert exact.objective == pytest.approx(cpsat.objective)

    def test_greedy_feasible_on_tiny(
        self, tiny_problem: AssignmentProblem
    ) -> None:
        """Greedy should find a feasible solution, verified by problem checker."""
        greedy = solve_greedy(tiny_problem)
        assert greedy.status == "feasible"
        assert greedy.solution is not None
        report = tiny_problem.check_feasibility(greedy.solution)
        assert report.is_feasible

    def test_all_solvers_infeasible_agree(
        self, infeasible_capacity_problem: AssignmentProblem
    ) -> None:
        """All three solvers should report infeasible."""
        exact = solve_exact(infeasible_capacity_problem)
        cpsat = solve_cpsat(infeasible_capacity_problem, time_limit=10)
        greedy = solve_greedy(infeasible_capacity_problem)

        assert exact.status == "infeasible"
        assert cpsat.status == "infeasible"
        assert greedy.status == "infeasible"

    def test_exact_optimal_le_quality(
        self, eligible_problem: AssignmentProblem
    ) -> None:
        """
        Enumeration is the ground truth: its objective should be <=
        greedy's objective (enumeration is optimal, greedy is heuristic).
        """
        exact = solve_exact(eligible_problem)
        greedy = solve_greedy(eligible_problem)
        assert exact.objective is not None
        assert greedy.objective is not None
        assert exact.objective <= greedy.objective + 1e-10

    def test_cpsat_optimal_le_quality(
        self, eligible_problem: AssignmentProblem
    ) -> None:
        """
        CP-SAT (if optimal) should have objective <= greedy.
        """
        cpsat = solve_cpsat(eligible_problem, time_limit=30)
        greedy = solve_greedy(eligible_problem)
        if cpsat.status == "optimal":
            assert cpsat.objective is not None
            assert greedy.objective is not None
            assert cpsat.objective <= greedy.objective + 1e-10