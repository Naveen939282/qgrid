"""
Tests for the job-to-server assignment problem module.

Covers:
  - Feasible assignments
  - Capacity violations
  - Missing assignments
  - Duplicate assignments
  - Ineligible assignments
"""

import pytest

from qgrid.problem import (
    AssignmentProblem,
    AssignmentSolution,
    JobAssignmentInstance,
)


# ---------------------------------------------------------------------------
# Instance validation tests
# ---------------------------------------------------------------------------

class TestInstanceValidation:
    """Tests for JobAssignmentInstance input validation."""

    def test_valid_instance_constructs(self) -> None:
        """A correctly formed instance should construct without error."""
        instance = JobAssignmentInstance(
            n_jobs=2,
            n_servers=2,
            job_demands=[3.0, 2.0],
            server_capacities=[5.0, 10.0],
            eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
            energy_costs={(0, 0): 5.0, (1, 0): 2.0, (0, 1): 3.0, (1, 1): 4.0},
        )
        assert instance.n_jobs == 2
        assert instance.n_servers == 2

    def test_negative_demand_raises(self) -> None:
        """Negative job demands should raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            JobAssignmentInstance(
                n_jobs=1,
                n_servers=1,
                job_demands=[-1.0],
                server_capacities=[10.0],
                eligible_jobs=[frozenset({0})],
                energy_costs={(0, 0): 1.0},
            )

    def test_negative_capacity_raises(self) -> None:
        """Negative server capacity should raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            JobAssignmentInstance(
                n_jobs=1,
                n_servers=1,
                job_demands=[1.0],
                server_capacities=[-5.0],
                eligible_jobs=[frozenset({0})],
                energy_costs={(0, 0): 1.0},
            )

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched list lengths should raise ValueError."""
        with pytest.raises(ValueError, match="n_jobs"):
            JobAssignmentInstance(
                n_jobs=2,
                n_servers=1,
                job_demands=[1.0],
                server_capacities=[10.0],
                eligible_jobs=[frozenset({0})],
                energy_costs={(0, 0): 1.0},
            )

    def test_eligibility_out_of_range_raises(self) -> None:
        """Job index in eligibility set exceeding n_jobs should raise."""
        with pytest.raises(ValueError, match="invalid job index"):
            JobAssignmentInstance(
                n_jobs=1,
                n_servers=1,
                job_demands=[1.0],
                server_capacities=[10.0],
                eligible_jobs=[frozenset({0, 5})],
                energy_costs={(0, 0): 1.0},
            )

    def test_non_finite_energy_cost_raises(self) -> None:
        """Non-finite energy costs should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            JobAssignmentInstance(
                n_jobs=1,
                n_servers=1,
                job_demands=[1.0],
                server_capacities=[10.0],
                eligible_jobs=[frozenset({0})],
                energy_costs={(0, 0): float("inf")},
            )


# ---------------------------------------------------------------------------
# Feasibility tests
# ---------------------------------------------------------------------------

class TestFeasibility:
    """Tests for the feasibility checker."""

    def test_feasible_assignment(
        self,
        tiny_problem: AssignmentProblem,
    ) -> None:
        """Optimal assignment {0:1, 1:0} should be feasible."""
        solution = AssignmentSolution(assignment={0: 1, 1: 0})
        report = tiny_problem.check_feasibility(solution)
        assert report.is_feasible is True
        assert len(report.missing_assignments) == 0
        assert len(report.capacity_violations) == 0
        assert len(report.ineligible_assignments) == 0

    def test_all_feasible_assignments(
        self,
        tiny_problem: AssignmentProblem,
    ) -> None:
        """All four possible assignments should be feasible for the tiny instance."""
        for s0 in range(2):
            for s1 in range(2):
                sol = AssignmentSolution(assignment={0: s0, 1: s1})
                report = tiny_problem.check_feasibility(sol)
                assert report.is_feasible, (
                    f"Assignment {sol.assignment} should be feasible"
                )

    def test_capacity_violation(
        self,
        tiny_instance: JobAssignmentInstance,
    ) -> None:
        """Two jobs with combined demand exceeding server capacity."""
        instance = JobAssignmentInstance(
            n_jobs=2,
            n_servers=1,
            job_demands=[3.0, 4.0],
            server_capacities=[5.0],
            eligible_jobs=[frozenset({0, 1})],
            energy_costs={(0, 0): 1.0, (0, 1): 2.0},
        )
        problem = AssignmentProblem(instance=instance)
        solution = AssignmentSolution(assignment={0: 0, 1: 0})
        report = problem.check_feasibility(solution)
        assert report.is_feasible is False
        assert len(report.capacity_violations) == 1
        # Server 0, used 7.0, capacity 5.0
        assert report.capacity_violations[0] == (0, 7.0, 5.0)

    def test_capacity_edge_case_equal(
        self,
        tiny_instance: JobAssignmentInstance,
    ) -> None:
        """When demand exactly equals capacity, it should be feasible."""
        instance = JobAssignmentInstance(
            n_jobs=1,
            n_servers=1,
            job_demands=[5.0],
            server_capacities=[5.0],
            eligible_jobs=[frozenset({0})],
            energy_costs={(0, 0): 1.0},
        )
        problem = AssignmentProblem(instance=instance)
        solution = AssignmentSolution(assignment={0: 0})
        report = problem.check_feasibility(solution)
        assert report.is_feasible is True

    def test_missing_assignment(
        self,
        tiny_instance: JobAssignmentInstance,
    ) -> None:
        """A solution missing a job should be flagged as infeasible."""
        problem = AssignmentProblem(instance=tiny_instance)
        solution = AssignmentSolution(assignment={0: 0})
        report = problem.check_feasibility(solution)
        assert report.is_feasible is False
        assert report.missing_assignments == [1]

    def test_ineligible_assignment(
        self,
        eligibility_instance: JobAssignmentInstance,
    ) -> None:
        """Job 1 is NOT eligible for server 0; assigning it there is infeasible."""
        problem = AssignmentProblem(instance=eligibility_instance)
        solution = AssignmentSolution(assignment={0: 0, 1: 0})
        report = problem.check_feasibility(solution)
        assert report.is_feasible is False
        assert report.ineligible_assignments == [(1, 0)]

    def test_infeasible_capacity_instance(
        self,
        infeasible_capacity_instance: JobAssignmentInstance,
    ) -> None:
        """Job with demand 10 on a server with capacity 5 should violate."""
        problem = AssignmentProblem(instance=infeasible_capacity_instance)
        solution = AssignmentSolution(assignment={0: 0})
        report = problem.check_feasibility(solution)
        assert report.is_feasible is False
        assert len(report.capacity_violations) == 1

    def test_assignment_empty_raises(self) -> None:
        """An empty assignment dictionary should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            AssignmentSolution(assignment={})


# ---------------------------------------------------------------------------
# Objective evaluation tests
# ---------------------------------------------------------------------------

class TestObjectiveEvaluation:
    """Tests for the objective (energy cost) function."""

    def test_optimal_solution_cost(
        self,
        tiny_problem: AssignmentProblem,
    ) -> None:
        """
        Optimal assignment:
          job 0 → server 1 (cost 2.0)
          job 1 → server 0 (cost 3.0)
        Total = 5.0
        """
        solution = AssignmentSolution(assignment={0: 1, 1: 0})
        cost = tiny_problem.evaluate(solution)
        assert cost == pytest.approx(5.0)

    def test_all_assignment_costs(
        self,
        tiny_problem: AssignmentProblem,
    ) -> None:
        """Verify energy costs for all four feasible assignments."""
        expected = {
            (0, 0): 8.0,
            (0, 1): 9.0,
            (1, 0): 5.0,   # optimal
            (1, 1): 6.0,
        }
        for (s0, s1), expected_cost in expected.items():
            sol = AssignmentSolution(assignment={0: s0, 1: s1})
            cost = tiny_problem.evaluate(sol)
            assert cost == pytest.approx(expected_cost), (
                f"Assignment {{0:{s0}, 1:{s1}}} "
                f"expected {expected_cost}, got {cost}"
            )

    def test_infeasible_raises_on_evaluate(
        self,
        tiny_problem: AssignmentProblem,
    ) -> None:
        """Calling evaluate on an infeasible solution should raise ValueError."""
        solution = AssignmentSolution(assignment={0: 0})
        with pytest.raises(ValueError, match="infeasible"):
            tiny_problem.evaluate(solution)


# ---------------------------------------------------------------------------
# Solution utility tests
# ---------------------------------------------------------------------------

class TestSolutionUtilities:
    """Tests for AssignmentSolution helper methods."""

    def test_to_list(self) -> None:
        """to_list should produce a server-per-job list."""
        sol = AssignmentSolution(assignment={0: 1, 1: 0, 2: 1})
        result = sol.to_list(n_jobs=3)
        assert result == [1, 0, 1]

    def test_non_integer_keys_raise(self) -> None:
        """Non-integer keys should raise ValueError."""
        with pytest.raises(ValueError, match="must be integers"):
            AssignmentSolution(assignment={0.5: 0})  # type: ignore

    def test_non_integer_values_raise(self) -> None:
        """Non-integer values should raise ValueError."""
        with pytest.raises(ValueError, match="must be integers"):
            AssignmentSolution(assignment={0: 0.5})  # type: ignore