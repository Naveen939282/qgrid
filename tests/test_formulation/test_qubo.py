"""
Tests for the QUBO encoding of the job-to-server assignment problem.

Tests cover:
  - Variable ordering and indexing
  - QUBO matrix dimensions and symmetry
  - Energy calculation matches penalty decomposition
  - Assignment constraint penalties
  - Capacity slack encoding correctness
  - Ground state matches constrained optimum
  - Infeasible instance handling
  - Penalty weight sufficiency checking
"""

import itertools

import numpy as np
import pytest

from qgrid.formulation import (
    encode_qubo,
    compute_qubo_energy,
    decode_solution,
    compute_penalty_terms,
    compute_penalty_weights,
    check_penalty_sufficiency,
)
from qgrid.problem import (
    AssignmentProblem,
    AssignmentSolution,
    JobAssignmentInstance,
)
from qgrid.solvers import solve_exact

class TestQUBOEncoding:
    """Tests for QUBO matrix construction and variable indexing."""

    def test_qubo_dimensions(self, tiny_instance: JobAssignmentInstance) -> None:
        """Q matrix should be square with size = n_assignment + n_slack vars."""
        qubo = encode_qubo(tiny_instance)
        n_vars = qubo.n_assignment_vars + qubo.n_slack_vars
        assert qubo.Q.shape == (n_vars, n_vars)
        assert qubo.n_vars == n_vars

    def test_assignment_var_count(self, tiny_instance: JobAssignmentInstance) -> None:
        """All eligible (server, job) pairs should have assignment variables."""
        qubo = encode_qubo(tiny_instance)
        assert qubo.n_assignment_vars == 4

    def test_variable_metadata_consistency(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Var metadata should correctly identify assignment vs slack vars."""
        qubo = encode_qubo(tiny_instance)
        for i, meta in enumerate(qubo.var_metadata):
            assert meta["type"] in ("assignment", "slack")

    def test_q_matrix_upper_triangular(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Q matrix should be upper-triangular (zeros below diagonal)."""
        qubo = encode_qubo(tiny_instance)
        Q = qubo.Q
        n = Q.shape[0]
        for i in range(n):
            for j in range(i):
                assert Q[i, j] == 0.0

    def test_slack_var_count(self, tiny_instance: JobAssignmentInstance) -> None:
        """Slack variables should exist for capacity constraints."""
        qubo = encode_qubo(tiny_instance)
        assert qubo.n_slack_vars > 0

    def test_slack_weights_are_powers_of_two(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Slack variable weights should be powers of 2 (binary encoding)."""
        qubo = encode_qubo(tiny_instance)
        for meta in qubo.var_metadata:
            if meta["type"] == "slack":
                weight = meta.get("weight", 1)
                assert (weight & (weight - 1)) == 0


class TestQUBOEnergy:
    """Tests for QUBO energy computation and penalty decomposition."""

    def test_energy_matches_penalty_decomposition(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """QUBO energy should match sum of decomposed penalty terms."""
        qubo = encode_qubo(tiny_instance)
        n_vars = qubo.n_vars

        for combo in itertools.product([0, 1], repeat=n_vars):
            x = np.array(combo, dtype=float)
            energy = compute_qubo_energy(x, qubo)
            terms = compute_penalty_terms(x, qubo)
            reconstructed = (
                terms["objective"]
                + terms["assignment_penalty"]
                + terms["capacity_penalty"]
            )
            assert energy == pytest.approx(reconstructed, rel=1e-3), (
                f"Mismatch at {combo}: energy={energy}, recon={reconstructed}"
            )

    def test_feasible_solution_zero_penalties(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Feasible solutions should have zero penalty terms."""
        qubo = encode_qubo(tiny_instance)
        n_vars = qubo.n_vars
        problem = AssignmentProblem(instance=tiny_instance)

        for combo in itertools.product([0, 1], repeat=n_vars):
            x = np.array(combo, dtype=float)
            terms = compute_penalty_terms(x, qubo)
            if terms["assignment_penalty"] == 0 and terms["capacity_penalty"] == 0:
                assignment, _ = decode_solution(x, qubo)
                if len(assignment) == tiny_instance.n_jobs:
                    sol = AssignmentSolution(assignment=assignment)
                    if problem.check_feasibility(sol).is_feasible:
                        expected = problem.evaluate(sol)
                        assert terms["objective"] == pytest.approx(expected, rel=1e-6)

    def test_penalty_decomposition_keys(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """compute_penalty_terms should return expected keys."""
        qubo = encode_qubo(tiny_instance)
        x = np.ones(qubo.n_vars, dtype=float)
        terms = compute_penalty_terms(x, qubo)
        for key in ("objective", "assignment_penalty", "capacity_penalty", "total"):
            assert key in terms


class TestQUBOGroundState:
    """Tests that QUBO ground state matches the original constrained optimum."""

    def _find_ground_state(self, instance: JobAssignmentInstance, lambda_1=None, lambda_2=None):
        """Helper: find QUBO ground state via enumeration."""
        qubo = encode_qubo(instance, lambda_1=lambda_1, lambda_2=lambda_2)
        n_vars = qubo.n_vars
        best_energy = float("inf")
        best_x = None
        for combo in itertools.product([0, 1], repeat=n_vars):
            x = np.array(combo, dtype=float)
            energy = compute_qubo_energy(x, qubo)
            if energy < best_energy:
                best_energy = energy
                best_x = x
        return qubo, best_x, best_energy

    def test_ground_state_matches_optimum(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """QUBO ground state should match enumeration-optimal solution."""
        qubo, best_x, best_energy = self._find_ground_state(tiny_instance)
        assignment, _ = decode_solution(best_x, qubo)
        assert len(assignment) == tiny_instance.n_jobs

        problem = AssignmentProblem(instance=tiny_instance)
        sol = AssignmentSolution(assignment=assignment)
        assert problem.check_feasibility(sol).is_feasible

        exact = solve_exact(problem)
        assert best_energy == pytest.approx(exact.objective, abs=0.01)

    def test_ground_state_eligible_instance(
        self, qubo_test_instance: JobAssignmentInstance
    ) -> None:
        """Ground state of QUBO should match enumeration optimum."""
        qubo, best_x, best_energy = self._find_ground_state(qubo_test_instance)
        assignment, _ = decode_solution(best_x, qubo)
        problem = AssignmentProblem(instance=qubo_test_instance)
        sol = AssignmentSolution(assignment=assignment)
        assert problem.check_feasibility(sol).is_feasible

        exact = solve_exact(problem)
        assert best_energy == pytest.approx(exact.objective, abs=0.01)

    def test_sufficient_penalty_feasible_ground_state(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """With sufficient penalties, ground state must be feasible."""
        l1, l2 = compute_penalty_weights(tiny_instance, alpha=2.0)
        qubo, best_x, _ = self._find_ground_state(
            tiny_instance, lambda_1=l1, lambda_2=l2
        )
        assignment, _ = decode_solution(best_x, qubo)
        problem = AssignmentProblem(instance=tiny_instance)
        sol = AssignmentSolution(assignment=assignment)
        assert problem.check_feasibility(sol).is_feasible


class TestInfeasibleInstance:
    """Tests for infeasible original instances in QUBO form."""

    def test_infeasible_instance_has_penalty(
        self, infeasible_capacity_instance: JobAssignmentInstance
    ) -> None:
        """Infeasible instance: QUBO ground state should have nonzero penalties."""
        qubo = encode_qubo(infeasible_capacity_instance)
        n_vars = qubo.n_vars

        best_energy = float("inf")
        best_x = None

        for combo in itertools.product([0, 1], repeat=n_vars):
            x = np.array(combo, dtype=float)
            energy = compute_qubo_energy(x, qubo)
            if energy < best_energy:
                best_energy = energy
                best_x = x

        terms = compute_penalty_terms(best_x, qubo)
        assert terms["assignment_penalty"] > 0 or terms["capacity_penalty"] > 0


class TestPenaltyWeights:
    """Tests for penalty weight selection and sufficiency."""

    def test_auto_penalty_weights_positive(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Auto-computed penalties should be positive."""
        l1, l2 = compute_penalty_weights(tiny_instance)
        assert l1 > 0
        assert l2 > 0

    def test_penalty_weights_sufficient(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Auto-computed penalties should pass sufficiency check."""
        l1, l2 = compute_penalty_weights(tiny_instance)
        sufficient, msg = check_penalty_sufficiency(tiny_instance, l1, l2)
        assert sufficient, msg

    def test_insufficient_penalty_detected(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Penalty weights too small should be detected."""
        sufficient, _ = check_penalty_sufficiency(tiny_instance, 0.001, 0.001)
        assert not sufficient

    def test_sufficient_penalty_with_alpha(
        self, tiny_instance: JobAssignmentInstance
    ) -> None:
        """Penalties with alpha > 1 should be sufficient."""
        l1, l2 = compute_penalty_weights(tiny_instance, alpha=1.5)
        sufficient, _ = check_penalty_sufficiency(tiny_instance, l1, l2)
        assert sufficient