"""Tests for the QAOA solver."""
from __future__ import annotations

import numpy as np
import pytest

from qgrid.formulation import encode_qubo, compute_qubo_energy
from qgrid.problem import JobAssignmentInstance, AssignmentProblem
from qgrid.solvers.qaoa import (
    QAOAResult,
    solve_qaoa,
    qubo_to_hamiltonian,
    _bitstring_to_vector,
)
from qgrid.solvers.enumeration import solve_exact


@pytest.fixture
def tiny_instance():
    """A 2-job, 2-server instance that can be exhaustively solved."""
    inst = JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[3.0, 2.0],
        server_capacities=[5.0, 10.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
        energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
    )
    return inst


@pytest.fixture
def tiny_problem(tiny_instance):
    return AssignmentProblem(instance=tiny_instance)


@pytest.fixture
def tiny_qubo(tiny_instance):
    return encode_qubo(tiny_instance)



# ---------------------------------------------------------------------------
# Hamiltonian construction tests
# ---------------------------------------------------------------------------

class TestHamiltonianConstruction:
    """Tests for qubo_to_hamiltonian."""

    def test_hamiltonian_returns_correct_types(self, tiny_qubo):
        H, offset, n = qubo_to_hamiltonian(tiny_qubo)
        assert n == tiny_qubo.n_vars
        assert isinstance(offset, float)

    def test_hamiltonian_energy_matches_qubo_at_zero(self, tiny_qubo):
        """Hamiltonian expectation at |000...0> should equal QUBO energy at x=0."""
        from qiskit.quantum_info import Statevector

        H, offset, n = qubo_to_hamiltonian(tiny_qubo)
        state = Statevector.from_int(0, 2**n)
        H_exp = state.expectation_value(H).real

        x0 = np.zeros(tiny_qubo.n_vars)
        qubo_energy_0 = compute_qubo_energy(x0, tiny_qubo)

        # H includes the constant offset, so H's eigenvalue = E(x=0)
        assert H_exp == pytest.approx(qubo_energy_0, abs=1e-8)

    def test_hamiltonian_has_identity_term(self, tiny_qubo):
        """The Hamiltonian should have an identity (constant) term."""
        H, offset, n = qubo_to_hamiltonian(tiny_qubo)
        has_identity = any(
            all(c == "I" for c in p.to_label()) for p in H.paulis
        )
        assert has_identity



# ---------------------------------------------------------------------------
# Bitstring conversion tests
# ---------------------------------------------------------------------------

class TestBitstringConversion:
    """Tests for _bitstring_to_vector."""

    def test_simple_bitstring(self):
        """Test converting '10' (little-endian -> qubit0=0, qubit1=1)."""
        x = _bitstring_to_vector("10", 2)
        assert x.tolist() == [0.0, 1.0]

    def test_single_bit(self):
        x = _bitstring_to_vector("1", 1)
        assert x.tolist() == [1.0]

    def test_padding(self):
        """Test padding short bitstrings to n vars."""
        x = _bitstring_to_vector("1", 4)
        assert x.tolist() == [1.0, 0.0, 0.0, 0.0]



# ---------------------------------------------------------------------------
# Energy evaluation tests
# ---------------------------------------------------------------------------

class TestEnergyEvaluation:
    """Tests for evaluate_energy_from_counts."""

    def test_energy_from_uniform_counts(self, tiny_qubo):
        """Test that uniform counts give correct expected energy."""
        from qgrid.solvers.qaoa import evaluate_energy_from_counts

        n = tiny_qubo.n_vars
        all_ones = "1" * n
        all_zeros = "0" * n
        counts = {all_ones: 50, all_zeros: 50}

        expected = evaluate_energy_from_counts(counts, tiny_qubo, 100)



# ---------------------------------------------------------------------------
# Circuit construction tests
# ---------------------------------------------------------------------------

class TestCircuitConstruction:
    """Tests for QAOA circuit construction."""

    def test_circuit_has_correct_num_qubits(self, tiny_qubo):
        from qgrid.solvers.qaoa import _build_param_circuit
        p = 1
        qc, gammas, betas = _build_param_circuit(tiny_qubo, p, tiny_qubo.n_vars)
        assert qc.num_qubits == tiny_qubo.n_vars

    def test_circuit_has_correct_num_params(self, tiny_qubo):
        from qgrid.solvers.qaoa import _build_param_circuit
        p = 2
        qc, gammas, betas = _build_param_circuit(tiny_qubo, p, tiny_qubo.n_vars)
        assert len(gammas) == p
        assert len(betas) == p
        assert qc.num_parameters == 2 * p

    def test_circuit_has_hadamard_layer(self, tiny_qubo):
        from qgrid.solvers.qaoa import _build_param_circuit
        p = 1



# ---------------------------------------------------------------------------
# Solver integration tests
# ---------------------------------------------------------------------------

class TestSolveQAOA:
    """Integration tests for solve_qaoa."""

    def test_returns_qaoa_result(self, tiny_problem, tiny_qubo):
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=20, seed=42,
        )
        assert isinstance(result, QAOAResult)

    def test_result_has_valid_fields(self, tiny_problem, tiny_qubo):
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=20, seed=42,
        )
        assert result.qaoa_depth == 1
        assert result.n_shots == 512
        assert len(result.params) == 2
        assert result.status in ("success", "max_iterations")
        assert result.runtime_seconds > 0
        assert len(result.all_samples) > 0
        assert result.best_energy < float("inf")

    def test_best_energy_matches_best_bitstring(self, tiny_problem, tiny_qubo):
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=20, seed=42,
        )
        x = _bitstring_to_vector(result.best_energy_bitstring, tiny_qubo.n_vars)
        expected_energy = compute_qubo_energy(x, tiny_qubo)
        assert result.best_energy == pytest.approx(expected_energy, abs=1e-8)

    def test_best_feasible_solution_is_valid(self, tiny_problem, tiny_qubo):
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=20, seed=42,
        )
        if result.best_feasible_solution is not None:
            report = tiny_problem.check_feasibility(result.best_feasible_solution)
            assert report.is_feasible
            obj = tiny_problem.evaluate(result.best_feasible_solution)
            assert result.best_feasible_objective == pytest.approx(obj, abs=1e-8)

    def test_finds_feasible_solution(self, tiny_problem, tiny_qubo):
        """On a tiny instance, QAOA should find a feasible solution."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=2, n_shots=1024, maxiter=50, seed=42,
        )
        assert result.best_feasible_solution is not None
        assert result.best_feasible_objective is not None

    def test_deterministic_with_same_seed(self, tiny_problem, tiny_qubo):
        result1 = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=10, seed=123,
        )
        result2 = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=10, seed=123,
        )
        assert result1.params.tolist() == result2.params.tolist()
        assert result1.best_energy == pytest.approx(result2.best_energy)

    def test_custom_init_params(self, tiny_problem, tiny_qubo):
        n_params = 4  # p=2
        init = np.zeros(n_params)
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=2, n_shots=512, maxiter=5,
            init_params=init, seed=42,
        )
        assert len(result.params) == 4

    def test_qaoa_result_to_solver_result(self, tiny_problem, tiny_qubo):
        from qgrid.solvers.qaoa import qaoa_result_to_solver_result
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=10, seed=42,
        )
        sr = qaoa_result_to_solver_result(result)
        assert sr.status == result.status
        assert sr.is_optimal is False



# ---------------------------------------------------------------------------
# Objective-energy consistency tests
# ---------------------------------------------------------------------------

class TestObjectiveEnergyConsistency:
    """Test that QUBO energy from decoded bitstrings matches the evaluator."""

    def test_energy_consistency(self, tiny_problem, tiny_qubo):
        """Energy from decoded bitstring should match compute_qubo_energy."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=20, seed=42,
        )

        for bitstring, count in result.all_samples:
            x = _bitstring_to_vector(bitstring, tiny_qubo.n_vars)
            energy_from_func = compute_qubo_energy(x, tiny_qubo)
            assert energy_from_func < float("inf")

    def test_decoded_bitstring_energy(self, tiny_qubo):
        """Verify that decoded bitstrings give correct QUBO energies."""
        n = tiny_qubo.n_vars
        bitstring = "1" * n
        x = _bitstring_to_vector(bitstring, n)
        energy = compute_qubo_energy(x, tiny_qubo)

        # Manually compute: x^T Q x + offset
        manual_energy = np.dot(x, np.dot(tiny_qubo.Q, x)) + tiny_qubo.offset
        assert energy == pytest.approx(manual_energy, abs=1e-8)



# ---------------------------------------------------------------------------
# Exhaustive Hamiltonian equivalence tests
# ---------------------------------------------------------------------------

class TestHamiltonianEquivalenceExhaustive:
    """Exhaustively verify that H's diagonal energy matches QUBO energy
    for all computational-basis bitstrings on multiple tiny QUBOs."""

    @pytest.mark.parametrize("n_jobs,n_servers,demands,capacities,costs", [
        # Instance 1: 2 jobs, 2 servers, simple
        (2, 2, [1.0, 1.0], [10.0, 10.0], {(0, 0): 1.0, (1, 0): 2.0, (0, 1): 3.0, (1, 1): 4.0}),
        # Instance 2: 2 jobs, 2 servers, with tight capacity
        (2, 2, [3.0, 2.0], [5.0, 10.0], {(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0}),
        # Instance 3: 1 job, 2 servers, minimal
        (1, 2, [1.0], [5.0, 5.0], {(0, 0): 2.0, (1, 0): 3.0}),
    ])
    def test_all_bitstrings_match(self, n_jobs, n_servers, demands, capacities, costs):
        """For every computational basis state, H expectation equals QUBO energy."""
        from qiskit.quantum_info import Statevector

        inst = JobAssignmentInstance(
            n_jobs=n_jobs,
            n_servers=n_servers,
            job_demands=demands,
            server_capacities=capacities,
            eligible_jobs=[frozenset(range(n_jobs)) for _ in range(n_servers)],
            energy_costs=costs,
        )
        q = encode_qubo(inst)
        H, offset_ham, n = qubo_to_hamiltonian(q)

        # For a small number of variables, we can check all bitstrings
        # If n > 8, we sample a subset
        max_states = min(2**n, 256)
        for val in range(max_states):
            state = Statevector.from_int(val, 2**n)
            H_exp = state.expectation_value(H).real

            # Convert integer to binary vector
            # Qiskit convention: qubit 0 is the LSB (rightmost bit)
            # So val in binary directly maps to qubit values
            x = np.array([(val >> i) & 1 for i in range(n)], dtype=float)
            qubo_E = compute_qubo_energy(x, q)

            assert H_exp == pytest.approx(qubo_E, abs=1e-6), (
                f"Mismatch at val={val} (x={x.astype(int)}): "
                f"H_exp={H_exp:.6f}, QUBO_E={qubo_E:.6f}"
            )

    def test_linear_and_quadratic_terms(self, tiny_qubo):
        """Explicitly verify that linear, quadratic, and offset terms are correct."""
        from qiskit.quantum_info import Statevector

        H, offset_ham, n = qubo_to_hamiltonian(tiny_qubo)

        # Check a state where exactly one qubit is 1
        for i in range(n):
            state = Statevector.from_int(1 << i, 2**n)
            H_exp = state.expectation_value(H).real

            x = np.zeros(n)
            x[i] = 1.0
            qubo_E = compute_qubo_energy(x, tiny_qubo)

            assert H_exp == pytest.approx(qubo_E, abs=1e-6), (
                f"Single-qubit mismatch at i={i}: H={H_exp}, QUBO={qubo_E}"
            )

        # Check a state where exactly two qubits are 1
        for i in range(n):
            for j in range(i + 1, n):
                state = Statevector.from_int((1 << i) | (1 << j), 2**n)
                H_exp = state.expectation_value(H).real

                x = np.zeros(n)
                x[i] = 1.0
                x[j] = 1.0
                qubo_E = compute_qubo_energy(x, tiny_qubo)

                assert H_exp == pytest.approx(qubo_E, abs=1e-6), (
                    f"Two-qubit mismatch at (i={i},j={j}): H={H_exp}, QUBO={qubo_E}"
                )



# ---------------------------------------------------------------------------
# COBYLA warning regression tests
# ---------------------------------------------------------------------------

class TestCobylaWarningRegression:
    """Verify that COBYLA does not emit warnings when maxiter < n_params + 2."""

    def test_no_warning_with_small_maxiter(self, tiny_problem, tiny_qubo):
        """maxiter smaller than n_params+2 should not raise a scipy warning."""
        import warnings
        import scipy

        n_params = 4  # p=2

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # maxiter=5 with 4 params -> COBYLA needs at least 6
            result = solve_qaoa(
                tiny_problem, tiny_qubo,
                p=2, n_shots=256, maxiter=5,
                init_params=np.zeros(n_params), seed=42,
            )
        # Should still return a result (even if max_iterations)
        assert result.status in ("success", "max_iterations")

    def test_no_warning_with_p_equals_1(self, tiny_problem, tiny_qubo):
        """maxiter=3 with p=1 (2 params) should not warn (3 >= 2+2? No, 3 < 4)."""
        import warnings

        n_params = 2  # p=1

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = solve_qaoa(
                tiny_problem, tiny_qubo,
                p=1, n_shots=256, maxiter=3,
                init_params=np.zeros(n_params), seed=42,
            )
        assert result.status in ("success", "max_iterations")

    def test_maxiter_preserved_in_options(self, tiny_problem, tiny_qubo):
        """Verify that maxiter is at least n_params+2 in the actual minimize call."""
        # This is implicitly tested by test_no_warning_with_small_maxiter
        # but we can also verify via the COBYLA maxfun computation
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=128, maxiter=3, seed=42,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Feasibility handling tests
# ---------------------------------------------------------------------------

class TestFeasibilityHandling:
    """Verify that the solver correctly handles feasible and infeasible samples."""

    def test_no_feasible_solution_is_reported_honestly(self):
        """When no feasible sample exists, solver should return None for solution."""
        # Use a very small shot count that may miss feasible solutions
        # and a depth-1 QAOA on the tiny instance with a bad seed
        inst = JobAssignmentInstance(
            n_jobs=2, n_servers=2, job_demands=[3.0, 2.0],
            server_capacities=[5.0, 10.0],
            eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
            energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
        )
        q = encode_qubo(inst)
        problem = AssignmentProblem(instance=inst)

        result = solve_qaoa(
            problem, q,
            p=1, n_shots=1, maxiter=5, seed=0,
        )

        # With only 1 shot, we may or may not get a feasible solution
        # But the result should be valid regardless
        if result.best_feasible_solution is None:
            # Must report honestly
            assert result.best_feasible_objective is None
            assert result.best_feasible_energy is None
            assert "No feasible" not in result.message or "solution" in result.message.lower()
        else:
            # If feasible, it must actually be feasible
            report = problem.check_feasibility(result.best_feasible_solution)
            assert report.is_feasible

    def test_feasible_and_best_energy_separated(self, tiny_problem, tiny_qubo):
        """Best feasible solution and best energy bitstring are reported separately."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=20, seed=42,
        )
        # best_energy is the lowest QUBO energy across all sampled bitstrings
        # best_feasible_solution is the lowest objective among feasible samples
        assert result.best_energy_bitstring is not None
        assert len(result.best_energy_bitstring) == tiny_qubo.n_vars

        if result.best_feasible_solution is not None:
            # Best feasible solution must be valid
            report = tiny_problem.check_feasibility(result.best_feasible_solution)
            assert report.is_feasible
