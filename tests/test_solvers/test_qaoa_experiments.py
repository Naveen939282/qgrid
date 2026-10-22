"""Tests for QAOA finite-shot and noise experiments."""
from __future__ import annotations

import numpy as np
import pytest

from qgrid.formulation import encode_qubo, compute_qubo_energy, decode_solution
from qgrid.problem import JobAssignmentInstance, AssignmentProblem, AssignmentSolution
from qgrid.solvers.qaoa import solve_qaoa, _bitstring_to_vector
from qiskit_aer.noise import NoiseModel, depolarizing_error


@pytest.fixture
def tiny_instance():
    return JobAssignmentInstance(
        n_jobs=2, n_servers=2, job_demands=[3.0, 2.0],
        server_capacities=[5.0, 10.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
        energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
    )


@pytest.fixture
def tiny_problem(tiny_instance):
    return AssignmentProblem(instance=tiny_instance)


@pytest.fixture
def tiny_qubo(tiny_instance):
    return encode_qubo(tiny_instance)


@pytest.fixture
def deterministic_init_params():
    rng = np.random.RandomState(42)
    return rng.uniform(0, np.pi / 2, size=4)


def _is_feasible_bitstring(bitstring, qubo, problem) -> bool:
    x = _bitstring_to_vector(bitstring, qubo.n_vars)
    try:
        assignment, _ = decode_solution(x, qubo)
    except Exception:
        return False
    if not assignment:
        return False
    return problem.check_feasibility(AssignmentSolution(assignment=assignment)).is_feasible


# ---------------------------------------------------------------------------
# Finite-shot experiment tests
# ---------------------------------------------------------------------------

class TestFiniteShotExperiment:
    """Tests for finite-shot QAOA experiments."""

    def test_statevector_finite_shots(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """QAOA with finite shots should return results on all shot budgets."""
        for n_shots in [128, 512, 2048]:
            result = solve_qaoa(
                tiny_problem, tiny_qubo,
                p=2, n_shots=n_shots, maxiter=50,
                seed=42, init_params=deterministic_init_params,
                sampler_type="statevector",
            )
            assert result.n_shots == n_shots
            assert result.status in ("success", "max_iterations")
            total = sum(cnt for _, cnt in result.all_samples)
            assert total == n_shots
            assert np.isfinite(result.best_energy)
            assert len(result.best_energy_bitstring) == tiny_qubo.n_vars

    def test_best_energy_matches_evaluator(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """Reported best energy must match compute_qubo_energy for that bitstring."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=2, n_shots=512, maxiter=50,
            seed=42, init_params=deterministic_init_params,
            sampler_type="statevector",
        )
        x = _bitstring_to_vector(result.best_energy_bitstring, tiny_qubo.n_vars)
        expected_energy = compute_qubo_energy(x, tiny_qubo)
        assert result.best_energy == pytest.approx(expected_energy, abs=1e-6)

    def test_feasible_solution_is_valid(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """If a feasible solution is reported, it must actually be feasible."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=2, n_shots=512, maxiter=50,
            seed=42, init_params=deterministic_init_params,
            sampler_type="statevector",
        )
        if result.best_feasible_solution is not None:
            report = tiny_problem.check_feasibility(result.best_feasible_solution)
            assert report.is_feasible
            obj = tiny_problem.evaluate(result.best_feasible_solution)
            assert result.best_feasible_objective == pytest.approx(obj, abs=1e-6)

    def test_feasible_fraction_in_range(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """Feasible fraction should be in [0, 1]."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=128, maxiter=20,
            seed=42, init_params=deterministic_init_params[:2],
            sampler_type="statevector",
        )
        total = sum(cnt for _, cnt in result.all_samples)
        feasible = sum(
            cnt for bs, cnt in result.all_samples
            if _is_feasible_bitstring(bs, tiny_qubo, tiny_problem)
        )
        frac = feasible / total if total > 0 else 0.0
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# Aer sampler tests
# ---------------------------------------------------------------------------

class TestAerSampler:
    """Tests for Aer-based sampling in QAOA."""

    def test_aer_noiseless_valid(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """Aer noiseless sampler should run without error and produce samples."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=50,
            seed=42, init_params=deterministic_init_params[:2],
            sampler_type="aer", aer_noise_model=None,
        )
        assert result.sampler_type == "aer"
        assert len(result.all_samples) > 0
        total = sum(cnt for _, cnt in result.all_samples)
        assert total == 512

    def test_aer_with_noise_model(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """Aer with noise model should run without error and produce samples."""
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h", "rx", "rz"])
        noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["rzz"])

        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=50,
            seed=42, init_params=deterministic_init_params[:2],
            sampler_type="aer", aer_noise_model=noise,
        )
        assert result.sampler_type == "aer"
        assert len(result.all_samples) > 0
        total = sum(cnt for _, cnt in result.all_samples)
        assert total == 512

    def test_aer_reproducibility(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """Aer with same seed should produce same counts."""
        result1 = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=50,
            seed=42, init_params=deterministic_init_params[:2],
            sampler_type="aer", aer_noise_model=None,
        )
        result2 = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=512, maxiter=50,
            seed=42, init_params=deterministic_init_params[:2],
            sampler_type="aer", aer_noise_model=None,
        )
        assert dict(result1.all_samples) == dict(result2.all_samples)

    def test_noise_model_gate_coverage(self):
        """Verify noise model covers all gates used in QAOA circuit."""
        from qgrid.solvers.qaoa import _build_param_circuit
        from qgrid.formulation import encode_qubo

        inst = JobAssignmentInstance(
            n_jobs=2, n_servers=2, job_demands=[3.0, 2.0],
            server_capacities=[5.0, 10.0],
            eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
            energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
        )
        q = encode_qubo(inst)
        qc, gammas, betas = _build_param_circuit(q, p=1, n_qubits=q.n_vars)
        param_dict = {}
        for i in range(1):
            param_dict[str(gammas[i])] = 0.1
            param_dict[str(betas[i])] = 0.1
        qc_meas = qc.assign_parameters(param_dict)
        qc_meas.measure_all()

        used_gates = set(qc_meas.count_ops().keys())
        unitary_gates = {g for g in used_gates if g not in ("measure", "barrier")}

        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h", "rx", "rz"])
        noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["rzz"])

        noise_gate_names = set(noise.noise_instructions)

        assert unitary_gates.issubset(noise_gate_names | {"measure", "barrier"})


# ---------------------------------------------------------------------------
# Noise experiment tests
# ---------------------------------------------------------------------------

class TestNoiseExperiment:
    """Tests for noise model configuration and effects."""

    def test_noise_model_creation(self):
        """Verify noise model is created with correct structure."""
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h", "rx", "rz"])
        noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["rzz"])
        assert noise is not None
        assert len(noise.noise_instructions) > 0

    def test_aer_runs_with_and_without_noise(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """Verify Aer runs both with and without noise model."""
        result_clean = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=128, maxiter=30,
            seed=42, init_params=deterministic_init_params[:2],
            sampler_type="aer", aer_noise_model=None,
        )
        assert result_clean.sampler_type == "aer"
        assert len(result_clean.all_samples) > 0

        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h"])
        noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["rzz"])
        result_noisy = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=1, n_shots=128, maxiter=30,
            seed=42, init_params=deterministic_init_params[:2],
            sampler_type="aer", aer_noise_model=noise,
        )
        assert result_noisy.sampler_type == "aer"
        assert len(result_noisy.all_samples) > 0


# ---------------------------------------------------------------------------
# Experiment metrics tests
# ---------------------------------------------------------------------------

class TestExperimentMetrics:
    """Tests for experiment output structure and metrics."""

    def test_qaoa_result_fields(self, tiny_problem, tiny_qubo, deterministic_init_params):
        """Verify QAOAResult has all expected fields."""
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=2, n_shots=512, maxiter=50,
            seed=42, init_params=deterministic_init_params,
            sampler_type="statevector",
        )
        for field in ["best_feasible_objective", "best_feasible_energy",
                      "best_energy_bitstring", "best_energy", "all_samples",
                      "effective_maxiter", "sampler_type", "runtime_seconds",
                      "n_shots", "qaoa_depth", "status"]:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_effective_maxiter_bumped(self, tiny_problem, tiny_qubo):
        """Verify effective_maxiter = max(maxiter, n_params + 2)."""
        p = 2
        n_params = 2 * p  # 4
        maxiter = 3  # smaller than n_params + 2 = 6
        result = solve_qaoa(
            tiny_problem, tiny_qubo,
            p=p, n_shots=128, maxiter=maxiter,
            seed=42,
        )
        expected = max(maxiter, n_params + 2)
        assert result.effective_maxiter == expected