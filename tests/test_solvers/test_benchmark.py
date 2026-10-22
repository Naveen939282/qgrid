"""Tests for the centralized benchmark harness."""
import json
from pathlib import Path

import pytest

from qgrid.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTrial,
    compute_aggregate,
    compute_gap,
    get_dependency_versions,
    greedy_fails_instance,
    medium_benchmark_instance,
    measure_qaoa_circuit_resources,
    reconstruct_instance,
    run_benchmark,
    run_exact,
    save_benchmark,
    serialize_instance,
    tiny_benchmark_config,
    tiny_benchmark_instance,
)
from qgrid.formulation import encode_qubo
from qgrid.problem import AssignmentProblem, AssignmentSolution, JobAssignmentInstance
from qgrid.solvers.enumeration import solve_exact
from qgrid.solvers.cpsat import solve_cpsat
from qgrid.solvers.greedy import solve_greedy
from qgrid.solvers.qaoa import solve_qaoa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_instance() -> JobAssignmentInstance:
    return tiny_benchmark_instance()


@pytest.fixture
def tiny_problem(tiny_instance: JobAssignmentInstance) -> AssignmentProblem:
    return AssignmentProblem(instance=tiny_instance)


@pytest.fixture
def medium_instance() -> JobAssignmentInstance:
    return medium_benchmark_instance()


@pytest.fixture
def medium_problem(medium_instance: JobAssignmentInstance) -> AssignmentProblem:
    return AssignmentProblem(instance=medium_instance)


# ---------------------------------------------------------------------------
# Instance serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for full instance serialization."""

    def test_serialize_contains_all_fields(self, tiny_instance):
        """Serialization must capture all instance fields, not just a label."""
        data = serialize_instance(tiny_instance, "my_instance")

        # Must not be just a label
        assert data["name"] == "my_instance"
        assert "n_jobs" in data
        assert "n_servers" in data
        assert "job_demands" in data
        assert "server_capacities" in data
        assert "eligible_jobs" in data
        assert "energy_costs" in data

        assert data["n_jobs"] == tiny_instance.n_jobs
        assert data["n_servers"] == tiny_instance.n_servers
        assert data["job_demands"] == [float(d) for d in tiny_instance.job_demands]
        assert data["server_capacities"] == [float(c) for c in tiny_instance.server_capacities]
        assert data["eligible_jobs"] == [sorted(list(j)) for j in tiny_instance.eligible_jobs]
        assert len(data["energy_costs"]) == len(tiny_instance.energy_costs)

    def test_serialize_preserves_demands_and_capacities(self, tiny_instance):
        """All demands and capacities must be preserved."""
        data = serialize_instance(tiny_instance, "test")
        assert data["job_demands"] == [3.0, 2.0]
        assert data["server_capacities"] == [5.0, 10.0]

    def test_serialize_preserves_eligibility(self, tiny_instance):
        """Eligibility restrictions must be fully captured."""
        data = serialize_instance(tiny_instance, "test")
        assert data["eligible_jobs"] == [[0, 1], [0, 1]]

    def test_serialize_preserves_energy_costs(self, tiny_instance):
        """All energy costs must be preserved with exact values."""
        data = serialize_instance(tiny_instance, "test")
        assert data["energy_costs"]["(0,0)"] == 5.0
        assert data["energy_costs"]["(0,1)"] == 3.0
        assert data["energy_costs"]["(1,0)"] == 2.0
        assert data["energy_costs"]["(1,1)"] == 4.0

    def test_round_trip_preserves_instance(self, tiny_instance):
        """serialize -> reconstruct must produce an equal instance."""
        data = serialize_instance(tiny_instance, "test")
        reconstructed = reconstruct_instance(data)
        assert reconstructed == tiny_instance

    def test_round_trip_preserves_eligibility_instance(self, eligibility_instance):
        """Round-trip works for instances with eligibility restrictions."""
        problem = AssignmentProblem(instance=eligibility_instance)
        # Add to problem for metadata lookup
        data = serialize_instance(eligibility_instance, "eligibility_test")
        reconstructed = reconstruct_instance(data)
        assert reconstructed == eligibility_instance

    def test_round_trip_preserves_medium_instance(self, medium_instance):
        """Round-trip works for larger instances."""
        data = serialize_instance(medium_instance, "medium")
        reconstructed = reconstruct_instance(data)
        assert reconstructed == medium_instance


# ---------------------------------------------------------------------------
# Gap computation tests
# ---------------------------------------------------------------------------

class TestGapComputation:
    """Tests for objective gap calculation with safe zero/negative handling."""

    def test_positive_gap(self):
        """A worse candidate should give a positive gap."""
        gap = compute_gap(6.0, 5.0)
        assert gap["absolute_gap"] == 1.0
        assert gap["relative_gap"] == pytest.approx(0.2)
        assert gap["valid"] is True

    def test_zero_gap(self):
        """Candidate equal to optimum should give zero gap."""
        gap = compute_gap(5.0, 5.0)
        assert gap["absolute_gap"] == pytest.approx(0.0)
        assert gap["valid"] is True

    def test_zero_optimum_relative(self):
        """Zero optimum should produce None relative gap but valid absolute."""
        gap = compute_gap(0.0, 0.0)
        assert gap["absolute_gap"] == pytest.approx(0.0)
        assert gap["relative_gap"] is None
        assert gap["valid"] is True

    def test_nonzero_optimum_relative(self):
        """Non-zero optimum should produce valid relative gap."""
        gap = compute_gap(6.0, 5.0)
        assert gap["relative_gap"] == pytest.approx(0.2)
        assert gap["valid"] is True

    def test_negative_optimum(self):
        """Negative optimum should work correctly."""
        gap = compute_gap(-3.0, -5.0)
        assert gap["absolute_gap"] == pytest.approx(2.0)
        assert gap["relative_gap"] == pytest.approx(2.0 / 5.0)
        assert gap["valid"] is True

    def test_negative_gap_detected(self):
        """Candidate better than optimum beyond tolerance should be invalid."""
        gap = compute_gap(4.0, 5.0)  # 4 < 5, should be negative gap
        assert gap["absolute_gap"] == pytest.approx(-1.0)
        assert gap["valid"] is False

    def test_negative_gap_within_tolerance(self):
        """Tiny negative gap within tolerance is valid (numerical)."""
        gap = compute_gap(4.9999999999, 5.0)  # gap ≈ -1e-10, within tolerance
        assert gap["valid"] is True

    def test_none_candidate(self):
        """None candidate should produce all-None gap."""
        gap = compute_gap(None, 5.0)
        assert gap["absolute_gap"] is None
        assert gap["relative_gap"] is None
        assert gap["valid"] is False

    def test_none_optimum(self):
        """None optimum should produce all-None gap."""
        gap = compute_gap(5.0, None)
        assert gap["absolute_gap"] is None
        assert gap["relative_gap"] is None
        assert gap["valid"] is False

    def test_both_none(self):
        """Both None should produce all-None gap."""
        gap = compute_gap(None, None)
        assert gap["absolute_gap"] is None
        assert gap["valid"] is False


# ---------------------------------------------------------------------------
# Circuit metrics tests
# ---------------------------------------------------------------------------

class TestCircuitMetrics:
    """Tests for QAOA circuit resource measurement."""

    def test_circuit_metrics_from_actual_circuit(self, tiny_instance):
        """Circuit metrics must come from an actual QAOA circuit."""
        qubo = encode_qubo(tiny_instance)
        metrics = measure_qaoa_circuit_resources(qubo, p=1)

        assert "n_qubits" in metrics
        assert "depth" in metrics
        assert "gate_counts" in metrics
        assert "total_gates" in metrics
        assert metrics["n_qubits"] == qubo.n_vars
        assert isinstance(metrics["gate_counts"], dict)
        assert len(metrics["gate_counts"]) > 0
        assert metrics["total_gates"] == sum(metrics["gate_counts"].values())

    def test_circuit_metrics_transpilation_documented(self, tiny_instance):
        """Transpilation config must be explicitly documented."""
        qubo = encode_qubo(tiny_instance)
        metrics = measure_qaoa_circuit_resources(qubo, p=1)
        assert "transpilation_config" in metrics

    def test_circuit_metrics_depth_gt_zero(self, tiny_instance):
        """Circuit depth should be positive."""
        qubo = encode_qubo(tiny_instance)
        metrics = measure_qaoa_circuit_resources(qubo, p=1)
        assert metrics["depth"] > 0


# ---------------------------------------------------------------------------
# Solver execution wrapper tests
# ---------------------------------------------------------------------------

class TestSolverExecution:
    """Tests for individual solver execution wrappers."""

    def test_exact_solver_returns_objective(self, tiny_problem, tiny_instance):
        """Exact solver must find the optimal solution."""
        trial = run_exact(tiny_problem, tiny_instance, max_combinations=10_000_000)
        assert trial.method == "exact"
        assert trial.status == "optimal"
        assert trial.objective == 5.0
        assert trial.is_optimal is True
        assert trial.feasible is True

    def test_exact_solver_too_large(self, tiny_problem, tiny_instance):
        """Exact solver with small limit should mark as too_large."""
        trial = run_exact(tiny_problem, tiny_instance, max_combinations=1)
        assert trial.status == "too_large"
        assert trial.objective is None
        assert trial.feasible is False

    def test_cpsat_solver_returns_result(self, tiny_problem):
        """CP-SAT should produce a valid trial."""
        from qgrid.benchmark import run_cpsat
        trial = run_cpsat(tiny_problem, time_limit=5.0)
        assert trial.method == "cpsat"

    def test_greedy_solver_returns_result(self, tiny_problem):
        """Greedy should produce a valid trial."""
        from qgrid.benchmark import run_greedy
        trial = run_greedy(tiny_problem)
        assert trial.method == "greedy"
        assert trial.is_optimal is False
        assert trial.is_optimal is False


# ---------------------------------------------------------------------------
# Full benchmark run tests
# ---------------------------------------------------------------------------

class TestBenchmarkRun:
    """Tests for the full benchmark runner."""

    def test_run_benchmark_returns_result(self):
        """run_benchmark must return a BenchmarkResult."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        assert isinstance(result, BenchmarkResult)

    def test_run_benchmark_all_methods(self):
        """All four solver methods must be present in trials."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        methods = {t.method for t in result.trials}
        assert methods == {"exact", "cpsat", "greedy", "qaoa"}

    def test_run_benchmark_exact_optimum(self):
        """Exact optimum must be computed and stored in aggregate."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        assert result.aggregate["exact_optimum"] == 5.0

    def test_run_benchmark_qaoa_gap(self):
        """QAOA trial must have objective_gap in metadata."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        qaoa_trials = [t for t in result.trials if t.method == "qaoa"]
        assert len(qaoa_trials) == 1
        assert "objective_gap" in qaoa_trials[0].metadata

    def test_run_benchmark_qaoa_circuit_metrics(self):
        """QAOA trial must have circuit_metrics in metadata."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        qaoa_trials = [t for t in result.trials if t.method == "qaoa"]
        assert len(qaoa_trials) == 1
        assert "circuit_metrics" in qaoa_trials[0].metadata
        assert "circuit_metrics" in qaoa_trials[0].metadata


# ---------------------------------------------------------------------------
# JSON output and schema tests
# ---------------------------------------------------------------------------

class TestJsonOutput:
    """Tests for JSON serialization and schema."""

    def test_save_benchmark_creates_json(self, tmp_path):
        """save_benchmark must create a JSON file."""
        from qgrid.benchmark import result_to_dict
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)

        assert Path(path).exists()
        with open(path, "r") as f:
            data = json.load(f)
        assert data["schema_version"] == BENCHMARK_SCHEMA_VERSION

    def test_json_contains_instance_definition(self, tmp_path):
        """JSON must contain complete instance definition."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)

        with open(path, "r") as f:
            data = json.load(f)
        inst = data["instance_definition"]
        assert inst["n_jobs"] == 2
        assert inst["n_servers"] == 2
        assert len(inst["energy_costs"]) == 4

    def test_json_timestamped_filename(self, tmp_path):
        """Filename must include timestamp for uniqueness."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path1 = save_benchmark(result, results_dir=tmp_path)
        import time as _time
        _time.sleep(0.01)
        path2 = save_benchmark(result, results_dir=tmp_path)

        assert path1 != path2

    def test_json_contains_dependency_versions(self, tmp_path):
        """JSON must contain dependency versions."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)

        with open(path, "r") as f:
            data = json.load(f)
        assert "dependency_versions" in data
        assert "qiskit" in data["dependency_versions"]
        assert "numpy" in data["dependency_versions"]
        assert "python" in data["dependency_versions"]

    def test_json_contains_qaoa_config(self, tmp_path):
        """JSON must contain QAOA configuration."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)

        with open(path, "r") as f:
            data = json.load(f)
        assert data["configuration"]["qaoa_depth"] == 1
        assert data["configuration"]["n_shots"] == 512
        assert "seeds" in data["configuration"]
        assert "seeds" in data["configuration"]


# ---------------------------------------------------------------------------
# Per-trial and aggregate structure tests
# ---------------------------------------------------------------------------

class TestResultStructure:
    """Tests for result structure integrity."""

    def test_trials_have_all_required_fields(self):
        """Each trial must have all required fields."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        for trial in result.trials:
            assert trial.method in {"exact", "cpsat", "greedy", "qaoa"}
            assert isinstance(trial.status, str)
            assert isinstance(trial.runtime_seconds, float)
            assert isinstance(trial.feasible, bool)

    def test_aggregate_has_all_methods(self):
        """Aggregate must have entries for all methods."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        for method in ["exact", "cpsat", "greedy", "qaoa"]:
            assert method in result.aggregate["methods"]

    def test_aggregate_preserves_raw_trials(self):
        """Aggregate must preserve raw per-trial values."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        for method, summary in result.aggregate["methods"].items():
            assert "raw_objectives" in summary
            assert "n_trials" in summary
            assert summary["n_trials"] == len(summary["raw_objectives"])

    def test_qaoa_has_seed(self):
        """QAOA trials must have seed recorded."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        qaoa_trials = [t for t in result.trials if t.method == "qaoa"]
        for t in qaoa_trials:
            assert t.seed is not None
            assert t.seed is not None


# ---------------------------------------------------------------------------
# Existing API compatibility tests
# ---------------------------------------------------------------------------

class TestApiCompatibility:
    """Tests that existing solver APIs and experiment scripts still work."""

    def test_solve_exact_still_works(self, tiny_problem):
        """solve_exact must still work directly."""
        result = solve_exact(tiny_problem)
        assert result.objective is not None

    def test_solve_cpsat_still_works(self, tiny_problem):
        """solve_cpsat must still work directly."""
        result = solve_cpsat(tiny_problem, time_limit=5.0)
        assert result is not None

    def test_solve_greedy_still_works(self, tiny_problem):
        """solve_greedy must still work directly."""
        result = solve_greedy(tiny_problem)
        assert result is not None

    def test_solve_qaoa_still_works(self, tiny_problem, tiny_instance):
        """solve_qaoa must still work directly."""
        qubo = encode_qubo(tiny_instance)
        result = solve_qaoa(
            problem=tiny_problem,
            qubo=qubo,
            p=1,
            n_shots=128,
            maxiter=50,
            seed=42,
        )
        assert result is not None
        assert hasattr(result, "best_feasible_solution")
        assert hasattr(result, "best_feasible_objective")

    def test_experiment_script_module_exists(self):
        """Existing experiment scripts must still be importable."""
        import os
        assert os.path.exists("run_qaoa_experiment.py")
        assert os.path.exists("run_qaoa_finite_shot_experiment.py")
