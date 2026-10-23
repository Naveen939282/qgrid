"""Tests for the centralized benchmark harness."""
import json
from pathlib import Path

import pytest

from qgrid.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTrial,
    asymmetric_3job_3server_instance,
    compute_aggregate,
    compute_gap,
    cost_asymmetry_2job_2server_instance,
    get_dependency_versions,
    greedy_fails_instance,
    medium_benchmark_instance,
    measure_qaoa_circuit_resources,
    reconstruct_instance,
    run_benchmark,
    run_exact,
    save_benchmark,
    serialize_instance,
    tight_capacity_3job_2server_instance,
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

    def test_candidate_equal_optimum(self):
        """Candidate equal to optimum should give zero gap, valid."""
        gap = compute_gap(5.0, 5.0)
        assert gap["absolute_gap"] == pytest.approx(0.0)
        assert gap["valid"] is True

    def test_candidate_worse_than_optimum(self):
        """Candidate worse than optimum should give positive gap, valid."""
        gap = compute_gap(10.0, 5.0)
        assert gap["absolute_gap"] == pytest.approx(5.0)
        assert gap["relative_gap"] == pytest.approx(1.0)
        assert gap["valid"] is True

    def test_candidate_lower_than_optimum_beyond_tolerance(self):
        """Candidate below optimum by > tolerance must be flagged invalid."""
        # Exact optimum is 5.0, candidate claims 3.0 (should be impossible)
        gap = compute_gap(3.0, 5.0)
        assert gap["absolute_gap"] == pytest.approx(-2.0)
        assert gap["valid"] is False
        # Must NOT hide the negative gap with abs()
        assert gap["absolute_gap"] < 0

    def test_tiny_negative_gap_within_tolerance(self):
        """Tiny negative gap within tolerance is considered valid (numerical)."""
        # Use a gap of ~1e-10, well within the 1e-9 tolerance
        gap = compute_gap(4.9999999999, 5.0)
        assert gap["absolute_gap"] < 0  # Slightly negative
        assert gap["valid"] is True

    def test_negative_optimum_zero_gap(self):
        """Zero gap when optimum is negative."""
        gap = compute_gap(-5.0, -5.0)
        assert gap["absolute_gap"] == pytest.approx(0.0)
        # |optimum| = 5.0 > tolerance, so relative_gap should be 0.0
        assert gap["relative_gap"] == pytest.approx(0.0)
        assert gap["valid"] is True

    def test_negative_optimum_worse_candidate(self):
        """Negative optimum with worse candidate: positive gap, valid."""
        gap = compute_gap(-3.0, -5.0)
        assert gap["absolute_gap"] == pytest.approx(2.0)
        assert gap["relative_gap"] == pytest.approx(2.0 / 5.0)
        assert gap["valid"] is True


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

    def test_json_contains_utc_timestamps(self, tmp_path):
        """JSON must contain timezone-aware UTC timestamps."""
        from datetime import datetime

        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)

        with open(path, "r") as f:
            data = json.load(f)
        ts = data["timestamps"]
        assert "utc_start" in ts
        assert "utc_end" in ts

        # Parse and verify UTC offset is zero
        start = datetime.fromisoformat(ts["utc_start"])
        end = datetime.fromisoformat(ts["utc_end"])
        assert start.tzinfo is not None
        assert end.tzinfo is not None
        assert start.utcoffset().total_seconds() == 0.0
        assert end.utcoffset().total_seconds() == 0.0

    def test_json_timestamps_start_before_end(self, tmp_path):
        """Start timestamp must not occur after end timestamp."""
        from datetime import datetime

        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)

        with open(path, "r") as f:
            data = json.load(f)
        ts = data["timestamps"]
        start = datetime.fromisoformat(ts["utc_start"])
        end = datetime.fromisoformat(ts["utc_end"])
        assert start <= end


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
# Extended aggregate statistics tests
# ---------------------------------------------------------------------------

class TestExtendedAggregate:
    """Tests for extended aggregate metrics (Checkpoint B1)."""

    def test_n_attempted_and_n_completed(self):
        """Aggregate must report n_attempted and n_completed."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        for method, summary in result.aggregate["methods"].items():
            assert "n_attempted" in summary
            assert "n_completed" in summary
            assert summary["n_attempted"] == len(summary["raw_status"])
            assert summary["n_completed"] == summary["n_attempted"]

    def test_n_failed_field(self):
        """Aggregate must report n_failed."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        for method, summary in result.aggregate["methods"].items():
            assert "n_failed" in summary
            assert summary["n_failed"] == 0

    def test_failed_trials_counted_in_n_attempted(self):
        """Manually constructed aggregate must count failed trials."""
        trials = [
            BenchmarkTrial(
                method="qaoa", seed=0, objective=None, status="error",
                is_optimal=False, runtime_seconds=1.0, feasible=False,
                solution=None, message="error",
            ),
            BenchmarkTrial(
                method="qaoa", seed=1, objective=5.0, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=True,
                solution={"assignment": {0: 1, 1: 0}},
            ),
        ]
        agg = compute_aggregate(trials, exact_obj=5.0)
        s = agg["methods"]["qaoa"]
        assert s["n_attempted"] == 2
        assert s["n_failed"] == 1
        assert s["n_completed"] == 1
        assert s["n_feasible"] == 1
        assert s["feasible_rate"] == pytest.approx(0.5)

    def test_feasible_rate_denominator(self):
        """feasible_rate must use n_attempted as denominator."""
        trials = [
            BenchmarkTrial(
                method="qaoa", seed=0, objective=5.0, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=True,
                solution={"assignment": {0: 1, 1: 0}},
            ),
            BenchmarkTrial(
                method="qaoa", seed=1, objective=None, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=False,
                solution=None,
            ),
        ]
        agg = compute_aggregate(trials, exact_obj=5.0)
        s = agg["methods"]["qaoa"]
        assert s["n_attempted"] == 2
        assert s["n_feasible"] == 1
        assert s["feasible_rate"] == pytest.approx(0.5)

    def test_matched_optimum_count(self):
        """matched_optimum_count counts feasible trials matching exact optimum."""
        trials = [
            BenchmarkTrial(
                method="qaoa", seed=0, objective=5.0, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=True,
                solution={"assignment": {0: 1, 1: 0}},
            ),
            BenchmarkTrial(
                method="qaoa", seed=1, objective=6.0, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=True,
                solution={"assignment": {0: 0, 1: 0}},
            ),
        ]
        agg = compute_aggregate(trials, exact_obj=5.0)
        s = agg["methods"]["qaoa"]
        assert s["matched_optimum_count"] == 1
        assert s["matched_optimum_rate"] == pytest.approx(0.5)

    def test_qaoa_not_counted_as_proven_optimal(self):
        """QAOA matching optimum is not proven optimal."""
        trials = [
            BenchmarkTrial(
                method="qaoa", seed=0, objective=5.0, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=True,
                solution={"assignment": {0: 1, 1: 0}},
            ),
        ]
        agg = compute_aggregate(trials, exact_obj=5.0)
        s = agg["methods"]["qaoa"]
        assert s["proven_optimal_count"] == 0
        assert s["matched_optimum_count"] == 1

    def test_exact_proven_optimal(self):
        """Exact solver trials should be counted as proven_optimal."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        s = result.aggregate["methods"]["exact"]
        assert s["proven_optimal_count"] == 1

    def test_feasible_only_objective_stats(self):
        """Objective stats computed over feasible trials only."""
        trials = [
            BenchmarkTrial(
                method="qaoa", seed=0, objective=5.0, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=True,
                solution={"assignment": {0: 1, 1: 0}},
            ),
            BenchmarkTrial(
                method="qaoa", seed=1, objective=8.0, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=True,
                solution={"assignment": {0: 0, 1: 0}},
            ),
            BenchmarkTrial(
                method="qaoa", seed=2, objective=None, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=False,
                solution=None,
            ),
        ]
        agg = compute_aggregate(trials, exact_obj=5.0)
        s = agg["methods"]["qaoa"]
        assert s["n_feasible_objectives"] == 2
        assert s["best_feasible_objective"] == 5.0
        assert s["worst_feasible_objective"] == 8.0
        assert s["mean_feasible_objective"] == pytest.approx(6.5)
        assert s["median_feasible_objective"] == pytest.approx(6.5)
        assert s["std_feasible_objective"] is not None

    def test_empty_feasible_set(self):
        """All infeasible trials should produce null feasible stats."""
        trials = [
            BenchmarkTrial(
                method="qaoa", seed=0, objective=None, status="success",
                is_optimal=False, runtime_seconds=1.0, feasible=False,
                solution=None,
            ),
        ]
        agg = compute_aggregate(trials, exact_obj=5.0)
        s = agg["methods"]["qaoa"]
        assert s["n_feasible"] == 0
        assert s["feasible_rate"] == 0.0
        assert s["best_feasible_objective"] is None
        assert s["best_feasible_gap"]["absolute_gap"] is None

    def test_missing_exact_optimum(self):
        """Missing exact optimum should produce null gaps."""
        config = tiny_benchmark_config()
        config.exact_max_combinations = 0
        result = run_benchmark(config)
        s = result.aggregate["methods"]["qaoa"]
        assert s["best_feasible_gap"]["absolute_gap"] is None
        assert s["best_feasible_gap"]["valid"] is False

    def test_deterministic_solvers_one_trial(self):
        """Deterministic solvers should have exactly 1 trial."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        for method in ["exact", "cpsat", "greedy"]:
            s = result.aggregate["methods"][method]
            assert s["n_attempted"] == 1

    def test_multi_seed_config_serialization(self):
        """Multi-seed config must serialize all seeds."""
        config = BenchmarkConfig(
            instance=tiny_benchmark_instance(),
            instance_name="tiny_2job_2server",
            seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            qaoa_depth=1, n_shots=512, maxiter=50,
        )
        result = run_benchmark(config)
        assert result.configuration["seeds"] == list(range(10))
        qaoa_trials = [t for t in result.trials if t.method == "qaoa"]
        assert len(qaoa_trials) == 10
        seeds_in_trials = sorted(t.seed for t in qaoa_trials)
        assert seeds_in_trials == list(range(10))

    def test_aggregate_contains_verification_status(self):
        """Configuration must record exact optimum verification status."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        assert "exact_optimum_verified" in result.configuration
        assert result.configuration["exact_optimum_verified"] is True

    def test_json_valid_serialization(self, tmp_path):
        """Saved JSON must be valid JSON with no NaN."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)
        with open(path, "r") as f:
            data = json.load(f)
        assert "aggregate" in data
        assert "methods" in data["aggregate"]

    def test_json_multi_seed_saves_all_trials(self, tmp_path):
        """Multi-seed runs must save all trials."""
        config = BenchmarkConfig(
            instance=tiny_benchmark_instance(),
            instance_name="tiny_2job_2server",
            seeds=[0, 1, 2],
            qaoa_depth=1, n_shots=512, maxiter=50,
        )
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=tmp_path)
        with open(path, "r") as f:
            data = json.load(f)
        qaoa_trials = [t for t in data["trials"] if t["method"] == "qaoa"]
        assert len(qaoa_trials) == 3

    def test_qubo_resources_recorded(self):
        """Instance serialization must include QUBO resource counts."""
        config = tiny_benchmark_config()
        result = run_benchmark(config)
        qb = result.instance_definition["qubo_resources"]
        assert "n_qubo_variables" in qb
        assert "n_assignment_bits" in qb
        assert "n_slack_bits" in qb
        assert qb["n_qubo_variables"] == qb["n_assignment_bits"] + qb["n_slack_bits"]


# ---------------------------------------------------------------------------
# API compatibility tests
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

    def test_run_benchmark_script_exists(self):
        """run_benchmark.py must still exist."""
        import os
        assert os.path.exists("run_benchmark.py")

    def test_multi_seed_script_exists(self):
        """multi_seed_benchmark.py must exist."""
        import os
        assert os.path.exists("multi_seed_benchmark.py")
