"""Centralized benchmark harness for Q-Grid solvers.

This module provides a reproducible benchmark harness that runs exact
enumeration, CP-SAT, greedy, and QAOA solvers against the same instance,
records per-trial results, and saves timestamped JSON outputs.

Usage:
    from qgrid.benchmark import run_benchmark, tiny_benchmark_config
    results = run_benchmark(tiny_benchmark_config())
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from qgrid.formulation import encode_qubo
from qgrid.problem import AssignmentProblem, AssignmentSolution, JobAssignmentInstance
from qgrid.solvers.enumeration import solve_exact
from qgrid.solvers.cpsat import solve_cpsat
from qgrid.solvers.greedy import solve_greedy
from qgrid.solvers.qaoa import solve_qaoa


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

BENCHMARK_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Instance serialization
# ---------------------------------------------------------------------------

def serialize_instance(
    instance: JobAssignmentInstance,
    name: str,
) -> dict:
    """Serialize a JobAssignmentInstance to a JSON-compatible dict.

    Captures the complete instance definition including job demands,
    server capacities, eligibility restrictions, and energy costs.
    """
    return {
        "name": name,
        "n_jobs": instance.n_jobs,
        "n_servers": instance.n_servers,
        "job_demands": [float(d) for d in instance.job_demands],
        "server_capacities": [float(c) for c in instance.server_capacities],
        "eligible_jobs": [
            sorted(list(jobs)) for jobs in instance.eligible_jobs
        ],
        "energy_costs": {
            f"({s},{j})": float(cost)
            for (s, j), cost in instance.energy_costs.items()
        },
    }


def reconstruct_instance(data: dict) -> JobAssignmentInstance:
    """Reconstruct a JobAssignmentInstance from serialized dict."""
    eligible_jobs = [
        frozenset(jobs) for jobs in data["eligible_jobs"]
    ]
    energy_costs = {
        tuple(int(x) for x in k.strip("()").split(",")): v
        for k, v in data["energy_costs"].items()
    }
    return JobAssignmentInstance(
        n_jobs=data["n_jobs"],
        n_servers=data["n_servers"],
        job_demands=data["job_demands"],
        server_capacities=data["server_capacities"],
        eligible_jobs=eligible_jobs,
        energy_costs=energy_costs,
    )


# ---------------------------------------------------------------------------
# Gap computation
# ---------------------------------------------------------------------------

GAP_TOLERANCE = 1e-9


def compute_gap(
    candidate_objective: Optional[float],
    exact_optimum: Optional[float],
) -> dict:
    """Compute minimization objective gap.

    For a minimization problem, the gap is:
        absolute_gap = candidate - exact_optimum

    A negative gap beyond numerical tolerance indicates the candidate
    reported a lower objective than the proven optimum, which should be
    investigated.

    Relative gap is only reported when the exact optimum is non-zero
    and has a valid sign:
        relative_gap = absolute_gap / |exact_optimum|

    Returns None for all fields when candidate or optimum is unavailable.
    """
    if candidate_objective is None or exact_optimum is None:
        return {
            "absolute_gap": None,
            "relative_gap": None,
            "valid": False,
            "note": "Missing candidate or optimum",
        }

    absolute_gap = candidate_objective - exact_optimum
    valid = True

    if absolute_gap < -GAP_TOLERANCE:
        valid = False  # Candidate better than "optimal" — investigate

    if abs(exact_optimum) > GAP_TOLERANCE:
        relative_gap = absolute_gap / abs(exact_optimum)
    else:
        relative_gap = None

    return {
        "absolute_gap": float(absolute_gap),
        "relative_gap": (float(relative_gap) if relative_gap is not None else None),
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# Circuit resource metrics
# ---------------------------------------------------------------------------

def measure_qaoa_circuit_resources(qubo, p: int) -> dict:
    """Measure circuit resource metrics for a QAOA ansatz.

    Uses the actual circuit construction path from solve_qaoa.
    Records qubit count, depth, and gate counts.
    """
    from qgrid.solvers.qaoa import _build_param_circuit

    n_qubits = qubo.n_vars
    qc, gammas, betas = _build_param_circuit(qubo, p, n_qubits)

    # Assign dummy parameters to get a concrete circuit
    param_dict = {}
    for i in range(p):
        param_dict[str(gammas[i])] = 0.1
        param_dict[str(betas[i])] = 0.1

    qc_concrete = qc.assign_parameters(param_dict)

    gate_counts = dict(qc_concrete.count_ops())
    total_gates = qc_concrete.size()
    depth = qc_concrete.depth()

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "gate_counts": gate_counts,
        "total_gates": total_gates,
        "transpilation_config": {
            "note": "No transpilation applied; metrics from ansatze circuit before measurement.",
            "basis_gates": None,
            "backend_name": None,
        },
    }


# ---------------------------------------------------------------------------
# Dependency versions
# ---------------------------------------------------------------------------

def get_dependency_versions() -> dict:
    """Get versions of all relevant dependencies."""
    versions = {}
    for pkg in ["qiskit", "qiskit_aer", "scipy", "numpy", "ortools"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    versions["python"] = f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
    return versions


# ---------------------------------------------------------------------------
# Dataclasses for benchmark configuration and results
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes
    ----------
    instance : JobAssignmentInstance
        The assignment problem instance to solve.
    instance_name : str
        Human-readable identifier for the instance.
    seeds : list of int
        Random seeds for stochastic solvers (QAOA).
    qaoa_depth : int
        QAOA circuit depth (p).
    n_shots : int
        Number of sampling shots for QAOA.
    maxiter : int
        Maximum optimizer iterations for QAOA.
    cp_sat_time_limit : float
        Time limit in seconds for CP-SAT.
    exact_max_combinations : int
        Maximum number of assignments for exact solver (S^J).
        Instances larger than this are skipped with a "too_large" status.
    """
    instance: JobAssignmentInstance
    instance_name: str = "unnamed"
    seeds: List[int] = field(default_factory=lambda: [42])
    qaoa_depth: int = 1
    n_shots: int = 512
    maxiter: int = 50
    cp_sat_time_limit: float = 10.0
    exact_max_combinations: int = 10_000_000


@dataclass
class BenchmarkTrial:
    """Result of running one solver on one trial.

    A trial is one invocation of one solver with one configuration.
    For deterministic solvers (exact, greedy, CP-SAT), this is one per solver.
    For QAOA, this is one per seed.
    """
    method: str
    seed: Optional[int]
    objective: Optional[float]
    status: str
    is_optimal: bool
    runtime_seconds: float
    feasible: bool
    solution: Optional[dict]
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one instance.

    Attributes
    ----------
    instance_definition : dict
        Complete serialized instance definition.
    schema_version : str
        Version of the benchmark schema.
    configuration : dict
        Benchmark configuration parameters.
    dependency_versions : dict
        Versions of relevant Python packages.
    trials : list of BenchmarkTrial
        All individual solver runs.
    aggregate : dict
        Summary statistics across trials.
    timestamps : dict
        Timestamp metadata.
    """
    instance_definition: dict
    schema_version: str
    configuration: dict
    dependency_versions: dict
    trials: List[BenchmarkTrial]
    aggregate: dict
    timestamps: dict


# ---------------------------------------------------------------------------
# Benchmark instances
# ---------------------------------------------------------------------------

def tiny_benchmark_instance() -> JobAssignmentInstance:
    """Small instance suitable for exact enumeration and validation."""
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[3.0, 2.0],
        server_capacities=[5.0, 10.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
        energy_costs={
            (0, 0): 5.0,
            (0, 1): 3.0,
            (1, 0): 2.0,
            (1, 1): 4.0,
        },
    )


def medium_benchmark_instance() -> JobAssignmentInstance:
    """2-job, 2-server instance with full eligibility and small capacity.

    Uses small integer capacities so slack variable count stays low
    (ceil(log2(cap+1)) per server), making QAOA tractable on a laptop.
    """
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[3.0, 2.0],
        server_capacities=[4.0, 4.0],
        eligible_jobs=[
            frozenset({0, 1}),
            frozenset({0, 1}),
        ],
        energy_costs={
            (0, 0): 5.0,
            (0, 1): 3.0,
            (1, 0): 2.0,
            (1, 1): 4.0,
        },
    )


def greedy_fails_instance() -> JobAssignmentInstance:
    """Instance with tight capacity and eligibility restrictions.

    Server 0: capacity 3, eligible for {0, 1}
    Server 1: capacity 3, eligible for {1}

    Job 0: demand 3, cost (0,0) = 1
    Job 1: demand 3, cost (0,1) = 2, (1,1) = 1

    Greedy in index order:
      - Job 0 → eligible servers [0]. Server 0 has cap 3 ≥ 3. Assigns.
        Remaining: server 0 = 0, server 1 = 3.
      - Job 1 → eligible servers [0, 1]. Sorted by cost: server 1 (cost 1),
        server 0 (cost 2). Server 1 has cap 3 ≥ 3. Assigns.
        Result: Feasible! Objective = 1 + 1 = 2.

    This instance tests greedy behavior with tight capacity margins.
    """
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[3.0, 3.0],
        server_capacities=[3.0, 3.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({1})],
        energy_costs={
            (0, 0): 1.0,
            (0, 1): 2.0,
            (1, 1): 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Solver execution wrappers
# ---------------------------------------------------------------------------

def _solution_to_dict(solution) -> Optional[dict]:
    """Convert AssignmentSolution to dict, or None."""
    if solution is None:
        return None
    return {"assignment": dict(solution.assignment)}


def run_exact(
    problem: AssignmentProblem,
    instance: JobAssignmentInstance,
    max_combinations: int,
) -> BenchmarkTrial:
    """Run exact enumeration with size guard."""
    combinations = instance.n_servers ** instance.n_jobs

    if combinations > max_combinations:
        return BenchmarkTrial(
            method="exact",
            seed=None,
            objective=None,
            status="too_large",
            is_optimal=False,
            runtime_seconds=0.0,
            feasible=False,
            solution=None,
            message=(
                f"Instance too large for exact enumeration: "
                f"S^J = {instance.n_servers}^{instance.n_jobs} = "
                f"{combinations} > {max_combinations}"
            ),
            metadata={"combinations": combinations},
        )

    start = time.perf_counter()
    result = solve_exact(problem)
    elapsed = time.perf_counter() - start

    return BenchmarkTrial(
        method="exact",
        seed=None,
        objective=result.objective,
        status=result.status,
        is_optimal=result.is_optimal,
        runtime_seconds=elapsed,
        feasible=result.solution is not None,
        solution=_solution_to_dict(result.solution),
        message=result.message,
        metadata={
            "internal_runtime_seconds": result.runtime_seconds,
            "notes": result.notes,
        },
    )


def run_cpsat(problem: AssignmentProblem, time_limit: float) -> BenchmarkTrial:
    """Run CP-SAT solver."""
    start = time.perf_counter()
    result = solve_cpsat(problem, time_limit=time_limit)
    elapsed = time.perf_counter() - start

    return BenchmarkTrial(
        method="cpsat",
        seed=None,
        objective=result.objective,
        status=result.status,
        is_optimal=result.is_optimal,
        runtime_seconds=elapsed,
        feasible=result.solution is not None,
        solution=_solution_to_dict(result.solution),
        message=result.message,
        metadata={
            "internal_runtime_seconds": result.runtime_seconds,
            "notes": str(result.notes) if result.notes else None,
        },
    )


def run_greedy(problem: AssignmentProblem) -> BenchmarkTrial:
    """Run greedy heuristic."""
    start = time.perf_counter()
    result = solve_greedy(problem)
    elapsed = time.perf_counter() - start

    return BenchmarkTrial(
        method="greedy",
        seed=None,
        objective=result.objective,
        status=result.status,
        is_optimal=False,
        runtime_seconds=elapsed,
        feasible=result.solution is not None,
        solution=_solution_to_dict(result.solution),
        message=result.message,
        metadata={
            "internal_runtime_seconds": result.runtime_seconds,
            "notes": str(result.notes) if result.notes else None,
        },
    )


# ---------------------------------------------------------------------------
# QAOA execution and helpers
# ---------------------------------------------------------------------------

def _get_problem_from_qubo(qubo) -> AssignmentProblem:
    """Build AssignmentProblem from QUBO's instance."""
    return AssignmentProblem(instance=qubo.instance)


def _bitstring_to_vector_local(bitstring: str, n: int):
    """Convert Qiskit bitstring (little-endian) to binary vector."""
    x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)), mode="constant")
    return x[:n]


def _feasible_sample_fraction(
    samples: list, qubo, n_shots: int
) -> Optional[float]:
    """Compute fraction of sampled shots that are feasible solutions."""
    from qgrid.formulation import decode_solution
    from qgrid.problem import AssignmentSolution

    if n_shots == 0 or not samples:
        return None

    total_count = sum(count for _, count in samples)
    if total_count == 0:
        return 0.0

    problem = _get_problem_from_qubo(qubo)
    feasible_count = 0

    for bitstring, count in samples:
        x = _bitstring_to_vector_local(bitstring, qubo.n_vars)
        assignment, slack = decode_solution(x, qubo)
        if not assignment:
            continue
        solution = AssignmentSolution(assignment=assignment)
        report = problem.check_feasibility(solution)
        if report.is_feasible:
            feasible_count += count

    return feasible_count / total_count


def run_qaoa(
    problem: AssignmentProblem,
    instance: JobAssignmentInstance,
    seed: int,
    depth: int,
    n_shots: int,
    maxiter: int,
    exact_obj: Optional[float],
) -> BenchmarkTrial:
    """Run QAOA solver with circuit metrics and gap.

    Preserves the distinction between lowest-energy sample and best
    feasible sample. Uses the original problem feasibility checker.
    """
    qubo = encode_qubo(instance)

    start = time.perf_counter()
    qaoa_result = solve_qaoa(
        problem=problem,
        qubo=qubo,
        p=depth,
        n_shots=n_shots,
        maxiter=maxiter,
        seed=seed,
        init_params=None,
        sampler_type="statevector",
    )
    elapsed = time.perf_counter() - start

    gap = compute_gap(qaoa_result.best_feasible_objective, exact_obj)
    circuit_metrics = measure_qaoa_circuit_resources(qubo, depth)
    feasible_frac = _feasible_sample_fraction(
        qaoa_result.all_samples, qubo, n_shots
    )

    return BenchmarkTrial(
        method="qaoa",
        seed=seed,
        objective=qaoa_result.best_feasible_objective,
        status=qaoa_result.status,
        is_optimal=False,
        runtime_seconds=elapsed,
        feasible=qaoa_result.best_feasible_solution is not None,
        solution=_solution_to_dict(qaoa_result.best_feasible_solution),
        message=qaoa_result.message,
        metadata={
            "qaoa_depth": depth,
            "n_shots": n_shots,
            "sampler_type": qaoa_result.sampler_type,
            "effective_maxiter": qaoa_result.effective_maxiter,
            "best_feasible_energy": qaoa_result.best_feasible_energy,
            "best_energy": qaoa_result.best_energy,
            "best_energy_bitstring": qaoa_result.best_energy_bitstring,
            "feasible_sample_fraction": feasible_frac,
            "n_unique_bitstrings": len(qaoa_result.all_samples),
            "circuit_metrics": circuit_metrics,
            "objective_gap": gap,
        },
    )


def _run_exact_objective(
    problem: AssignmentProblem,
    instance: JobAssignmentInstance,
    max_combinations: int,
) -> Optional[float]:
    """Run exact solver and return just the objective for gap computation."""
    combinations = instance.n_servers ** instance.n_jobs
    if combinations > max_combinations:
        return None
    result = solve_exact(problem)
    return result.objective


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------

def compute_aggregate(
    trials: List[BenchmarkTrial], exact_obj: Optional[float]
) -> dict:
    """Compute aggregate statistics for a set of trials.

    For stochastic methods, reports mean and std of trial-level metrics.
    Preserves raw per-trial values in the returned dict.
    """
    from collections import defaultdict
    import statistics

    aggregate: Dict[str, Any] = {
        "exact_optimum": exact_obj,
        "methods": {},
    }

    by_method: Dict[str, List[BenchmarkTrial]] = defaultdict(list)
    for t in trials:
        by_method[t.method].append(t)

    for method, method_trials in by_method.items():
        method_summary: Dict[str, Any] = {
            "n_trials": len(method_trials),
            "raw_objectives": [t.objective for t in method_trials],
            "raw_runtimes": [t.runtime_seconds for t in method_trials],
            "raw_feasible": [t.feasible for t in method_trials],
            "raw_status": [t.status for t in method_trials],
            "feasible_count": sum(1 for t in method_trials if t.feasible),
            "feasible_rate": (
                sum(1 for t in method_trials if t.feasible) / len(method_trials)
                if method_trials
                else 0.0
            ),
        }

        feasible_objs = [
            t.objective for t in method_trials
            if t.feasible and t.objective is not None
        ]
        if feasible_objs:
            method_summary["best_feasible_objective"] = min(feasible_objs)
        else:
            method_summary["best_feasible_objective"] = None

        all_objs = [
            t.objective for t in method_trials if t.objective is not None
        ]
        if all_objs:
            method_summary["best_objective"] = min(all_objs)
        else:
            method_summary["best_objective"] = None

        if method_summary["best_feasible_objective"] is not None and exact_obj is not None:
            method_summary["best_feasible_gap"] = compute_gap(
                method_summary["best_feasible_objective"], exact_obj
            )
        else:
            method_summary["best_feasible_gap"] = {
                "absolute_gap": None,
                "relative_gap": None,
                "valid": False,
                "note": "No feasible solution or no exact optimum",
            }

        if len(method_trials) > 1:
            method_summary["objective_mean"] = (
                statistics.mean(all_objs) if all_objs else None
            )
            method_summary["objective_std"] = (
                statistics.stdev(all_objs) if len(all_objs) > 1 else 0.0
            )
            method_summary["runtime_mean"] = statistics.mean(
                t.runtime_seconds for t in method_trials
            )
            method_summary["runtime_std"] = (
                statistics.stdev(t.runtime_seconds for t in method_trials)
                if len(method_trials) > 1
                else 0.0
            )
        else:
            method_summary["objective_mean"] = all_objs[0] if all_objs else None
            method_summary["objective_std"] = 0.0
            method_summary["runtime_mean"] = (
                method_trials[0].runtime_seconds if method_trials else 0.0
            )
            method_summary["runtime_std"] = 0.0

        aggregate["methods"][method] = method_summary

    return aggregate


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run all solvers against a single instance.

    Executes exact enumeration, CP-SAT, greedy, and QAOA solvers
    against the same JobAssignmentInstance. Records per-trial results
    and computes aggregate statistics.

    Runtime includes end-to-end wall-clock time for each solver invocation.
    Note: solver-specific overhead (e.g., QAOA circuit construction and
    optimization loop) is included in runtime, making cross-method
    comparison indicative rather than strictly equivalent.
    """
    problem = AssignmentProblem(instance=config.instance)

    now = datetime.now()
    start_timestamp = now.isoformat()

    instance_def = serialize_instance(config.instance, config.instance_name)
    dep_versions = get_dependency_versions()

    # Run exact solver for reference optimum
    exact_result = run_exact(
        problem, config.instance, config.exact_max_combinations
    )
    exact_obj = exact_result.objective

    # Run CP-SAT
    cpsat_result = run_cpsat(problem, config.cp_sat_time_limit)

    # Run Greedy
    greedy_result = run_greedy(problem)

    # Run QAOA for each seed
    qaoa_results = []
    for seed in config.seeds:
        qaoa_result = run_qaoa(
            problem=problem,
            instance=config.instance,
            seed=seed,
            depth=config.qaoa_depth,
            n_shots=config.n_shots,
            maxiter=config.maxiter,
            exact_obj=exact_obj,
        )
        qaoa_results.append(qaoa_result)

    all_trials = [exact_result, cpsat_result, greedy_result] + qaoa_results
    aggregate = compute_aggregate(all_trials, exact_obj)

    result = BenchmarkResult(
        instance_definition=instance_def,
        schema_version=BENCHMARK_SCHEMA_VERSION,
        configuration={
            "seeds": config.seeds,
            "qaoa_depth": config.qaoa_depth,
            "n_shots": config.n_shots,
            "maxiter": config.maxiter,
            "cp_sat_time_limit": config.cp_sat_time_limit,
            "exact_max_combinations": config.exact_max_combinations,
        },
        dependency_versions=dep_versions,
        trials=all_trials,
        aggregate=aggregate,
        timestamps={
            "start": start_timestamp,
            "end": datetime.now().isoformat(),
        },
    )

    return result


def _trial_to_dict(trial: BenchmarkTrial) -> dict:
    """Convert a BenchmarkTrial to a JSON-safe dict."""
    return {
        "method": trial.method,
        "seed": trial.seed,
        "objective": trial.objective,
        "status": trial.status,
        "is_optimal": trial.is_optimal,
        "runtime_seconds": trial.runtime_seconds,
        "feasible": trial.feasible,
        "solution": trial.solution,
        "message": trial.message,
        "metadata": trial.metadata,
    }


def result_to_dict(result: BenchmarkResult) -> dict:
    """Convert a BenchmarkResult to a JSON-safe dict."""
    return {
        "schema_version": result.schema_version,
        "instance_definition": result.instance_definition,
        "configuration": result.configuration,
        "dependency_versions": result.dependency_versions,
        "trials": [_trial_to_dict(t) for t in result.trials],
        "aggregate": result.aggregate,
        "timestamps": result.timestamps,
    }


def save_benchmark(
    result: BenchmarkResult,
    results_dir: Optional[Path] = None,
) -> str:
    """Save benchmark result to timestamped JSON file.

    Returns the path to the saved file.
    Filename includes timestamp and instance name for uniqueness.
    Does not overwrite previous results.
    """
    if results_dir is None:
        results_dir = Path("results/benchmarks")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    instance_name = result.instance_definition.get("name", "unknown")
    filename = f"{timestamp}_{instance_name}_benchmark.json"
    path = results_dir / filename

    with open(path, "w") as f:
        json.dump(result_to_dict(result), f, indent=2, default=str)

    return str(path)


# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

def tiny_benchmark_config() -> BenchmarkConfig:
    """Default config for the tiny benchmark instance."""
    return BenchmarkConfig(
        instance=tiny_benchmark_instance(),
        instance_name="tiny_2job_2server",
        seeds=[42],
        qaoa_depth=1,
        n_shots=512,
        maxiter=50,
        cp_sat_time_limit=5.0,
        exact_max_combinations=10_000_000,
    )