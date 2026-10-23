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
from datetime import datetime, timezone
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
    Also records QUBO formulation resource counts (n_qubo_variables,
    n_slack_bits) for sizing context.
    """
    qubo = encode_qubo(instance)
    n_qubo_vars = qubo.n_vars
    n_assignment_bits = instance.n_jobs * instance.n_servers
    n_slack_bits = n_qubo_vars - n_assignment_bits

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
        "qubo_resources": {
            "n_qubo_variables": n_qubo_vars,
            "n_assignment_bits": n_assignment_bits,
            "n_slack_bits": n_slack_bits,
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


def asymmetric_3job_3server_instance() -> JobAssignmentInstance:
    """3-job, 3-server instance with asymmetric eligibility and costs.

    Server 0: capacity 5, eligible for all jobs; costs 5, 4, 6
    Server 1: capacity 6, eligible for jobs {0, 1}; costs 3, 2
    Server 2: capacity 7, eligible for all jobs; costs 7, 8, 1

    Exact optimum: 6.0 (job 0 → server 1, job 1 → server 1, job 2 → server 2)
    Capacity check: 2+3=5 ≤ 6; 1 ≤ 7. Total demand 6 ≤ total capacity 18.
    """
    return JobAssignmentInstance(
        n_jobs=3,
        n_servers=3,
        job_demands=[2.0, 3.0, 1.0],
        server_capacities=[5.0, 6.0, 7.0],
        eligible_jobs=[
            frozenset({0, 1, 2}),
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ],
        energy_costs={
            (0, 0): 5.0, (0, 1): 4.0, (0, 2): 6.0,
            (1, 0): 3.0, (1, 1): 2.0,
            (2, 0): 7.0, (2, 1): 8.0, (2, 2): 1.0,
        },
    )


def tight_capacity_3job_2server_instance() -> JobAssignmentInstance:
    """3-job, 2-server instance with tight capacities.

    Each server has capacity 5; total demand = 7 (fits exactly).
    Jobs: demand 3, 2, 2; all eligible on both servers.

    Exact optimum: 6.0 (job 0 → server 1, job 1 → server 0, job 2 → server 1)
    Capacity check: server 0: 2 ≤ 5; server 1: 3+2=5 ≤ 5.
    """
    return JobAssignmentInstance(
        n_jobs=3,
        n_servers=2,
        job_demands=[3.0, 2.0, 2.0],
        server_capacities=[5.0, 5.0],
        eligible_jobs=[frozenset({0, 1, 2}), frozenset({0, 1, 2})],
        energy_costs={
            (0, 0): 4.0, (0, 1): 3.0, (0, 2): 5.0,
            (1, 0): 2.0, (1, 1): 5.0, (1, 2): 1.0,
        },
    )


def cost_asymmetry_2job_2server_instance() -> JobAssignmentInstance:
    """2-job, 2-server instance with extreme cost asymmetry.

    Job 0: demand 4, only fits on server 0 (cap 4) or server 1 (cap 5)
    Job 1: demand 1, fits on either server

    Exact optimum: 3.0 (job 0 → server 0: cost 1, job 1 → server 1: cost 2)
    """
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[4.0, 1.0],
        server_capacities=[4.0, 5.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
        energy_costs={
            (0, 0): 1.0, (0, 1): 8.0,
            (1, 0): 7.0, (1, 1): 2.0,
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
        try:
            assignment, slack = decode_solution(x, qubo)
        except ValueError:
            # Invalid encoding: duplicate job assignments — treat as infeasible
            continue
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

# Default numerical tolerance for matching an exact optimum.
OPTIMUM_MATCH_TOL = 1e-6


def _safe_mean(values: List[float]) -> Optional[float]:
    """Population mean, or None if empty."""
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_stdev(values: List[float]) -> Optional[float]:
    """Population standard deviation (divides by N), or None if < 2 values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return float(variance ** 0.5)


def _safe_median(values: List[float]) -> Optional[float]:
    """Median, or None if empty."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def compute_aggregate(
    trials: List[BenchmarkTrial], exact_obj: Optional[float]
) -> dict:
    """Compute aggregate statistics for a set of trials.

    For stochastic methods, reports mean, median, std, and extreme values
    of trial-level metrics. Preserves raw per-trial values in the returned dict.

    Denominators:
        - ``n_attempted``      — all trials for this method (denominator for
          feasible_rate, matched_optimum_rate, etc.)
        - ``n_feasible``        — feasible trials with non-null original objective
          (denominator for feasible-only objective statistics)

    Objective statistics (best/mean/median/worst) are computed over feasible
    trials only. Missing objectives are never treated as zero.

    Standard deviations use population std (divides by N, not N-1). When
    fewer than 2 trials are present, std is reported as ``null``.

    A QAOA trial that happens to match the exact optimum is **not** counted
    as ``proven_optimal_count`` — it only increments
    ``matched_optimum_count``. ``proven_optimal_count`` is reserved for
    solvers that assert optimality (e.g. exact, CP-SAT with optimal status).

    Failed trials (status ``"error"`` or ``"unknown"``) are counted in
    ``n_attempted`` and ``n_failed`` but do not contribute to feasible
    objective statistics.  All trial records are preserved in raw fields.

    Parameters
    ----------
    trials : list of BenchmarkTrial
        All trials for one instance (one per deterministic solver, one
        per seed for QAOA).
    exact_obj : float or None
        Verified exact optimum (reference for gap and optimum matching).

    Returns
    -------
    dict
        ``"exact_optimum"`` and ``"methods"`` mapping method name to a
        summary dict with the fields documented above.
    """
    from collections import defaultdict

    _FAILED_STATUSES = {"error", "unknown"}

    aggregate: Dict[str, Any] = {
        "exact_optimum": exact_obj,
        "methods": {},
    }

    by_method: Dict[str, List[BenchmarkTrial]] = defaultdict(list)
    for t in trials:
        by_method[t.method].append(t)

    for method, method_trials in by_method.items():
        n_attempted = len(method_trials)

        n_completed = sum(
            1 for t in method_trials if t.status not in _FAILED_STATUSES
        )
        n_failed = n_attempted - n_completed

        n_feasible = sum(
            1 for t in method_trials
            if t.feasible and t.objective is not None
        )
        n_feasible_attempted = n_attempted

        feasible_rate = (
            n_feasible / n_feasible_attempted
            if n_feasible_attempted
            else 0.0
        )

        proven_optimal = sum(1 for t in method_trials if t.is_optimal)

        matched_optimum = 0
        if exact_obj is not None:
            for t in method_trials:
                if t.feasible and t.objective is not None:
                    if abs(t.objective - exact_obj) <= OPTIMUM_MATCH_TOL:
                        matched_optimum += 1

        matched_optimum_rate = (
            matched_optimum / n_feasible_attempted
            if n_feasible_attempted
            else 0.0
        )

        raw_objectives = [t.objective for t in method_trials]
        all_objs = [
            t.objective for t in method_trials if t.objective is not None
        ]
        feasible_objs = [
            t.objective for t in method_trials
            if t.feasible and t.objective is not None
        ]

        method_summary: Dict[str, Any] = {
            "n_attempted": n_attempted,
            "n_completed": n_completed,
            "n_failed": n_failed,
            "n_feasible": n_feasible,
            "n_feasible_attempted": n_feasible_attempted,
            "feasible_rate": feasible_rate,
            "matched_optimum_count": matched_optimum,
            "matched_optimum_rate": matched_optimum_rate,
            "proven_optimal_count": proven_optimal,
            "raw_objectives": raw_objectives,
            "raw_runtimes": [t.runtime_seconds for t in method_trials],
            "raw_feasible": [t.feasible for t in method_trials],
            "raw_status": [t.status for t in method_trials],
            "raw_seeds": [t.seed for t in method_trials],
            "best_feasible_objective": min(feasible_objs) if feasible_objs else None,
            "worst_feasible_objective": max(feasible_objs) if feasible_objs else None,
            "mean_feasible_objective": _safe_mean(feasible_objs),
            "median_feasible_objective": _safe_median(feasible_objs),
            "std_feasible_objective": _safe_stdev(feasible_objs),
            "n_feasible_objectives": len(feasible_objs),
            "best_objective": min(all_objs) if all_objs else None,
            "worst_objective": max(all_objs) if all_objs else None,
            "mean_objective": _safe_mean(all_objs),
            "median_objective": _safe_median(all_objs),
            "std_objective": _safe_stdev(all_objs),
            "n_objectives": len(all_objs),
            "runtime_mean": _safe_mean([t.runtime_seconds for t in method_trials]),
            "runtime_median": _safe_median(
                [t.runtime_seconds for t in method_trials]
            ),
            "runtime_std": _safe_stdev(
                [t.runtime_seconds for t in method_trials]
            ),
        }

        if (
            method_summary["best_feasible_objective"] is not None
            and exact_obj is not None
        ):
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

        method_summary["n_trials"] = n_attempted
        method_summary["feasible_count"] = n_feasible

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
    utc_start = datetime.now(timezone.utc).isoformat()

    instance_def = serialize_instance(config.instance, config.instance_name)
    dep_versions = get_dependency_versions()

    # Run exact solver for reference optimum
    exact_result = run_exact(
        problem, config.instance, config.exact_max_combinations
    )
    exact_obj = exact_result.objective

    # Record verification status of the exact optimum
    exact_verified = (
        exact_obj is not None
        and exact_result.status == "optimal"
    )

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
            "exact_optimum_verified": exact_verified,
            "exact_optimum_status": exact_result.status,
        },
        dependency_versions=dep_versions,
        trials=all_trials,
        aggregate=aggregate,
        timestamps={
            "start": start_timestamp,
            "end": datetime.now().isoformat(),
            "utc_start": utc_start,
            "utc_end": datetime.now(timezone.utc).isoformat(),
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