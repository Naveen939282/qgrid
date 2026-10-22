#!/usr/bin/env python
"""Run a small benchmark suite comparing all solvers against the same instances.

This script runs the benchmark harness on a small set of hand-defined
instances and saves timestamped JSON results under results/benchmarks/.

Usage:
    python run_benchmark.py

The instances used here are saved as benchmark data, not for parameter
tuning. See README.md for details.
"""
from __future__ import annotations

from pathlib import Path

from qgrid.benchmark import (
    BenchmarkConfig,
    medium_benchmark_instance,
    greedy_fails_instance,
    run_benchmark,
    save_benchmark,
    result_to_dict,
    tiny_benchmark_instance,
)


def make_configs() -> list[BenchmarkConfig]:
    """Create a list of benchmark configurations for all test instances."""
    return [
        BenchmarkConfig(
            instance=tiny_benchmark_instance(),
            instance_name="tiny_2job_2server",
            seeds=[42],
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
        BenchmarkConfig(
            instance=medium_benchmark_instance(),
            instance_name="medium_2job_2server",
            seeds=[42],
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
        BenchmarkConfig(
            instance=greedy_fails_instance(),
            instance_name="tight_capacity_restricted",
            seeds=[42],
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
    ]


def print_summary(result, config: BenchmarkConfig):
    """Print a small summary table for a benchmark result."""
    from qgrid.benchmark import result_to_dict

    d = result_to_dict(result)
    print(f"\n{'='*60}")
    print(f"Instance: {d['instance_definition']['name']}")
    print(f"Exact optimum: {d['aggregate']['exact_optimum']}")
    print(f"{'='*60}")
    print(f"{'Method':<10} {'Status':<12} {'Feasible':<10} {'Objective':<12} {'Gap':<15} {'Runtime':<10}")
    print(f"{'-'*60}")

    for trial in d["trials"]:
        obj = trial["objective"] if trial["objective"] is not None else "N/A"
        gap = "N/A"
        if "objective_gap" in trial.get("metadata", {}):
            gap_data = trial["metadata"]["objective_gap"]
            if gap_data.get("absolute_gap") is not None:
                gap = str(gap_data["absolute_gap"])

        feasible = "Yes" if trial["feasible"] else "No"
        runtime = f"{trial['runtime_seconds']:.3f}s"
        gap_str = f"{gap:>15}" if gap != "N/A" else "N/A".rjust(15)
        print(f"{trial['method']:<10} {trial['status']:<12} {feasible:<10} {str(obj):<12} {gap_str} {runtime:<10}")

    # QAOA circuit metrics
    qaoa_trials = [t for t in d["trials"] if t["method"] == "qaoa"]
    if qaoa_trials:
        q = qaoa_trials[0]
        if "circuit_metrics" in q.get("metadata", {}):
            cm = q["metadata"]["circuit_metrics"]
            print(f"\n  QAOA circuit metrics:")
            print(f"    Qubits: {cm['n_qubits']}")
            print(f"    Depth: {cm['depth']}")
            print(f"    Total gates: {cm['total_gates']}")
            print(f"    Gate counts: {cm['gate_counts']}")


def main():
    """Run benchmark on all test instances and save results."""
    results_dir = Path("results/benchmarks")
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = make_configs()
    saved_paths = []

    for config in configs:
        print(f"\nRunning benchmark for instance: {config.instance_name}")
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=results_dir)
        saved_paths.append(path)
        print_summary(result, config)

    print(f"\n{'='*60}")
    print(f"Results saved to:")
    for path in saved_paths:
        print(f"  {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()