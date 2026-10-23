#!/usr/bin/env python
"""Run reproducible multi-seed QAOA benchmark experiments.

This script runs the benchmark harness with a fixed, predetermined list
of seeds on a set of validated small instances. It records every trial
independently and writes timestamped JSON results under
results/benchmarks/multi_seed/.

Usage:
    python multi_seed_benchmark.py

The seeds are fixed and reproducible. This is a reproducibility study,
not a parameter-optimization contest. Tuning (if ever done) must be
kept separate with a different eval seed set.
"""
from __future__ import annotations

from pathlib import Path

from qgrid.benchmark import (
    BenchmarkConfig,
    asymmetric_3job_3server_instance,
    cost_asymmetry_2job_2server_instance,
    greedy_fails_instance,
    medium_benchmark_instance,
    run_benchmark,
    save_benchmark,
    tight_capacity_3job_2server_instance,
    tiny_benchmark_instance,
)


# Fixed, predetermined seed list for reproducible QAOA experiments
MULTI_SEED_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def make_multi_seed_configs() -> list[BenchmarkConfig]:
    """Create benchmark configurations for multi-seed experiments.

    Uses validated instances with known exact optima. Each config has
    10 seeds for QAOA reproducibility study.
    """
    return [
        BenchmarkConfig(
            instance=tiny_benchmark_instance(),
            instance_name="tiny_2job_2server",
            seeds=MULTI_SEED_LIST,
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
        BenchmarkConfig(
            instance=medium_benchmark_instance(),
            instance_name="medium_2job_2server",
            seeds=MULTI_SEED_LIST,
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
        BenchmarkConfig(
            instance=greedy_fails_instance(),
            instance_name="tight_capacity_restricted",
            seeds=MULTI_SEED_LIST,
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
        BenchmarkConfig(
            instance=asymmetric_3job_3server_instance(),
            instance_name="asymmetric_3job_3server",
            seeds=MULTI_SEED_LIST,
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
        BenchmarkConfig(
            instance=tight_capacity_3job_2server_instance(),
            instance_name="tight_capacity_3job_2server",
            seeds=MULTI_SEED_LIST,
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
        BenchmarkConfig(
            instance=cost_asymmetry_2job_2server_instance(),
            instance_name="cost_asymmetry_2job_2server",
            seeds=MULTI_SEED_LIST,
            qaoa_depth=1,
            n_shots=512,
            maxiter=50,
            cp_sat_time_limit=5.0,
            exact_max_combinations=10_000_000,
        ),
    ]


def print_summary(result, config: BenchmarkConfig):
    """Print a small summary table for one benchmark result."""
    d = result.aggregate
    exact = d.get("exact_optimum")

    print(f"\n{'='*70}")
    print(f"Instance: {result.instance_definition['name']}")
    print(f"Exact optimum: {exact}")
    print(f"Seeds: {config.seeds}")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'Status':<12} {'Feasible':<10} {'Obj':<8} "
          f"{'Gap':<12} {'RT(mean)':<10} {'#seeds':<8}")
    print(f"{'-'*70}")

    for method, summary in d["methods"].items():
        obj = summary.get("best_feasible_objective")
        obj_str = str(obj) if obj is not None else "N/A"

        gap = "N/A"
        gap_data = summary.get("best_feasible_gap", {})
        if gap_data.get("absolute_gap") is not None:
            gap = str(gap_data["absolute_gap"])

        feasible_str = "Yes" if summary.get("n_feasible", 0) > 0 else "No"
        status = summary.get("raw_status", ["unknown"])[0]
        runtime = f"{summary.get('runtime_mean', 0):.3f}s"
        n_seeds = summary.get("n_attempted", 1)

        print(f"{method:<12} {status:<12} {feasible_str:<10} {obj_str:<8} "
              f"{gap:<12} {runtime:<10} {n_seeds:<8}")

    # Print per-seed QAOA details
    qaoa_trials = [t for t in result.trials if t.method == "qaoa"]
    if qaoa_trials:
        print(f"\n  QAOA per-seed outcomes ({len(qaoa_trials)} trials):")
        print(f"  {'Seed':<6} {'Status':<12} {'Feasible':<10} {'Obj':<10} "
              f"{'Gap':<12} {'Runtime':<10}")
        print(f"  {'-'*60}")
        for t in qaoa_trials:
            obj_str = str(t.objective) if t.objective is not None else "N/A"
            gap_str = "N/A"
            if t.objective is not None and exact is not None:
                gap_val = t.objective - exact
                gap_str = f"{gap_val:.4f}"
            feasible_str = "Yes" if t.feasible else "No"
            print(f"  {t.seed:<6} {t.status:<12} {feasible_str:<10} "
                  f"{obj_str:<10} {gap_str:<12} {t.runtime_seconds:.3f}s")


def main():
    """Run multi-seed benchmark on all selected instances."""
    results_dir = Path("results/benchmarks/multi_seed")
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = make_multi_seed_configs()
    saved_paths = []

    for config in configs:
        print(f"\nRunning benchmark for instance: {config.instance_name}")
        result = run_benchmark(config)
        path = save_benchmark(result, results_dir=results_dir)
        saved_paths.append(path)
        print_summary(result, config)

    print(f"\n{'='*70}")
    print("Results saved to:")
    for path in saved_paths:
        print(f"  {path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()