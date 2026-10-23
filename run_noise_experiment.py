"""Full noise-sensitivity experiment for QAOA on Q-Grid assignment instances.

Runs 6 instances x 10 seeds x 4 noise conditions = 240 trials.
Output: results/noise_sensitivity/{timestamp}_noise_experiment.json

NOTE: Noise affects only final Aer sampling. Parameter optimization uses
StatevectorSampler (noiseless) for all conditions. This does NOT study
noisy optimization convergence.
"""
from __future__ import annotations

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from qgrid.formulation import encode_qubo, decode_solution
from qgrid.problem import AssignmentProblem, AssignmentSolution
from qgrid.solvers.qaoa import solve_qaoa, _bitstring_to_vector
from qgrid.solvers.enumeration import solve_exact
from qiskit_aer.noise import NoiseModel, depolarizing_error


def get_instances():
    """Return the six validated benchmark instances with known optima."""
    from qgrid.benchmark import (
        tiny_benchmark_instance, medium_benchmark_instance,
        greedy_fails_instance, asymmetric_3job_3server_instance,
        tight_capacity_3job_2server_instance, cost_asymmetry_2job_2server_instance,
    )
    return [
        ("tiny_2job_2server", 5.0, tiny_benchmark_instance()),
        ("medium_2job_2server", 5.0, medium_benchmark_instance()),
        ("tight_capacity_restricted", 2.0, greedy_fails_instance()),
        ("asymmetric_3job_3server", 6.0, asymmetric_3job_3server_instance()),
        ("tight_capacity_3job_2server", 6.0, tight_capacity_3job_2server_instance()),
        ("cost_asymmetry_2job_2server", 3.0, cost_asymmetry_2job_2server_instance()),
    ]


def deterministic_init_params(p: int, seed: int) -> np.ndarray:
    """Generate deterministic initial parameters matching project convention."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, np.pi / 2, size=2 * p)


def build_noise_model(p_depol: float) -> NoiseModel:
    """Depolarizing noise model covering all transpiled basis gates."""
    basis_gates = ["id", "rz", "sx", "x", "cx", "rzz", "measure"]
    noise = NoiseModel(basis_gates=basis_gates)
    noise.add_all_qubit_quantum_error(depolarizing_error(p_depol, 1), ["rz", "sx"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p_depol, 2), ["rzz", "cx"])
    return noise


def compute_feasible_fraction(all_samples, qubo, problem) -> float:
    """Compute fraction of sampled shots that decode to feasible assignments."""
    total = sum(cnt for _, cnt in all_samples)
    if total == 0:
        return 0.0
    feasible = 0
    for bitstring, count in all_samples:
        x = _bitstring_to_vector(bitstring, qubo.n_vars)
        try:
            assignment, _ = decode_solution(x, qubo)
        except ValueError:
            continue
        if assignment:
            sol = AssignmentSolution(assignment=assignment)
            if problem.check_feasibility(sol).is_feasible:
                feasible += count
    return feasible / total


def _get_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "unknown"


def main():
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p = 1
    n_shots = 512
    maxiter = 50

    noise_conditions = [
        ("noiseless_aer", None, 0.0),
        ("p=0.01", build_noise_model(0.01), 0.01),
        ("p=0.05", build_noise_model(0.05), 0.05),
        ("p=0.10", build_noise_model(0.10), 0.10),
    ]

    basis_gates = ["id", "rz", "sx", "x", "cx", "rzz", "measure"]
    instances = get_instances()

    total_expected = len(instances) * len(seeds) * len(noise_conditions)
    print("=" * 70)
    print("Milestone 6.5 -- Full Noise-Sensitivity Experiment")
    print("=" * 70)
    print(f"Instances: {len(instances)}")
    print(f"Seeds: {len(seeds)} (0-9)")
    print(f"Noise conditions: {len(noise_conditions)}")
    print(f"Expected total trials: {total_expected}")
    print(f"Optimization: StatevectorSampler (noiseless)")
    print(f"Sampling: Aer with explicit transpilation")
    print()

    config = {
        "experiment_type": "noise_sensitivity_full",
        "qaoa_depth": p,
        "optimizer": "COBYLA",
        "maxiter": maxiter,
        "n_shots": n_shots,
        "noise_basis_gates": basis_gates,
        "seeds": seeds,
        "noise_conditions": [
            {"label": label, "noise_p_depol": p_depol, "sampler_type": "aer"}
            for label, _, p_depol in noise_conditions
        ],
        "n_trials_expected": total_expected,
        "note": (
            "Noise affects only final Aer sampling. Parameter optimization "
            "uses StatevectorSampler (noiseless) for all conditions. "
            "This experiment does NOT study noisy optimization convergence."
        ),
        "package_versions": {
            "qiskit": _get_version("qiskit"),
            "qiskit-aer": _get_version("qiskit_aer"),
        },
    }

    all_trial_records = []
    n_attempted = 0
    n_completed = 0
    n_succeeded = 0
    n_failed = 0

    for inst_name, exact_obj, inst in instances:
        print(f"\n--- Instance: {inst_name} (exact optimum: {exact_obj}) ---")
        qubo = encode_qubo(inst)
        problem = AssignmentProblem(instance=inst)

        for seed in seeds:
            init_params = deterministic_init_params(p, seed)

            for label, noise_model, p_depol in noise_conditions:
                n_attempted += 1
                start = time.perf_counter()
                error_detail = None
                result = None

                try:
                    result = solve_qaoa(
                        problem=problem,
                        qubo=qubo,
                        p=p,
                        n_shots=n_shots,
                        maxiter=maxiter,
                        seed=seed,
                        init_params=init_params,
                        sampler_type="aer",
                        aer_noise_model=noise_model,
                    )
                    if result.status == "success":
                        n_succeeded += 1
                        n_completed += 1
                    else:
                        n_completed += 1
                        n_failed += 1
                        error_detail = f"Status: {result.status}"
                except Exception as e:
                    n_failed += 1
                    error_detail = f"{type(e).__name__}: {str(e)[:200]}"
                    result = None

                elapsed = time.perf_counter() - start

                if result is not None:
                    feas_frac = compute_feasible_fraction(
                        result.all_samples, qubo, problem
                    )
                    record = {
                        "instance_name": inst_name,
                        "seed": seed,
                        "noise_condition": label,
                        "noise_p_depol": p_depol,
                        "sampler_type": "aer",
                        "exact_objective": exact_obj,
                        "best_feasible_objective": result.best_feasible_objective,
                        "best_feasible_energy": result.best_feasible_energy,
                        "best_energy": result.best_energy,
                        "best_energy_bitstring": result.best_energy_bitstring,
                        "feasible_sample_fraction": feas_frac,
                        "status": result.status,
                        "is_feasible": result.best_feasible_solution is not None,
                        "runtime_seconds": round(elapsed, 3),
                        "n_shots": result.n_shots,
                        "qaoa_depth": result.qaoa_depth,
                        "effective_maxiter": result.effective_maxiter,
                        "noise_basis_gates": basis_gates,
                        "circuit_gates": result.noise_info.get("circuit_gates", {}),
                        "objective_gap": None,
                        "error_detail": None,
                    }
                    if result.best_feasible_objective is not None:
                        record["objective_gap"] = (
                            result.best_feasible_objective - exact_obj
                        )
                else:
                    record = {
                        "instance_name": inst_name,
                        "seed": seed,
                        "noise_condition": label,
                        "noise_p_depol": p_depol,
                        "sampler_type": "aer",
                        "exact_objective": exact_obj,
                        "best_feasible_objective": None,
                        "best_feasible_energy": None,
                        "best_energy": None,
                        "best_energy_bitstring": None,
                        "feasible_sample_fraction": None,
                        "status": "error",
                        "is_feasible": False,
                        "runtime_seconds": round(elapsed, 3),
                        "n_shots": n_shots,
                        "qaoa_depth": p,
                        "effective_maxiter": maxiter,
                        "noise_basis_gates": basis_gates,
                        "circuit_gates": {},
                        "objective_gap": None,
                        "error_detail": error_detail,
                    }

                all_trial_records.append(record)
                status_marker = "OK" if record["status"] == "success" else "FAIL"
                obj_str = record["best_feasible_objective"]
                if obj_str is None:
                    obj_str = "N/A"
                print(
                    f"  [{status_marker}] {inst_name} seed={seed} "
                    f"noise={label}: obj={obj_str} "
                    f"feas={record.get('feasible_sample_fraction') or 0:.4f} "
                    f"time={elapsed:.2f}s"
                )

    # Compute per-instance-per-condition aggregates
    by_group = defaultdict(list)
    for t in all_trial_records:
        key = (t["instance_name"], t["noise_condition"])
        by_group[key].append(t)

    per_group_aggregate = {}
    for (inst_name, noise_label), group_trials in sorted(by_group.items()):
        n_att = len(group_trials)
        n_succ = sum(1 for t in group_trials if t["status"] == "success")
        n_fail = sum(1 for t in group_trials if t["status"] == "error")
        objs = [t["best_feasible_objective"] for t in group_trials
                if t["best_feasible_objective"] is not None]
        exact = group_trials[0]["exact_objective"]

        feasible_objs = [t["best_feasible_objective"] for t in group_trials
                         if t.get("is_feasible") and t["best_feasible_objective"] is not None]
        matched = sum(1 for t in group_trials
                      if t["best_feasible_objective"] is not None
                      and abs(t["best_feasible_objective"] - exact) <= 1e-6)

        per_group_aggregate[f"{inst_name}__{noise_label}"] = {
            "n_attempted": n_att,
            "n_completed": n_succ,
            "n_failed": n_fail,
            "feasible_rate": (
                sum(1 for t in group_trials if t.get("is_feasible")) / n_att
                if n_att > 0 else 0.0
            ),
            "matched_optimum_rate": matched / n_att if n_att > 0 else 0.0,
            "mean_objective": float(np.mean(objs)) if objs else None,
            "std_objective": float(np.std(objs)) if objs else None,
            "best_feasible_objective": min(objs) if objs else None,
            "n_feasible_objectives": len(feasible_objs),
            "exact_objective": exact,
        }

    config["results"] = {
        "n_attempted": n_attempted,
        "n_completed": n_completed,
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
        "trials": all_trial_records,
        "per_group_aggregate": per_group_aggregate,
    }

    # Save to NEW file (timestamped, won't overwrite existing results)
    output_dir = Path("results/noise_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{timestamp}_noise_experiment.json"
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total attempted: {n_attempted}")
    print(f"Completed: {n_completed}")
    print(f"Succeeded: {n_succeeded}")
    print(f"Failed: {n_failed}")
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()