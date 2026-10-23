"""Minimal smoke test for noise-sensitivity experiment (NOT a full run).
Runs exactly 4 trials: 2 seeds x 2 noise conditions (noiseless + p=0.10).
Saves to a NEW uniquely-named file to avoid overwriting existing results.
"""
from __future__ import annotations

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from qgrid.formulation import encode_qubo, decode_solution
from qgrid.problem import JobAssignmentInstance, AssignmentProblem, AssignmentSolution
from qgrid.solvers.qaoa import solve_qaoa, _bitstring_to_vector
from qgrid.solvers.enumeration import solve_exact
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_tiny_instance() -> JobAssignmentInstance:
    return JobAssignmentInstance(
        n_jobs=2, n_servers=2, job_demands=[3.0, 2.0],
        server_capacities=[5.0, 10.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
        energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
    )


def deterministic_init_params(p: int, seed: int) -> np.ndarray:
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
    inst = build_tiny_instance()
    qubo = encode_qubo(inst)
    problem = AssignmentProblem(instance=inst)
    exact = solve_exact(problem)
    exact_obj = exact.objective

    p = 2
    n_shots = 512
    maxiter = 50
    basis_gates = ["id", "rz", "sx", "x", "cx", "rzz", "measure"]
    seeds = [0, 1]
    noise_conditions = [
        ("noiseless", None, 0.0),
        ("p=0.10", build_noise_model(0.10), 0.10),
    ]

    config = {
        "experiment_type": "smoke_test_noise_sensitivity",
        "instance": "tiny_2job_2server",
        "qaoa_depth": p,
        "optimizer": "COBYLA",
        "maxiter": maxiter,
        "n_shots": n_shots,
        "exact_objective": exact_obj,
        "noise_basis_gates": basis_gates,
        "seeds": seeds,
        "noise_levels": [0.0, 0.10],
        "n_trials_expected": len(seeds) * len(noise_conditions),
        "package_versions": {
            "qiskit": _get_version("qiskit"),
            "qiskit-aer": _get_version("qiskit_aer"),
        },
    }

    trial_records = []
    for seed in seeds:
        init_params = deterministic_init_params(p, seed)
        for label, noise_model, p_depol in noise_conditions:
            start = time.perf_counter()
            result = solve_qaoa(
                problem, qubo, p=p, n_shots=n_shots, maxiter=maxiter,
                seed=seed, init_params=init_params,
                sampler_type="aer", aer_noise_model=noise_model,
            )
            elapsed = time.perf_counter() - start

            feas_frac = compute_feasible_fraction(result.all_samples, qubo, problem)

            record = {
                "seed": seed,
                "noise_condition": label,
                "noise_p_depol": p_depol,
                "sampler_type": result.sampler_type,
                "exact_objective": exact_obj,
                "best_feasible_objective": result.best_feasible_objective,
                "best_feasible_energy": result.best_feasible_energy,
                "best_energy": result.best_energy,
                "best_energy_bitstring": result.best_energy_bitstring,
                "feasible_sample_fraction": feas_frac,
                "status": result.status,
                "runtime_seconds": round(elapsed, 3),
                "n_shots": result.n_shots,
                "qaoa_depth": result.qaoa_depth,
                "effective_maxiter": result.effective_maxiter,
                "noise_basis_gates": basis_gates,
                "circuit_gates": result.noise_info.get("circuit_gates", {}),
                "objective_gap": None,
            }
            if record["best_feasible_objective"] is not None:
                record["objective_gap"] = record["best_feasible_objective"] - exact_obj

            trial_records.append(record)
            print(
                f"  seed={seed} noise={label}: "
                f"obj={record['best_feasible_objective']} "
                f"feas_frac={feas_frac:.4f} "
                f"status={record['status']} "
                f"time={elapsed:.2f}s"
            )

    config["results"] = {
        "trials": trial_records,
        "n_attempted": len(trial_records),
        "n_succeeded": sum(1 for r in trial_records if r["status"] == "success"),
        "n_failed": sum(1 for r in trial_records if r["status"] == "error"),
    }

    results_dir = Path("results/smoke_tests")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"{timestamp}_smoke_noise_test.json"
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\nResults saved to: {filepath}")
    print(f"Exact objective: {exact_obj}")
    print(f"Trials completed: {len(trial_records)}")


if __name__ == "__main__":
    main()