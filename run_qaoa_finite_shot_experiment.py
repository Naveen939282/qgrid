"""Finite-shot and simulated-noise experiments for QAOA.

Run with:
    python run_qaoa_finite_shot_experiment.py

Saves timestamped JSON results under results/qaoa_experiments/.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from qgrid.formulation import encode_qubo, compute_qubo_energy, decode_solution
from qgrid.problem import JobAssignmentInstance, AssignmentProblem, AssignmentSolution
from qgrid.solvers.qaoa import solve_qaoa, _bitstring_to_vector
from qgrid.solvers.enumeration import solve_exact
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_tiny_instance() -> JobAssignmentInstance:
    """Create the tiny 2-job, 2-server benchmark instance."""
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[3.0, 2.0],
        server_capacities=[5.0, 10.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
        energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
    )


def build_noise_model(p_1q: float = 0.01, p_2q: float = 0.02) -> NoiseModel:
    """Build a depolarizing noise model with gates matching Aer's transpiled basis.

    The noise model must cover ALL gates that appear in the transpiled circuit,
    not just the original QAOA gate names.  When AerSimulator transpiles a QAOA
    circuit using the shared basis set ['id', 'rz', 'sx', 'x', 'cx', 'rzz',
    'measure'], the following decompositions occur:

      - ``rz`` : preserved (cost-layer Z rotations)
      - ``rx`` → decomposed to ``rz-sx-rz`` sequence
      - ``h``  → decomposed to ``rz-sx-rz`` sequence
      - ``rzz``: preserved (cost-layer ZZ interactions)
      - ``cx`` : used for multi-qubit operations

    Therefore noise is attached to:
      - Single-qubit: ``rz``, ``sx`` (covers decomposed rx/rz/h)
      - Two-qubit:     ``rzz``, ``cx``

    Parameters
    ----------
    p_1q : float
        Single-qubit depolarizing probability.
    p_2q : float
        Two-qubit depolarizing probability.

    Returns
    -------
    NoiseModel
        A Qiskit Aer noise model with depolarizing errors on all transpiled gates.
    """
    # Basis gates must match the transpile_basis_gates used in QAOASolver.__call__.
    basis_gates = ["id", "rz", "sx", "x", "cx", "rzz", "measure"]
    noise = NoiseModel(basis_gates=basis_gates)
    noise.add_all_qubit_quantum_error(depolarizing_error(p_1q, 1), ["rz", "sx"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ["rzz", "cx"])
    return noise


def _deterministic_init_params(p: int, seed: int) -> np.ndarray:
    """Generate deterministic initial parameters for reproducibility."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, np.pi / 2, size=2 * p)


def _get_qiskit_version() -> str:
    """Get the installed Qiskit version string."""
    try:
        import qiskit
        return qiskit.__version__
    except (ImportError, AttributeError):
        return "unknown"


def _get_aer_version() -> str:
    """Get the installed Qiskit Aer version string."""
    try:
        import qiskit_aer
        return qiskit_aer.__version__
    except (ImportError, AttributeError):
        return "unknown"


def _feasible_sample_fraction(all_samples, qubo, problem) -> float:
    """Compute fraction of sampled bitstrings that are feasible."""
    total_shots = sum(count for _, count in all_samples)
    if total_shots == 0:
        return 0.0
    total_feasible = 0
    for bitstring, count in all_samples:
        x = _bitstring_to_vector(bitstring, qubo.n_vars)
        try:
            assignment, _ = decode_solution(x, qubo)
        except Exception:
            continue
        if assignment:
            sol = AssignmentSolution(assignment=assignment)
            report = problem.check_feasibility(sol)
            if report.is_feasible:
                total_feasible += count
    return total_feasible / total_shots


def _save_results(data: dict, filename: str) -> None:
    """Save experiment results as timestamped JSON."""
    results_dir = Path("results/qaoa_experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"{timestamp}_{filename}"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")


def run_finite_shot_experiment() -> dict:
    """Run finite-shot experiment comparing shot budgets."""
    inst = build_tiny_instance()
    qubo = encode_qubo(inst)
    problem = AssignmentProblem(instance=inst)

    exact_result = solve_exact(problem)
    exact_objective = exact_result.objective

    config = {
        "experiment_type": "finite_shot",
        "instance": "tiny_2job_2server",
        "qaoa_depth": 2,
        "optimizer": "COBYLA",
        "maxiter": 50,
        "seed": 42,
        "n_vars": qubo.n_vars,
        "n_shots_budgets": [128, 512, 2048],
        "exact_objective": exact_objective,
        "package_versions": {
            "qiskit": _get_qiskit_version(),
            "qiskit-aer": _get_aer_version(),
        },
    }

    results = []
    shot_budgets = [128, 512, 2048]
    init_params = _deterministic_init_params(p=2, seed=42)

    for n_shots in shot_budgets:
        start = time.perf_counter()
        result = solve_qaoa(
            problem, qubo, p=2, n_shots=n_shots, maxiter=50,
            seed=42, init_params=init_params, sampler_type="statevector",
        )
        elapsed = time.perf_counter() - start

        total_shots = sum(count for _, count in result.all_samples)
        feas_frac = _feasible_sample_fraction(result.all_samples, qubo, problem)

        entry = {
            "n_shots_requested": n_shots,
            "actual_shots": total_shots,
            "seed": 42,
            "best_feasible_objective": result.best_feasible_objective,
            "best_feasible_energy": result.best_feasible_energy,
            "best_energy_bitstring": result.best_energy_bitstring,
            "best_energy": result.best_energy,
            "best_feasible_assignment": (
                {str(k): v for k, v in result.best_feasible_solution.assignment.items()}
                if result.best_feasible_solution else None
            ),
            "feasible_sample_fraction": feas_frac,
            "status": result.status,
            "effective_maxiter": result.effective_maxiter,
            "sampler_type": result.sampler_type,
            "runtime_seconds": elapsed,
            "n_unique_bitstrings": len(result.all_samples),
            "is_single_trial": True,
        }
        results.append(entry)

    config["results"] = results
    _save_results(config, "finite_shot_experiment.json")
    return config


def run_noise_experiment() -> dict:
    """Compare noiseless Aer vs noisy Aer with depolarizing noise."""
    inst = build_tiny_instance()
    qubo = encode_qubo(inst)
    problem = AssignmentProblem(instance=inst)

    exact_result = solve_exact(problem)
    exact_objective = exact_result.objective

    p_1q = 0.01
    p_2q = 0.02
    noise_model = build_noise_model(p_1q=p_1q, p_2q=p_2q)

    config = {
        "experiment_type": "noise_comparison",
        "instance": "tiny_2job_2server",
        "qaoa_depth": 2,
        "optimizer": "COBYLA",
        "maxiter": 50,
        "seed": 42,
        "n_vars": qubo.n_vars,
        "n_shots": 512,
        "exact_objective": exact_objective,
        "noise_model": "depolarizing",
        "noise_params": {"p_1q": p_1q, "p_2q": p_2q},
        "noisy_gates": ["rz", "sx", "rzz", "cx"],
        "noise_basis_gates": ["id", "rz", "sx", "x", "cx", "rzz", "measure"],
        "package_versions": {
            "qiskit": _get_qiskit_version(),
            "qiskit-aer": _get_aer_version(),
        },
    }

    init_params = _deterministic_init_params(p=2, seed=42)
    results = {}
    scenarios = [
        ("noiseless_aer", "aer", None),
        ("noisy_aer", "aer", noise_model),
    ]

    for label, sampler_type, noise in scenarios:
        start = time.perf_counter()
        result = solve_qaoa(
            problem, qubo, p=2, n_shots=512, maxiter=50,
            seed=42, init_params=init_params,
            sampler_type=sampler_type, aer_noise_model=noise,
        )
        elapsed = time.perf_counter() - start

        total_shots = sum(count for _, count in result.all_samples)
        feas_frac = _feasible_sample_fraction(result.all_samples, qubo, problem)

        results[label] = {
            "best_feasible_objective": result.best_feasible_objective,
            "best_feasible_energy": result.best_feasible_energy,
            "best_energy_bitstring": result.best_energy_bitstring,
            "best_energy": result.best_energy,
            "best_feasible_assignment": (
                {str(k): v for k, v in result.best_feasible_solution.assignment.items()}
                if result.best_feasible_solution else None
            ),
            "feasible_sample_fraction": feas_frac,
            "status": result.status,
            "sampler_type": result.sampler_type,
            "effective_maxiter": result.effective_maxiter,
            "n_unique_bitstrings": len(result.all_samples),
            "actual_shots": total_shots,
            "runtime_seconds": elapsed,
            "is_single_trial": True,
        }

    config["results"] = results
    _save_results(config, "noise_experiment.json")
    return config


def run_noise_distribution_check() -> dict:
    """Run multiple trials to verify noise model has observable effect."""
    from collections import Counter

    inst = build_tiny_instance()
    qubo = encode_qubo(inst)
    problem = AssignmentProblem(instance=inst)

    noise_model = build_noise_model(p_1q=0.02, p_2q=0.05)
    init_params = _deterministic_init_params(p=2, seed=42)

    config = {
        "experiment_type": "noise_distribution_check",
        "instance": "tiny_2job_2server",
        "qaoa_depth": 2,
        "optimizer": "COBYLA",
        "maxiter": 50,
        "seed": 42,
        "n_vars": qubo.n_vars,
        "n_shots": 512,
        "n_trials": 10,
        "noise_model": "depolarizing",
        "noise_params": {"p_1q": 0.02, "p_2q": 0.05},
        "noisy_gates": ["rz", "sx", "rzz", "cx"],
        "noise_basis_gates": ["id", "rz", "sx", "x", "cx", "rzz", "measure"],
        "package_versions": {
            "qiskit": _get_qiskit_version(),
            "qiskit-aer": _get_aer_version(),
        },
    }

    noiseless_counts_total: Counter = Counter()
    noisy_counts_total: Counter = Counter()

    for trial in range(10):
        seed_trial = 42 + trial

        r_clean = solve_qaoa(
            problem, qubo, p=2, n_shots=512, maxiter=50,
            seed=seed_trial, init_params=init_params,
            sampler_type="aer", aer_noise_model=None,
        )
        for bs, cnt in r_clean.all_samples:
            noiseless_counts_total[bs] += cnt

        r_noisy = solve_qaoa(
            problem, qubo, p=2, n_shots=512, maxiter=50,
            seed=seed_trial, init_params=init_params,
            sampler_type="aer", aer_noise_model=noise_model,
        )
        for bs, cnt in r_noisy.all_samples:
            noisy_counts_total[bs] += cnt

    all_bitstrings = set(noiseless_counts_total.keys()) | set(noisy_counts_total.keys())
    total_clean = sum(noiseless_counts_total.values())
    total_noisy = sum(noisy_counts_total.values())

    dist_clean = {bs: noiseless_counts_total.get(bs, 0) / total_clean for bs in all_bitstrings}
    dist_noisy = {bs: noisy_counts_total.get(bs, 0) / total_noisy for bs in all_bitstrings}
    tvd = sum(abs(dist_clean[bs] - dist_noisy[bs]) for bs in all_bitstrings) / 2.0

    config["results"] = {
        "noiseless_total_counts": dict(noiseless_counts_total),
        "noisy_total_counts": dict(noisy_counts_total),
        "noiseless_distribution": dist_clean,
        "noisy_distribution": dist_noisy,
        "total_variation_distance": tvd,
        "distributions_differ": tvd > 0.0,
        "n_trials": 10,
        "note": "TVD > 0 indicates noise affected the distribution. "
                "TVD can be 0 by chance for small samples.",
    }

    _save_results(config, "noise_distribution_check.json")
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 5: Finite-shot and Simulated-noise Experiments")
    print("=" * 60)

    print("\n[1/3] Finite-shot experiment (shot budgets: 128, 512, 2048)")
    fs_config = run_finite_shot_experiment()
    for r in fs_config["results"]:
        print(f"\n  Shots={r['n_shots_requested']}:")
        print(f"    Best feasible objective: {r['best_feasible_objective']}")
        print(f"    Best energy: {r['best_energy']:.4f} (bitstring: {r['best_energy_bitstring']})")
        print(f"    Feasible fraction: {r['feasible_sample_fraction']:.4f}")
        opt = fs_config['exact_objective']
        if r['best_feasible_objective'] is not None and opt != 0:
            gap = abs(r['best_feasible_objective'] - opt) / abs(opt)
            print(f"    Objective gap: {gap:.4f}")
        elif r['best_feasible_objective'] is not None:
            gap = abs(r['best_feasible_objective'] - opt)
            print(f"    Objective gap (absolute): {gap:.4f}")
        else:
            print(f"    Objective gap: N/A (no feasible solution sampled)")
        print(f"    Runtime: {r['runtime_seconds']:.2f}s")
        print(f"    Sampler: {r['sampler_type']}")

    print("\n[2/3] Noise experiment (noiseless Aer vs noisy Aer)")
    noise_config = run_noise_experiment()
    for label, r in noise_config["results"].items():
        print(f"\n  {label}:")
        print(f"    Best feasible objective: {r['best_feasible_objective']}")
        print(f"    Best energy: {r['best_energy']:.4f}")
        print(f"    Feasible fraction: {r['feasible_sample_fraction']:.4f}")
        print(f"    Sampler: {r['sampler_type']}")

    print("\n[3/3] Noise distribution check (10 trials, TVD measurement)")
    dist_config = run_noise_distribution_check()
    print(f"\n  Total variation distance: {dist_config['results']['total_variation_distance']:.6f}")
    print(f"  Distributions differ: {dist_config['results']['distributions_differ']}")

    print("\n" + "=" * 60)
    print("All experiments complete. Results saved to results/qaoa_experiments/")