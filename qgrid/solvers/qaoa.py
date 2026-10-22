"""
Minimal QAOA prototype for the Q-Grid job-to-server assignment problem.

This module implements a depth-p QAOA solver that:
1. Converts the existing QUBO representation into a cost Hamiltonian.
2. Builds the standard QAOA ansatz circuit (Hadamard init, alternating
   cost and mixer layers).
3. Optimizes expectation values using a classical optimizer (scipy).
4. Samples bitstrings, decodes them, and evaluates feasibility.
5. Returns the best feasible sampled solution and the lowest-energy bitstring.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qgrid.formulation import (
    QUBOProblem,
    compute_qubo_energy,
    decode_solution,
)
from qgrid.problem import AssignmentProblem, AssignmentSolution
from qgrid.solvers.result import SolverResult


# ---------------------------------------------------------------------------
# QAOA Result
# ---------------------------------------------------------------------------

@dataclass
class QAOAResult:
    """
    Result from a QAOA run.

    Parameters
    ----------
    best_feasible_solution : AssignmentSolution or None
        Best feasible solution decoded from sampled bitstrings.
        None if no feasible candidate was sampled.
    best_feasible_objective : float or None
        Objective value of the best feasible solution.
    best_feasible_energy : float or None
        QUBO energy of the best feasible bitstring.
    best_energy_bitstring : str
        Bitstring with the lowest QUBO energy (may be infeasible).
    best_energy : float
        QUBO energy of the lowest-energy bitstring.
    params : np.ndarray
        Optimized QAOA parameters [gamma_1, ..., gamma_p, beta_1, ..., beta_p].
    n_shots : int
        Number of shots used for final sampling.
    qaoa_depth : int
        QAOA depth p used.
    status : str
        "success", "max_iterations", or "error".
    runtime_seconds : float
        Wall-clock time of the QAOA run.
    message : str
        Additional information.
    all_samples : List[Tuple[str, int]]
        All sampled bitstrings with counts (for analysis).
    """

    best_feasible_solution: Optional[AssignmentSolution]
    best_feasible_objective: Optional[float]
    best_feasible_energy: Optional[float]
    best_energy_bitstring: str
    best_energy: float
    params: np.ndarray
    n_shots: int
    qaoa_depth: int
    status: str
    runtime_seconds: float
    message: str = ""
    all_samples: List[Tuple[str, int]] = field(default_factory=list)
    sampler_type: str = "statevector"
    effective_maxiter: int = 0



# ---------------------------------------------------------------------------
# Hamiltonian Construction
# ---------------------------------------------------------------------------

def _compute_linear_z_coeffs(Q: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the coefficients of single-qubit Z terms in the Ising Hamiltonian.

    From the QUBO-to-Ising mapping x_i = (1 - Z_i) / 2:
    - Linear term Q[i,i] * x_i contributes -Q[i,i]/2 to the Z_i coefficient.
    - Cross term Q[i,j] * x_i * x_j contributes -Q[i,j]/4 to each Z_i and Z_j.
    """
    linear_z = np.zeros(n)
    for i in range(n):
        linear_z[i] += -Q[i, i] / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if abs(qij) > 1e-15:
                linear_z[i] += -qij / 4.0
                linear_z[j] += -qij / 4.0
    return linear_z


def qubo_to_hamiltonian(
    qubo: QUBOProblem,
) -> Tuple[object, float, int]:
    """
    Convert the QUBO Q matrix to a cost Hamiltonian (Ising form).

    Uses the mapping: x_i = (1 - Z_i) / 2

    The QUBO energy is:
        E(x) = sum_i Q[i,i] * x_i + sum_{i<j} Q[i,j] * x_i * x_j + offset

    The resulting Ising Hamiltonian is:
        H = constant + sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i * Z_j

    Parameters
    ----------
    qubo : QUBOProblem
        The QUBO problem.

    Returns
    -------
    hamiltonian : SparsePauliOp
        The cost Hamiltonian (without offset).
    offset : float
        Constant offset such that E(x) = <psi|H|psi> + offset.
    n_qubits : int
        Number of qubits.
    """
    from qiskit.quantum_info import Pauli, SparsePauliOp

    Q = qubo.Q
    n = qubo.n_vars
    qubo_offset = qubo.offset
    linear_z = _compute_linear_z_coeffs(Q, n)

    # Compute sum of all J_ij terms for ZZ contributions at |000...0>
    zz_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if abs(qij) > 1e-15:
                zz_sum += qij / 4.0

    # At |000...0>, all Z_i = +1 and all Z_i*Z_j = +1, so:
    # <H> = constant + sum(linear_z) + sum(J_ij) = E(x=0) = offset
    constant = qubo_offset - np.sum(linear_z) - zz_sum

    pauli_strings: List[str] = ["I" * n]
    coeffs: List[float] = [constant]

    for i in range(n):
        if abs(linear_z[i]) > 1e-15:
            p_str = ["I"] * n
            p_str[n - 1 - i] = "Z"
            pauli_strings.append("".join(p_str))
            coeffs.append(float(linear_z[i]))

    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if abs(qij) > 1e-15:
                coeff = qij / 4.0
                p_str = ["I"] * n
                p_str[n - 1 - i] = "Z"
                p_str[n - 1 - j] = "Z"
                pauli_strings.append("".join(p_str))
                coeffs.append(float(coeff))

    paulis = [Pauli(s) for s in pauli_strings]
    hamiltonian = SparsePauliOp(paulis, coeffs=np.array(coeffs, dtype=complex))
    return hamiltonian, constant, n



# ---------------------------------------------------------------------------
# Circuit Construction
# ---------------------------------------------------------------------------

def _build_param_circuit(
    qubo: QUBOProblem,
    p: int,
    n_qubits: int,
):
    """Build a parameterized QAOA ansatz circuit.

    Returns (circuit, list_of_gamma_parameters, list_of_beta_parameters).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    Q = qubo.Q
    linear_z = _compute_linear_z_coeffs(Q, n_qubits)

    gammas = [Parameter(f"gamma_{i}") for i in range(p)]
    betas = [Parameter(f"beta_{i}") for i in range(p)]

    qc = QuantumCircuit(n_qubits, name="qaoa_ansatz")
    qc.h(range(n_qubits))

    for layer in range(p):
        gamma = gammas[layer]

        # Single-qubit Z rotations (cost layer)
        for i in range(n_qubits):
            if abs(linear_z[i]) > 1e-15:
                qc.rz(-2.0 * gamma * float(linear_z[i]), i)

        # ZZ interactions (cost layer)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qij = Q[i, j]
                if abs(qij) > 1e-15:
                    coeff = qij / 4.0
                    qc.rzz(-2.0 * gamma * float(coeff), i, j)

        # Mixer layer
        beta = betas[layer]
        for i in range(n_qubits):
            qc.rx(-2.0 * beta, i)

    return qc, gammas, betas



# Public alias for external use
def build_qaoa_circuit(
    qubo: QUBOProblem,
    p: int,
    n_qubits: int,
):
    """
    Build a parameterized QAOA ansatz circuit (public interface).

    This is a wrapper around _build_param_circuit that returns:
    (QuantumCircuit, gammas, betas)
    """
    return _build_param_circuit(qubo, p, n_qubits)



def _bitstring_to_vector(bitstring: str, n: int) -> np.ndarray:
    """Convert a Qiskit bitstring (little-endian) to a QUBO binary vector."""
    x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)), mode="constant")
    return x[:n]


# ---------------------------------------------------------------------------
# Energy Evaluation
# -----------------------------------------------------------------------

def evaluate_energy_from_counts(
    counts: Dict[str, int],
    qubo: QUBOProblem,
    n_shots: int,
) -> float:
    """Evaluate expected QUBO energy from measurement counts."""
    total_energy = 0.0
    for bitstring, count in counts.items():
        x = _bitstring_to_vector(bitstring, qubo.n_vars)
        energy = compute_qubo_energy(x, qubo)
        total_energy += energy * count
    return total_energy / n_shots



# ---------------------------------------------------------------------------
# QAOA Solver
# ---------------------------------------------------------------------------

def solve_qaoa(
    problem: AssignmentProblem,
    qubo: QUBOProblem,
    p: int = 1,
    n_shots: int = 1024,
    maxiter: int = 50,
    seed: int = 42,
    init_params: Optional[np.ndarray] = None,
    sampler_type: str = "statevector",
    aer_noise_model: Optional["NoiseModel"] = None,
) -> QAOAResult:
    """
    Run QAOA to find the best feasible job-to-server assignment.

    Parameters
    ----------
    problem : AssignmentProblem
        The assignment problem (for feasibility checking).
    qubo : QUBOProblem
        The QUBO encoding of the problem.
    p : int, default 1
        QAOA depth (number of alternating cost/mixer layers).
    n_shots : int, default 1024
        Number of shots for sampling.
    maxiter : int, default 50
        Maximum optimization iterations (mapped to COBYLA's MAXFUN).
        Internally, MAXFUN is set to at least ``n_params + 2`` to meet
        COBYLA's minimum requirement and avoid scipy warnings.
    seed : int, default 42
        Random seed for reproducibility.
    init_params : np.ndarray or None
        Initial parameters [gamma_1, ..., gamma_p, beta_1, ..., beta_p].
        If None, random initialization in [0, pi/2] is used.
    sampler_type : str, default "statevector"
        Sampler type for final measurement sampling: ``"statevector"``
        uses :class:`~qiskit.primitives.StatevectorSampler` (noiseless),
        ``"aer"`` uses :class:`~qiskit_aer.AerSimulator` (supports
        finite-shot and noise simulation).
    aer_noise_model : NoiseModel or None
        Optional Qiskit Aer noise model to use when ``sampler_type="aer"``.
        If None, Aer runs noiselessly. Only used when
        ``sampler_type="aer"``.

    Returns
    -------
    QAOAResult
        QAOA result with best feasible solution, best energy, etc.
    """
    from scipy.optimize import minimize
    from qiskit.primitives import StatevectorSampler

    np.random.seed(seed)
    start_time = time.perf_counter()

    # Build Hamiltonian (for validation)
    hamiltonian, ham_offset, n_qubits = qubo_to_hamiltonian(qubo)

    # Build parameterized circuit
    qc_template, gammas, betas = _build_param_circuit(qubo, p, n_qubits)
    n_params = 2 * p

    if init_params is None:
        init_params = np.random.uniform(0, np.pi / 2, size=n_params)

    def objective(params: np.ndarray) -> float:
        """Compute expected QUBO energy for given parameters by sampling."""
        gamma_vals = params[:p]
        beta_vals = params[p:]
        param_dict = {}
        for i in range(p):
            param_dict[str(gammas[i])] = float(gamma_vals[i])
            param_dict[str(betas[i])] = float(beta_vals[i])
        qc_meas = qc_template.assign_parameters(param_dict)
        qc_meas.measure_all()
        sampler = StatevectorSampler(seed=seed)
        pub = (qc_meas, [])
        result = sampler.run([pub], shots=n_shots).result()
        counts = result[0].data.meas.get_counts()
        return evaluate_energy_from_counts(counts, qubo, n_shots)

    # Optimize
    # COBYLA requires MAXFUN >= n_params + 2; if maxiter is smaller,
    # use the minimum required to avoid a scipy warning.
    n_params_opt = len(init_params)
    cobyla_maxfun = max(maxiter, n_params_opt + 2)

    result_opt = minimize(
        objective,
        init_params,
        method="COBYLA",
        options={"maxiter": cobyla_maxfun, "disp": False},
    )

    best_params = result_opt.x
    status = "success" if result_opt.success else "max_iterations"


    # Final sampling with optimized parameters
    param_dict = {}
    for i in range(p):
        param_dict[str(gammas[i])] = float(best_params[i])
        param_dict[str(betas[i])] = float(best_params[p + i])

    qc_final = qc_template.assign_parameters(param_dict)
    qc_final.measure_all()

    if sampler_type == "aer":
        from qiskit_aer import AerSimulator
        sim = AerSimulator(noise_model=aer_noise_model)
        job = sim.run(qc_final, shots=n_shots, seed_simulator=seed, seed_transpiler=seed)
        counts = job.result().get_counts()
    else:
        sampler = StatevectorSampler(seed=seed)
        pub = (qc_final, [])
        final_result = sampler.run([pub], shots=n_shots).result()
        counts = final_result[0].data.meas.get_counts()
    all_samples = sorted(counts.items(), key=lambda x: -x[1])

    # Find the lowest-energy bitstring (may be infeasible)
    best_energy_bitstring = ""
    best_energy = float("inf")
    for bitstring in counts:
        x = _bitstring_to_vector(bitstring, qubo.n_vars)
        energy = compute_qubo_energy(x, qubo)
        if energy < best_energy:
            best_energy = energy
            best_energy_bitstring = bitstring


    # Find the best FEASIBLE solution among sampled bitstrings
    best_feasible_solution = None
    best_feasible_objective = None
    best_feasible_energy = None

    for bitstring in counts:
        x = _bitstring_to_vector(bitstring, qubo.n_vars)
        assignment, slack = decode_solution(x, qubo)

        # Skip bitstrings with no assignments (empty dict)
        if not assignment:
            continue

        solution = AssignmentSolution(assignment=assignment)
        report = problem.check_feasibility(solution)
        if report.is_feasible:
            objective_val = problem.evaluate(solution)
            energy = compute_qubo_energy(x, qubo)
            if (best_feasible_objective is None or
                objective_val < best_feasible_objective):
                best_feasible_solution = solution
                best_feasible_objective = objective_val
                best_feasible_energy = energy


    elapsed = time.perf_counter() - start_time
    msg = (
        f"QAOA (p={p}) completed with status '{status}'. "
        f"{len(counts)} unique bitstrings sampled over {n_shots} shots. "
        f"Best energy: {best_energy:.6f}."
    )

    return QAOAResult(
        best_feasible_solution=best_feasible_solution,
        best_feasible_objective=best_feasible_objective,
        best_feasible_energy=best_feasible_energy,
        best_energy_bitstring=best_energy_bitstring,
        best_energy=best_energy,
        params=best_params,
        n_shots=n_shots,
        qaoa_depth=p,
        status=status,
        runtime_seconds=elapsed,
        message=msg,
        all_samples=all_samples,
        sampler_type=sampler_type,
        effective_maxiter=cobyla_maxfun,
    )


# ---------------------------------------------------------------------------
# Convenience: convert QAOAResult to SolverResult
# ---------------------------------------------------------------------------

def qaoa_result_to_solver_result(
    qaoa_result: QAOAResult,
    message: str = "",
) -> SolverResult:
    """
    Convert a QAOAResult to the standard SolverResult format.

    QAOA is heuristic and never provably optimal, so is_optimal is always False.
    """
    combined_msg = f"{qaoa_result.message} {message}".strip()
    notes = (
        f"QAOA p={qaoa_result.qaoa_depth}, "
        f"sampler={qaoa_result.sampler_type}, "
        f"n_shots={qaoa_result.n_shots}, "
        f"effective_maxiter={qaoa_result.effective_maxiter}, "
        f"runtime={qaoa_result.runtime_seconds:.3f}s"
    )
    return SolverResult(
        solution=qaoa_result.best_feasible_solution,
        objective=qaoa_result.best_feasible_objective,
        status=qaoa_result.status,
        runtime_seconds=qaoa_result.runtime_seconds,
        message=combined_msg,
        is_optimal=False,
        notes=notes,
    )
