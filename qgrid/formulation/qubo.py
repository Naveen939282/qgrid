"""
QUBO encoding for the job-to-server assignment problem.

This module translates the constrained assignment problem into a QUBO
(Quadratic Unconstrained Binary Optimization) formulation suitable for
QAOA and quantum annealing.

QUBO Convention
---------------
The QUBO energy is defined as:

    E(x) = x^T Q x

where x is a binary vector and Q is an upper-triangular matrix
(Q[i,j] = 0 for i > j). The energy also has a constant offset:

    E_total(x) = x^T Q x + offset

Variable Ordering
------------------
Binary variables are ordered as:

    [ x_{0,0}, x_{0,1}, ..., x_{0,J-1},   # server 0 assignments
      x_{1,0}, x_{1,1}, ..., x_{1,J-1},   # server 1 assignments
      ...
      x_{S-1,0}, ..., x_{S-1,J-1},         # last server assignments
      y_{0,0}, ..., y_{0,K_0-1},          # slack bits for server 0
      y_{1,0}, ...,                        # slack bits for server 1
      ...
      y_{S-1,0}, ..., y_{S-1,K_{S-1}-1}    # slack bits for last server
    ]

Where:
  - x_{i,j} = 1 means job j is assigned to server i (only for eligible pairs,
    ineligible pairs are excluded from the QUBO)
  - y_{i,k} are binary slack variables for server i's capacity constraint

Slack Variables
---------------
For each server i, we introduce K_i = ceil(log2(C_i + 1)) binary slack
variables y_{i,0}, ..., y_{i,K_i-1} to represent the unused capacity:

    slack_i = C_i - sum_j w_j * x_{ij}
            = sum_{k=0}^{K_i-1} 2^k * y_{i,k}

The capacity constraint becomes:
    sum_j w_j * x_{ij} + sum_k 2^k * y_{i,k} = C_i

This is enforced via the penalty:
    lambda_2 * (sum_j w_j * x_{ij} + sum_k 2^k * y_{i,k} - C_i)^2

Note: We use unary slack variables (one bit per unit of capacity) when
capacities are small integers, and binary-encoded slack variables
for larger capacities. The default uses binary encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qgrid.problem import JobAssignmentInstance
from .penalty import compute_penalty_weights


@dataclass
class QUBOProblem:
    """
    A QUBO problem encoding of a job-to-server assignment instance.

    Attributes
    ----------
    instance : JobAssignmentInstance
        The original problem instance.
    Q : np.ndarray
        Upper-triangular QUBO matrix of shape (n_vars, n_vars).
    offset : float
        Constant offset added to the QUBO energy.
    n_assignment_vars : int
        Number of assignment binary variables (x_{i,j}).
    n_slack_vars : int
        Number of slack binary variables for capacity constraints.
    n_vars : int
        Total number of QUBO variables.
    lambda_1 : float
        Penalty weight for assignment constraint.
    lambda_2 : float
        Penalty weight for capacity constraint.
    var_metadata : list of dict
        Metadata for each variable: {"type": "assignment"|"slack",
        "server": int, "job": int (for assignment), "server": int (for slack)}.
    scaled_demands : dict of int -> int
        Integer-scaled job demands used in the QUBO (for capacity terms).
    scaled_capacities : dict of int -> int
        Integer-scaled server capacities.
    slack_scale : int
        Scale factor for converting integer demands/capacities back.
    """

    instance: JobAssignmentInstance
    Q: np.ndarray
    offset: float
    n_assignment_vars: int
    n_slack_vars: int
    n_vars: int
    lambda_1: float
    lambda_2: float
    var_metadata: List[dict] = field(default_factory=list)
    scaled_demands: Dict[int, int] = field(default_factory=dict)
    scaled_capacities: Dict[int, int] = field(default_factory=dict)
    slack_scale: int = 1000

    @property
    def assignment_vars(self) -> List[int]:
        """Indices of assignment variables in the QUBO."""
        return [i for i, meta in enumerate(self.var_metadata)
                if meta["type"] == "assignment"]

    @property
    def slack_vars(self) -> List[int]:
        """Indices of slack variables in the QUBO."""
        return [i for i, meta in enumerate(self.var_metadata)
                if meta["type"] == "slack"]



_DEMAND_SCALE = 1000
# Also provide _SCALE as an alias for backwards compatibility
_SCALE = _DEMAND_SCALE


def _to_scaled_int(value: float) -> int:
    """Convert a float demand/capacity/cost to a scaled integer."""
    return int(round(value * _SCALE))


def _determine_scale(instance: JobAssignmentInstance) -> int:
    """
    Determine the appropriate scale factor based on whether
    demands and capacities are integer-valued.

    Returns 1 if all values are integers, otherwise returns _SCALE.
    """
    all_values = list(instance.job_demands) + list(instance.server_capacities)
    all_values += list(instance.energy_costs.values())
    for v in all_values:
        if abs(v - round(v)) > 1e-9:
            return _SCALE
    return 1


def _ceil_log2(n: int) -> int:
    """Compute ceil(log2(n)) for n > 0. Returns 0 if n <= 1."""
    if n <= 1:
        return 0
    k = 0
    while (1 << k) < n:
        k += 1
    return k


def encode_qubo(
    instance: JobAssignmentInstance,
    lambda_1: float | None = None,
    lambda_2: float | None = None,
    alpha: float = 1.0,
    use_unary_slack: bool = False,
) -> QUBOProblem:
    """Encode a job-server assignment instance as a QUBO problem."""
    n_jobs = instance.n_jobs
    n_servers = instance.n_servers

    # Step 1: Determine penalty weights
    if lambda_1 is None or lambda_2 is None:
        l1, l2 = compute_penalty_weights(instance, lambda_1, lambda_2, alpha)
        lambda_1 = l1
        lambda_2 = l2
    lambda_1 = float(lambda_1)
    lambda_2 = float(lambda_2)

    # Step 2: Scale demands and capacities to integers
    # Use adaptive scale: 1 for integer values, 1000 for floats
    scale = _determine_scale(instance)
    scaled_demands: Dict[int, int] = {
        j: int(round(instance.job_demands[j] * scale))
        for j in range(n_jobs)
    }
    scaled_capacities: Dict[int, int] = {
        s: int(round(instance.server_capacities[s] * scale))
        for s in range(n_servers)
    }

    # Step 3: Build variable mapping
    var_metadata: List[dict] = []
    assignment_var_indices: Dict[Tuple[int, int], int] = {}

    for s in range(n_servers):
        for j in range(n_jobs):
            if j in instance.eligible_jobs[s]:
                idx = len(var_metadata)
                var_metadata.append({"type": "assignment", "server": s, "job": j})
                assignment_var_indices[(s, j)] = idx

    n_assignment_vars = len(var_metadata)

    # Slack variables for capacity constraints
    slack_var_indices: Dict[int, List[int]] = {}
    slack_weights: Dict[int, List[int]] = {}

    for s in range(n_servers):
        cap = scaled_capacities[s]
        if use_unary_slack:
            n_slack = max(cap, 0)
            slack_var_indices[s] = []
            slack_weights[s] = []
            for k in range(n_slack):
                idx = len(var_metadata)
                var_metadata.append(
                    {"type": "slack", "server": s, "slack_bit": k}
                )
                slack_var_indices[s].append(idx)
                slack_weights[s].append(1)
        else:
            if cap <= 0:
                slack_var_indices[s] = []
                slack_weights[s] = []
            else:
                n_bits = _ceil_log2(cap + 1)
                slack_var_indices[s] = []
                slack_weights[s] = []
                for k in range(n_bits):
                    idx = len(var_metadata)
                    var_metadata.append({
                        "type": "slack",
                        "server": s,
                        "slack_bit": k,
                        "weight": (1 << k),
                    })
                    slack_var_indices[s].append(idx)
                    slack_weights[s].append(1 << k)

    n_slack_vars = len(var_metadata) - n_assignment_vars
    n_vars = len(var_metadata)

    # Build Q matrix and offset
    Q = np.zeros((n_vars, n_vars), dtype=float)
    offset = 0.0

    def add_quadratic(i: int, j: int, coeff: float) -> None:
        """Add coeff * x_i * x_j to the QUBO (upper-triangular form).
        Stores 2*coeff for off-diagonal to account for x^T Q x convention
        where each pair appears once in the upper triangle.
        """
        if i == j:
            Q[i, i] += coeff
        else:
            lo, hi = min(i, j), max(i, j)
            Q[lo, hi] += 2.0 * coeff

    def add_linear(i: int, coeff: float) -> None:
        Q[i, i] += coeff

    def add_constant(c: float) -> None:
        nonlocal offset
        offset += c

    # Step 5: Objective terms: sum c_ij * x_ij
    for s in range(n_servers):
        for j in range(n_jobs):
            if (s, j) in assignment_var_indices:
                idx = assignment_var_indices[(s, j)]
                cost = instance.energy_costs.get((s, j), 0.0)
                add_linear(idx, cost)

    # Step 6: Assignment constraint penalties
    # lambda_1 * sum_j (sum_i x_ij - 1)^2
    for j in range(n_jobs):
        job_vars: List[int] = []
        for s in range(n_servers):
            if (s, j) in assignment_var_indices:
                job_vars.append(assignment_var_indices[(s, j)])

        if not job_vars:
            continue

        # Linear term: -lambda_1 * x_ij
        for idx in job_vars:
            add_linear(idx, -lambda_1)

        # Quadratic: lambda_1 * x_ia * x_ib (a != b)
        for a in range(len(job_vars)):
            for b in range(a + 1, len(job_vars)):
                add_quadratic(job_vars[a], job_vars[b], lambda_1)

        # Constant: lambda_1
        add_constant(lambda_1)

    # Step 7: Capacity constraint penalties
    # lambda_2 * sum_i (load_i + slack_i - C_i)^2
    for s in range(n_servers):
        cap = scaled_capacities[s]
        if cap <= 0:
            continue

        pairs: List[Tuple[int, float]] = []
        for j in range(n_jobs):
            if (s, j) in assignment_var_indices:
                idx = assignment_var_indices[(s, j)]
                coeff = scaled_demands[j]
                pairs.append((idx, float(coeff)))
        for k, idx in enumerate(slack_var_indices[s]):
            coeff = slack_weights[s][k]
            pairs.append((idx, float(coeff)))

        if not pairs:
            add_constant(lambda_2 * cap * cap)
            continue

        # Linear: -2 * lambda_2 * C_i * coeff * x
        for idx, coeff in pairs:
            add_linear(idx, -2.0 * lambda_2 * cap * coeff)

        # Quadratic terms
        for a in range(len(pairs)):
            for b in range(len(pairs)):
                if a == b:
                    add_linear(pairs[a][0], lambda_2 * pairs[a][1] ** 2)
                elif a < b:
                    coeff_prod = pairs[a][1] * pairs[b][1]
                    add_quadratic(
                        pairs[a][0], pairs[b][0], lambda_2 * coeff_prod
                    )

        # Constant: lambda_2 * C_i^2
        add_constant(lambda_2 * cap * cap)

    return QUBOProblem(
        instance=instance,
        Q=Q,
        offset=offset,
        n_assignment_vars=n_assignment_vars,
        n_slack_vars=n_slack_vars,
        n_vars=n_vars,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        var_metadata=var_metadata,
        scaled_demands=scaled_demands,
        scaled_capacities=scaled_capacities,
        slack_scale=scale,
    )


def compute_qubo_energy(
    binary_vector: np.ndarray,
    qubo: QUBOProblem,
) -> float:
    """
    Compute the QUBO energy: E = x^T Q x + offset.

    Parameters
    ----------
    binary_vector : np.ndarray
        Binary vector of shape (n_vars,) with entries in {0, 1}.
    qubo : QUBOProblem
        The QUBO problem definition.

    Returns
    -------
    float
        The QUBO energy value.
    """
    x = np.asarray(binary_vector, dtype=float)
    energy = 0.0
    n = len(x)
    for i in range(n):
        if x[i] != 0:
            energy += qubo.Q[i, i] * x[i]
            for j in range(i + 1, n):
                if x[j] != 0:
                    energy += qubo.Q[i, j] * x[i] * x[j]
    energy += qubo.offset
    return energy


def decode_solution(
    binary_vector: np.ndarray,
    qubo: QUBOProblem,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Decode a QUBO binary vector into assignment and slack variables.

    Parameters
    ----------
    binary_vector : np.ndarray
        Binary vector of shape (n_vars,).
    qubo : QUBOProblem
        The QUBO problem with metadata for variable interpretation.

    Returns
    -------
    tuple of (dict, dict)
        (assignment, slack_values)
        - assignment: dict mapping job -> server (for assignment vars = 1)
        - slack_values: dict mapping server -> total slack value
          (sum of binary slack vars * their weights, in scaled units)

    Raises
    ------
    ValueError
        If the bitstring encodes multiple servers for the same job
        (i.e., duplicate assignment variables are active). Such bitstrings
        are invalid assignment encodings and should be treated as infeasible.
    """
    x = np.asarray(binary_vector)
    assignment: Dict[int, int] = {}
    slack_values: Dict[int, int] = {}

    for i, meta in enumerate(qubo.var_metadata):
        if x[i] == 1:
            if meta["type"] == "assignment":
                job = meta["job"]
                server = meta["server"]
                if job in assignment:
                    raise ValueError(
                        f"Bitstring encodes multiple servers for job {job} "
                        f"(server {assignment[job]} and server {server}). "
                        f"This is an invalid assignment encoding."
                    )
                assignment[job] = server
            elif meta["type"] == "slack":
                server = meta["server"]
                weight = meta.get("weight", 1)
                slack_values[server] = slack_values.get(server, 0) + weight

    return (assignment, slack_values)


def compute_penalty_terms(
    binary_vector: np.ndarray,
    qubo: QUBOProblem,
) -> Dict[str, float]:
    """
    Decompose the QUBO energy into its component terms for validation.

    Returns a dictionary with:
      - 'objective': Original objective value (sum of c_ij * x_ij)
      - 'assignment_penalty': lambda_1 * sum_j (sum_i x_ij - 1)^2
      - 'capacity_penalty': lambda_2 * sum_i (load_i + slack_i - C_i)^2
      - 'total': Full QUBO energy (should equal objective + penalties)

    This function iterates over the raw binary vector and var_metadata
    to correctly handle cases where multiple assignment variables for
    the same job are active (which would be an infeasible solution).

    Parameters
    ----------
    binary_vector : np.ndarray
        Binary vector of shape (n_vars,).
    qubo : QUBOProblem
        The QUBO problem definition.

    Returns
    -------
    dict of str -> float
    """
    x = np.asarray(binary_vector)
    instance = qubo.instance

    # Objective: sum of c_ij * x_ij over active assignment variables
    obj = 0.0
    for i, meta in enumerate(qubo.var_metadata):
        if x[i] == 1 and meta["type"] == "assignment":
            server = meta["server"]
            job = meta["job"]
            cost = instance.energy_costs.get((server, job), 0.0)
            obj += cost

    # Assignment penalty: lambda_1 * sum_j (sum_i x_ij - 1)^2
    # Count all active assignment vars per job (not decoded)
    assignment_penalty = 0.0
    job_counts: Dict[int, int] = {}
    for i, meta in enumerate(qubo.var_metadata):
        if x[i] == 1 and meta["type"] == "assignment":
            job = meta["job"]
            job_counts[job] = job_counts.get(job, 0) + 1
    for j in range(instance.n_jobs):
        n_assigned = job_counts.get(j, 0)
        assignment_penalty += qubo.lambda_1 * (n_assigned - 1) ** 2

    # Capacity penalty: lambda_2 * sum_i (load_i + slack_i - C_i)^2
    # Uses scaled values matching the QUBO construction
    capacity_penalty = 0.0
    server_loads: Dict[int, int] = {}
    slack_values: Dict[int, int] = {}
    for i, meta in enumerate(qubo.var_metadata):
        if x[i] == 1:
            s = meta["server"]
            if meta["type"] == "assignment":
                demand = qubo.scaled_demands[meta["job"]]
                server_loads[s] = server_loads.get(s, 0) + demand
            elif meta["type"] == "slack":
                weight = meta.get("weight", 1)
                slack_values[s] = slack_values.get(s, 0) + weight

    for s in range(instance.n_servers):
        load = server_loads.get(s, 0)
        slack = slack_values.get(s, 0)
        cap = qubo.scaled_capacities[s]
        excess = load + slack - cap
        capacity_penalty += qubo.lambda_2 * excess ** 2

    total = obj + assignment_penalty + capacity_penalty

    return {
        "objective": obj,
        "assignment_penalty": assignment_penalty,
        "capacity_penalty": capacity_penalty,
        "total": total,
    }