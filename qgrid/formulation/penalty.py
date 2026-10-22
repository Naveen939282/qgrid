"""
Penalty weight selection for QUBO encoding.

Defines strategies for choosing penalty weights (lambda_1 for
assignment constraints, lambda_2 for capacity constraints)
that ensure the QUBO ground state corresponds to a feasible solution
of the original problem when one exists.
"""

from __future__ import annotations

from typing import Tuple

from qgrid.problem import JobAssignmentInstance


def compute_max_objective_coefficient(instance: JobAssignmentInstance) -> float:
    """
    Compute the maximum absolute coefficient in the original objective.

    For the assignment problem, the objective is:
        sum_{i,j} c_{ij} * x_{ij}

    The maximum coefficient is max(|c_{ij}|) over all eligible pairs.
    """
    if not instance.energy_costs:
        return 0.0
    return max(abs(cost) for cost in instance.energy_costs.values())


def compute_min_assignment_penalty(
    instance: JobAssignmentInstance,
) -> float:
    """
    Compute a sufficient lower bound on lambda_1 (assignment penalty weight).

    The assignment constraint penalty is:
        lambda_1 * sum_j (sum_i x_{ij} - 1)^2

    The maximum violation per job (when all S eligible servers are
    assigned) is (S - 1)^2, so the maximum possible assignment penalty
    contribution is lambda_1 * sum_j (S_j - 1)^2 where S_j is the
    number of eligible servers for job j.

    For the ground state to correspond to a valid assignment, we need:
        lambda_1 > max_objective_coefficient * 2^n_vars

    However, a simpler and more practical bound is:
        lambda_1 >= max_objective_coefficient * (n_jobs + 1)

    This ensures that any assignment violation is penalized more than
    the worst-case objective improvement from an invalid assignment.
    """
    max_coeff = compute_max_objective_coefficient(instance)
    # Account for the fact that at most n_jobs assignments can be violated
    # Each violation costs at least lambda_1, but an assignment can
    # reduce the objective by at most max_coeff
    return max_coeff * (instance.n_jobs + 1)


def compute_min_capacity_penalty(
    instance: JobAssignmentInstance,
) -> float:
    """
    Compute a sufficient lower bound on lambda_2 (capacity penalty weight).

    The capacity constraint penalty is:
        lambda_2 * sum_i (sum_j w_j * x_{ij} + s_i - C_i)^2

    The maximum possible violation on a single server occurs when all
    eligible jobs are assigned to it, giving a violation of:
        max(0, sum_j w_j - C_i)

    A sufficient bound is:
        lambda_2 >= max_objective_coefficient * (n_jobs + 1)

    This ensures capacity violations are penalized more than
    objective improvements from over-subscription.
    """
    max_coeff = compute_max_objective_coefficient(instance)
    return max_coeff * (instance.n_jobs + 1)


def compute_penalty_weights(
    instance: JobAssignmentInstance,
    lambda_1: float | None = None,
    lambda_2: float | None = None,
    alpha: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute penalty weights for the QUBO encoding.

    Parameters
    ----------
    instance : JobAssignmentInstance
        The problem instance.
    lambda_1 : float or None
        User-specified assignment penalty weight. If None, a sufficient
        bound is computed automatically.
    lambda_2 : float or None
        User-specified capacity penalty weight. If None, a sufficient
        bound is computed automatically.
    alpha : float, default 1.0
        Safety multiplier for auto-computed penalties (e.g., 1.2 for 20% margin).

    Returns
    -------
    tuple of (float, float)
        (lambda_1, lambda_2) penalty weights.

    Notes
    -----
    The penalty weights must satisfy:
      lambda_1 > max_coeff * (n_jobs + 1)   for assignment penalties
      lambda_2 > max_coeff * (n_jobs + 1)   for capacity penalties

    Where max_coeff is the maximum absolute energy cost coefficient.
    This ensures that valid solutions always have lower energy than
    infeasible ones, so the QUBO ground state is feasible.

    If a user provides weights that are insufficient, the QUBO energy
    evaluator will detect this during validation.
    """
    if lambda_1 is None:
        lambda_1 = compute_min_assignment_penalty(instance) * alpha
    if lambda_2 is None:
        lambda_2 = compute_min_capacity_penalty(instance) * alpha
    return (lambda_1, lambda_2)


def check_penalty_sufficiency(
    instance: JobAssignmentInstance,
    lambda_1: float,
    lambda_2: float,
) -> Tuple[bool, str]:
    """
    Check whether the given penalty weights are sufficient.

    Parameters
    ----------
    instance : JobAssignmentInstance
        The problem instance.
    lambda_1 : float
        Assignment penalty weight.
    lambda_2 : float
        Capacity penalty weight.

    Returns
    -------
    tuple of (bool, str)
        (is_sufficient, message)
    """
    min_l1 = compute_min_assignment_penalty(instance)
    min_l2 = compute_min_capacity_penalty(instance)

    if lambda_1 < min_l1:
        return (
            False,
            f"lambda_1={lambda_1} must be >= {min_l1:.4f} "
            f"(max_coeff * (n_jobs + 1))",
        )
    if lambda_2 < min_l2:
        return (
            False,
            f"lambda_2={lambda_2} must be >= {min_l2:.4f} "
            f"(max_coeff * (n_jobs + 1))",
        )
    return (True, "Penalty weights are sufficient.")