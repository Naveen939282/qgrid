"""
Formulation package for Q-Grid.

Contains modules for encoding the constrained assignment problem
into QUBO form:
  - penalty.py: Penalty weight selection strategies.
  - qubo.py: QUBO matrix construction, energy evaluation, and decoding.
"""

from .penalty import (
    compute_max_objective_coefficient,
    compute_min_assignment_penalty,
    compute_min_capacity_penalty,
    compute_penalty_weights,
    check_penalty_sufficiency,
)
from .qubo import (
    QUBOProblem,
    encode_qubo,
    compute_qubo_energy,
    decode_solution,
    compute_penalty_terms,
)

__all__ = [
    "QUBOProblem",
    "encode_qubo",
    "compute_qubo_energy",
    "decode_solution",
    "compute_penalty_terms",
    "compute_max_objective_coefficient",
    "compute_min_assignment_penalty",
    "compute_min_capacity_penalty",
    "compute_penalty_weights",
    "check_penalty_sufficiency",
]