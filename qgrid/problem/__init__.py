"""Problem package for Q-Grid.

Contains the job-to-server assignment problem definition, validation,
feasibility checking, and objective evaluation.
"""

from .instance import JobAssignmentInstance
from .problem import AssignmentProblem, AssignmentSolution, FeasibilityReport

__all__ = [
    "JobAssignmentInstance",
    "AssignmentProblem",
    "AssignmentSolution",
    "FeasibilityReport",
]