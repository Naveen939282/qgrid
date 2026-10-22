"""
Solution representation, feasibility reports, and problem evaluation.

This module provides:
  - AssignmentSolution: a complete mapping of jobs to servers
  - FeasibilityReport: structured result of constraint checking
  - AssignmentProblem: wrapper that evaluates feasibility and computes
    the objective (total energy cost) of a solution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .instance import JobAssignmentInstance


@dataclass(frozen=True)
class AssignmentSolution:
    """
    A complete assignment of every job to exactly one server.

    Parameters
    ----------
    assignment : dict of int -> int
        Maps each job index to the server index it is assigned to.
        Must cover every job exactly once.

    Raises
    ------
    ValueError
        If the assignment is empty or if any key or value is not an
        integer.
    """

    assignment: Dict[int, int]

    def __post_init__(self) -> None:
        if not self.assignment:
            raise ValueError("assignment must not be empty")
        for k, v in self.assignment.items():
            if not isinstance(k, int) or not isinstance(v, int):
                raise ValueError(
                    f"assignment keys and values must be integers; "
                    f"got key={k!r}, value={v!r}"
                )

    def to_list(self, n_jobs: int) -> List[int]:
        """Convert to a list indexed by job. Returns server for each job."""
        result = [-1] * n_jobs
        for job, server in self.assignment.items():
            result[job] = server
        return result


@dataclass
class FeasibilityReport:
    """
    Detailed result of checking a proposed solution against all hard
    constraints of the problem instance.
    """

    is_feasible: bool
    missing_assignments: List[int] = field(default_factory=list)
    duplicate_assignments: List[Tuple[int, int]] = field(default_factory=list)
    ineligible_assignments: List[Tuple[int, int]] = field(default_factory=list)
    capacity_violations: List[Tuple[int, float, float]] = field(
        default_factory=list
    )

    def __str__(self) -> str:
        if self.is_feasible:
            return "Feasible: all constraints satisfied."
        parts = ["Infeasible."]
        if self.missing_assignments:
            parts.append(f"Missing: {self.missing_assignments}")
        if self.duplicate_assignments:
            parts.append(f"Duplicate jobs: {self.duplicate_assignments}")
        if self.ineligible_assignments:
            parts.append(f"Ineligible: {self.ineligible_assignments}")
        if self.capacity_violations:
            parts.append(f"Capacity violations: {self.capacity_violations}")
        return " ".join(parts)


@dataclass
class AssignmentProblem:
    """
    Encapsulates a JobAssignmentInstance together with methods to evaluate
    feasibility and compute the objective value of a proposed solution.
    """

    instance: JobAssignmentInstance

    def check_feasibility(
        self, solution: AssignmentSolution
    ) -> FeasibilityReport:
        """
        Check whether solution satisfies every hard constraint.

        Constraints
        -----------
        1. Every job is assigned (no missing assignments).
        2. Each job appears at most once (no duplicate jobs).
        3. Each assignment must be eligible (job is in server's
           eligible set).
        4. Server capacity must not be exceeded.

        Returns
        -------
        FeasibilityReport
        """
        report = FeasibilityReport(is_feasible=True)

        # 1. Missing assignments
        assigned_jobs: Set[int] = set(solution.assignment.keys())
        expected_jobs: Set[int] = set(range(self.instance.n_jobs))
        report.missing_assignments = sorted(expected_jobs - assigned_jobs)

        # 2. Duplicate assignments
        seen: Set[int] = set()
        dupes: List[Tuple[int, int]] = []
        for job, server in solution.assignment.items():
            if job in seen:
                dupes.append((job, server))
            seen.add(job)
        report.duplicate_assignments = dupes

        # 3. Ineligible assignments
        for job, server in solution.assignment.items():
            if 0 <= server < self.instance.n_servers:
                if job not in self.instance.eligible_jobs[server]:
                    report.ineligible_assignments.append((job, server))

        # 4. Capacity violations
        usage: Dict[int, float] = {
            s: 0.0 for s in range(self.instance.n_servers)
        }
        for job, server in solution.assignment.items():
            usage[server] += self.instance.job_demands[job]
        for s in range(self.instance.n_servers):
            cap = self.instance.server_capacities[s]
            used = usage[s]
            if used > cap + 1e-9:
                report.capacity_violations.append((s, used, cap))

        # Aggregate
        report.is_feasible = (
            len(report.missing_assignments) == 0
            and len(report.duplicate_assignments) == 0
            and len(report.ineligible_assignments) == 0
            and len(report.capacity_violations) == 0
        )
        return report

    def evaluate(self, solution: AssignmentSolution) -> float:
        """
        Compute the total energy cost of solution.

        The objective is the sum of per-assignment energy costs over
        all jobs. This is well-defined only for feasible assignments
        where every (job, server) pair is eligible; otherwise a
        ValueError is raised.

        Parameters
        ----------
        solution : AssignmentSolution
            The proposed assignment.

        Returns
        -------
        float
            Total energy cost.

        Raises
        ------
        ValueError
            If the solution is infeasible.
        """
        report = self.check_feasibility(solution)
        if not report.is_feasible:
            raise ValueError(
                f"Cannot evaluate objective for infeasible solution: {report}"
            )

        total: float = 0.0
        for job, server in solution.assignment.items():
            key: Tuple[int, int] = (server, job)
            if key not in self.instance.energy_costs:
                raise ValueError(
                    f"Missing energy cost for eligible pair {key}"
                )
            total += self.instance.energy_costs[key]
        return total