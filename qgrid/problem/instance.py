"""
Instance data structures for job-to-server assignment.

This module defines an immutable problem instance dataclass that captures:
  - Job resource demands
  - Server resource capacities
  - Server eligibility for each job (which jobs may run on which servers)
  - Energy costs for each eligible (server, job) pair
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import numpy as np


@dataclass(frozen=True)
class JobAssignmentInstance:
    """
    Immutable description of a single job-to-server assignment instance.

    Parameters
    ----------
    n_jobs : int
        Number of jobs (J). Jobs are indexed 0 .. n_jobs - 1.
    n_servers : int
        Number of servers (S). Servers are indexed 0 .. n_servers - 1.
    job_demands : list of float
        Resource demand of each job. job_demands[j] is the demand of
        job j. Must have length n_jobs and all entries must be
        non-negative.
    server_capacities : list of float
        Resource capacity of each server. server_capacities[i] is the
        capacity of server i. Must have length n_servers and all
        entries must be non-negative.
    eligible_jobs : list of frozenset
        For each server i, eligible_jobs[i] is a frozenset of job
        indices that are allowed to run on server i.
    energy_costs : dict of (int, int) -> float
        Mapping (server_idx, job_idx) -> energy_cost. Only entries for
        eligible pairs need to be present. All costs must be finite.

    Raises
    ------
    ValueError
        If any dimension is invalid, demands/capacities are negative,
        eligibility sets reference non-existent jobs, or energy costs
        are non-finite.
    """

    n_jobs: int
    n_servers: int
    job_demands: List[float]
    server_capacities: List[float]
    eligible_jobs: List[FrozenSet[int]]
    energy_costs: Dict[Tuple[int, int], float]

    def __post_init__(self) -> None:
        self._validate()

    # -- public helpers --------------------------------------------------

    @property
    def total_demand(self) -> float:
        """Sum of all job demands."""
        return float(sum(self.job_demands))

    @property
    def total_capacity(self) -> float:
        """Sum of all server capacities."""
        return float(sum(self.server_capacities))

    def is_feasible_by_capacity(self) -> bool:
        """
        Quick check: total demand must not exceed total capacity.

        This is a necessary (but not sufficient) condition for the
        existence of a feasible assignment.
        """
        return self.total_demand <= self.total_capacity + 1e-9

    def eligible_servers(self, job_idx: int) -> List[int]:
        """
        Return a sorted list of server indices eligible for job_idx.

        Parameters
        ----------
        job_idx : int
            The job to find eligible servers for.

        Returns
        -------
        list of int
            Server indices where this job may be assigned.

        Raises
        ------
        ValueError
            If job_idx is out of range.
        """
        if not isinstance(job_idx, int) or job_idx < 0 or job_idx >= self.n_jobs:
            raise ValueError(
                f"job_idx {job_idx} is out of range (n_jobs={self.n_jobs})"
            )
        servers = []
        for s in range(self.n_servers):
            if job_idx in self.eligible_jobs[s]:
                servers.append(s)
        return servers

    def min_server_for_job(self, job_idx: int) -> int:
        """
        Identify the eligible server with the smallest capacity that
        can still host this job.  Returns -1 if no eligible server
        has sufficient capacity.

        This is used by the greedy baseline heuristic.
        """
        best = -1
        best_cap = float("inf")
        for s in range(self.n_servers):
            if job_idx in self.eligible_jobs[s]:
                if self.job_demands[job_idx] <= self.server_capacities[s] + 1e-9:
                    if self.server_capacities[s] < best_cap:
                        best = s
                        best_cap = self.server_capacities[s]
        return best

    # -- private validation ---------------------------------------------

    def _validate(self) -> None:
        """Validate all instance fields."""
        if not isinstance(self.n_jobs, int) or self.n_jobs < 0:
            raise ValueError("n_jobs must be a non-negative integer")
        if not isinstance(self.n_servers, int) or self.n_servers < 0:
            raise ValueError("n_servers must be a non-negative integer")

        if len(self.job_demands) != self.n_jobs:
            raise ValueError(
                f"job_demands has length {len(self.job_demands)} "
                f"but n_jobs={self.n_jobs}"
            )
        if len(self.server_capacities) != self.n_servers:
            raise ValueError(
                f"server_capacities has length {len(self.server_capacities)} "
                f"but n_servers={self.n_servers}"
            )
        if len(self.eligible_jobs) != self.n_servers:
            raise ValueError(
                f"eligible_jobs has length {len(self.eligible_jobs)} "
                f"but n_servers={self.n_servers}"
            )

        for j, d in enumerate(self.job_demands):
            if not np.isfinite(d) or d < -1e-12:
                raise ValueError(
                    f"job_demands[{j}] = {d} must be non-negative and finite"
                )
        for s, c in enumerate(self.server_capacities):
            if not np.isfinite(c) or c < -1e-12:
                raise ValueError(
                    f"server_capacities[{s}] = {c} must be non-negative and finite"
                )

        for s, jobs in enumerate(self.eligible_jobs):
            if not isinstance(jobs, (frozenset, set)):
                raise TypeError(
                    f"eligible_jobs[{s}] must be a frozenset/set, "
                    f"got {type(jobs)}"
                )
            for j in jobs:
                if not isinstance(j, int) or j < 0 or j >= self.n_jobs:
                    raise ValueError(
                        f"eligible_jobs[{s}] contains invalid job index {j} "
                        f"(n_jobs={self.n_jobs})"
                    )

        for (s, j), cost in self.energy_costs.items():
            if not isinstance(s, int) or s < 0 or s >= self.n_servers:
                raise ValueError(
                    f"energy_costs key server {s} is out of range "
                    f"(n_servers={self.n_servers})"
                )
            if not isinstance(j, int) or j < 0 or j >= self.n_jobs:
                raise ValueError(
                    f"energy_costs key job {j} is out of range "
                    f"(n_jobs={self.n_jobs})"
                )
            if not np.isfinite(cost):
                raise ValueError(
                    f"energy_costs[(server {s}, job {j})] = {cost} "
                    f"must be finite"
                )