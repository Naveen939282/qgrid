"""Pytest configuration for Q-Grid tests.

Sets up the Python path and provides shared fixtures for test instances.
"""

import sys
from pathlib import Path

import pytest

# Add the project root so `import qgrid` works
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qgrid.problem import (
    AssignmentProblem,
    AssignmentSolution,
    JobAssignmentInstance,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_instance() -> JobAssignmentInstance:
    """
    A 2-job, 2-server instance with known optimal solution.

    Instance layout (2 jobs, 2 servers):
    - Job 0: demand 3
    - Job 1: demand 2
    - Server 0: capacity 5, eligible for {0, 1}
    - Server 1: capacity 10, eligible for {0, 1}

    Energy costs:
    - (server 0, job 0) = 5.0
    - (server 0, job 1) = 3.0
    - (server 1, job 0) = 2.0
    - (server 1, job 1) = 4.0

    Feasible assignments:
    - {0: 0, 1: 0}  -> cost = 5.0 + 3.0 = 8.0  (capacity: 5 <= 5)
    - {0: 0, 1: 1}  -> cost = 5.0 + 4.0 = 9.0  (capacity: 3 <= 5, 2 <= 10)
    - {0: 1, 1: 0}  -> cost = 2.0 + 3.0 = 5.0  (capacity: 2 <= 5, 3 <= 10)  <- OPTIMAL
    - {0: 1, 1: 1}  -> cost = 2.0 + 4.0 = 6.0  (capacity: 5 <= 10)
    """
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[3.0, 2.0],
        server_capacities=[5.0, 10.0],
        eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
        energy_costs={
            (0, 0): 5.0,
            (0, 1): 3.0,
            (1, 0): 2.0,
            (1, 1): 4.0,
        },
    )


@pytest.fixture
def tiny_problem(tiny_instance: JobAssignmentInstance) -> AssignmentProblem:
    """AssignmentProblem wrapping the tiny_instance."""
    return AssignmentProblem(instance=tiny_instance)


@pytest.fixture
def infeasible_capacity_instance() -> JobAssignmentInstance:
    """1 job, 1 server, demand (10) exceeds capacity (5)."""
    return JobAssignmentInstance(
        n_jobs=1,
        n_servers=1,
        job_demands=[10.0],
        server_capacities=[5.0],
        eligible_jobs=[frozenset({0})],
        energy_costs={(0, 0): 1.0},
    )


@pytest.fixture
def infeasible_capacity_problem(
    infeasible_capacity_instance: JobAssignmentInstance,
) -> AssignmentProblem:
    """AssignmentProblem wrapping the infeasible instance."""
    return AssignmentProblem(instance=infeasible_capacity_instance)


@pytest.fixture
def eligibility_instance() -> JobAssignmentInstance:
    """
    Instance where job 1 is NOT eligible for server 0.
    2 jobs, 2 servers:
    - Job 0: eligible on server 0 and 1
    - Job 1: eligible ONLY on server 1
    """
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[1.0, 1.0],
        server_capacities=[10.0, 10.0],
        eligible_jobs=[
            frozenset({0}),
            frozenset({0, 1}),
        ],
        energy_costs={
            (0, 0): 1.0,
            (1, 0): 2.0,
            (1, 1): 3.0,
        },
    )


@pytest.fixture
def eligibility_problem(
    eligibility_instance: JobAssignmentInstance,
) -> AssignmentProblem:
    """AssignmentProblem wrapping the eligibility instance."""
    return AssignmentProblem(instance=eligibility_instance)


@pytest.fixture
def tight_capacity_instance() -> JobAssignmentInstance:
    """
    Instance where greedy with naive ordering may fail.
    3 jobs, 1 server with tight capacity.
    Jobs: demands 3, 3, 4. Capacity = 10.
    All jobs must fit for feasibility (3+3+4 = 10).
    """
    return JobAssignmentInstance(
        n_jobs=3,
        n_servers=1,
        job_demands=[3.0, 3.0, 4.0],
        server_capacities=[10.0],
        eligible_jobs=[frozenset({0, 1, 2})],
        energy_costs={(0, 0): 1.0, (0, 1): 2.0, (0, 2): 3.0},
    )


@pytest.fixture
def tight_capacity_problem(
    tight_capacity_instance: JobAssignmentInstance,
) -> AssignmentProblem:
    """AssignmentProblem wrapping the tight capacity instance."""
    return AssignmentProblem(instance=tight_capacity_instance)


@pytest.fixture
def eligible_instance() -> JobAssignmentInstance:
    """
    A 3-job, 3-server instance with full eligibility and known structure.
    Useful for detailed solver comparison tests.
    """
    return JobAssignmentInstance(
        n_jobs=3,
        n_servers=3,
        job_demands=[2.0, 3.0, 1.0],
        server_capacities=[10.0, 10.0, 10.0],
        eligible_jobs=[
            frozenset({0, 1, 2}),
            frozenset({0, 1, 2}),
            frozenset({0, 1, 2}),
        ],
        energy_costs={
            (0, 0): 5.0,
            (0, 1): 4.0,
            (0, 2): 6.0,
            (1, 0): 3.0,
            (1, 1): 2.0,
            (1, 2): 5.0,
            (2, 0): 4.0,
            (2, 1): 6.0,
            (2, 2): 1.0,
        },
    )


@pytest.fixture
def eligible_problem(
    eligible_instance: JobAssignmentInstance,
) -> AssignmentProblem:
    """AssignmentProblem wrapping the eligible instance."""
    return AssignmentProblem(instance=eligible_instance)


@pytest.fixture
def qubo_test_instance() -> JobAssignmentInstance:
    """
    A 2-job, 2-server instance with minimal variable count, optimized for
    exhaustive QUBO validation (fewer slack variables).

    With capacities of 4 and 5, this requires only 3+3=6 slack bits,
    giving 4 assignment + 6 slack = 10 QUBO variables (1024 states).
    """
    return JobAssignmentInstance(
        n_jobs=2,
        n_servers=2,
        job_demands=[2.0, 1.0],
        server_capacities=[4.0, 5.0],
        eligible_jobs=[
            frozenset({0, 1}),
            frozenset({0, 1}),
        ],
        energy_costs={
            (0, 0): 3.0,
            (0, 1): 4.0,
            (1, 0): 2.0,
            (1, 1): 5.0,
        },
    )