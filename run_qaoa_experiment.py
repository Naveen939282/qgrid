"""
Reproducible QAOA experiment on a tiny instance.

Run with:
    python run_qaoa_experiment.py

This script demonstrates QAOA on a 2-job, 2-server problem that can
be exhaustively solved. It shows:
  - QAOA best feasible solution and objective
  - QAOA lowest-energy bitstring (may be infeasible)
  - Exact optimal for comparison
  - Whether QAOA found the optimum
"""
from qgrid.formulation import encode_qubo
from qgrid.problem import JobAssignmentInstance, AssignmentProblem
from qgrid.solvers.qaoa import solve_qaoa
from qgrid.solvers.enumeration import solve_exact

# --- Tiny instance ---
inst = JobAssignmentInstance(
    n_jobs=2,
    n_servers=2,
    job_demands=[3.0, 2.0],
    server_capacities=[5.0, 10.0],
    eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
    energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
)
problem = AssignmentProblem(instance=inst)
qubo = encode_qubo(inst)

print(f"QUBO: {qubo.n_vars} variables, offset={qubo.offset}")
print(f"Lambda_1={qubo.lambda_1}, Lambda_2={qubo.lambda_2}")

# --- Exact optimal ---
exact = solve_exact(problem)
print()
print("Exact solver:")
print(f"  Objective: {exact.objective}")
print(f"  Assignment: {exact.solution.assignment}")
print(f"  Status: {exact.status}")

# --- QAOA ---
result = solve_qaoa(
    problem, qubo,
    p=2,
    n_shots=1024,
    maxiter=100,
    seed=42,
)

print()
print(f"QAOA result (p={result.qaoa_depth}):")
print(f"  Status: {result.status}")
print(f"  Runtime: {result.runtime_seconds:.3f}s")
print(f"  Params (gamma, beta): {result.params}")
print(f"  Best energy bitstring: {result.best_energy_bitstring}")
print(f"  Best energy: {result.best_energy:.6f}")
if result.best_feasible_solution is not None:
    print(f"  Best feasible objective: {result.best_feasible_objective}")
    print(f"  Best feasible assignment: {result.best_feasible_solution.assignment}")
    match = "MATCH" if result.best_feasible_objective == exact.objective else "NO MATCH"
    print(f"  vs optimal: {match}")
else:
    print("  Best feasible solution: None (no feasible candidate sampled)")

print()
print("Top samples:")
for bitstring, count in result.all_samples[:5]:
    print(f"  {bitstring}: {count}")
