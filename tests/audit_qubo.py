"""Minimal audit test."""
import itertools
import numpy as np
from qgrid.formulation import (
    encode_qubo, compute_qubo_energy, compute_penalty_terms,
    decode_solution, compute_penalty_weights,
)
from qgrid.formulation.penalty import check_penalty_sufficiency
from qgrid.problem import JobAssignmentInstance, AssignmentProblem, AssignmentSolution
from qgrid.solvers import solve_exact

# Tiny instance
inst = JobAssignmentInstance(
    n_jobs=2, n_servers=2, job_demands=[3.0, 2.0],
    server_capacities=[5.0, 10.0],
    eligible_jobs=[frozenset({0, 1}), frozenset({0, 1})],
    energy_costs={(0, 0): 5.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 4.0},
)

q = encode_qubo(inst)
n = q.n_vars
print(f"Tiny QUBO: {n} vars ({q.n_assignment_vars} assign + {q.n_slack_vars} slack)")

# Test 1: Energy convention
print("Test 1: Energy Convention")
for seed in range(3):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n).astype(float)
    e1 = compute_qubo_energy(x, q)
    e2 = sum(q.Q[i, i] * x[i] for i in range(n) if x[i] != 0)
    e2 += sum(q.Q[i, j] * x[i] * x[j] for i in range(n) for j in range(i + 1, n) if x[i] != 0 and x[j] != 0)
    e2 += q.offset
    Q_full = q.Q + q.Q.T - np.diag(np.diag(q.Q))
    e3 = x @ Q_full @ x + q.offset
    # Note: e1 and e2 use upper-triangular convention (standard QUBO)
    # e3 uses full symmetric matrix which doubles off-diagonal terms
    # Relationship: e3 = 2*(e1 - offset) + diag_part + offset
    # The key verification is e1 == e2 (upper-triangular consistency)
    ok = np.isclose(e1, e2, rtol=1e-8)
    status = "PASS" if ok else "FAIL"
    print(f"  E={e1:.4f} ut={e2:.4f} full={e3:.4f} [{status}]")

# Test 2: Penalty decomposition consistency
print("\nTest 2: Penalty Decomposition Consistency")
for seed in range(3):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n).astype(float)
    energy = compute_qubo_energy(x, q)
    terms = compute_penalty_terms(x, q)
    total = terms["objective"] + terms["assignment_penalty"] + terms["capacity_penalty"]
    ok = np.isclose(energy, total, rtol=1e-4, atol=1e-4)
    status = "PASS" if ok else "FAIL"
    print(f"  E={energy:.4f} obj+pen={total:.4f} [{status}]")
    if not ok:
        print(f"    obj={terms['objective']:.4f} assign={terms['assignment_penalty']:.4f} cap={terms['capacity_penalty']:.4f}")

# Test 3: Slack variable coverage
print("\nTest 3: Slack Variable Coverage")
for s in range(inst.n_servers):
    cap = q.scaled_capacities[s]
    print(f"  Server {s}: scaled_cap={cap}")
    if cap <= 0:
        continue
    slack_indices = [i for i, meta in enumerate(q.var_metadata) if meta["type"] == "slack" and meta["server"] == s]
    print(f"    Slack var indices: {slack_indices}")
    weights = [q.var_metadata[i].get("weight", 1) for i in slack_indices]
    print(f"    Weights: {weights}")
    max_rep = sum(weights)
    print(f"    Max representable: {max_rep}, requires: {cap}")
    missing = []
    for target in range(cap + 1):
        found = False
        for combo in itertools.product([0, 1], repeat=len(slack_indices)):
            val = sum(w * b for w, b in zip(weights, combo))
            if val == target:
                found = True
                break
        if not found:
            missing.append(target)
    if missing:
        print(f"    MISSING slack values: {missing}")
    else:
        print(f"    All values 0..{cap} representable [PASS]")


# Test 4: Under-utilization not penalized
print("\nTest 4: Under-utilization Not Penalized")
problem = AssignmentProblem(instance=inst)
underutil_checked = 0
for combo in itertools.product([0, 1], repeat=n):
    x = np.array(combo, dtype=float)
    terms = compute_penalty_terms(x, q)
    if terms["capacity_penalty"] == 0 and terms["assignment_penalty"] == 0:
        assignment, slack_vals = decode_solution(x, q)
        if len(assignment) == inst.n_jobs:
            sol = AssignmentSolution(assignment=assignment)
            report = problem.check_feasibility(sol)
            if report.is_feasible:
                loads = {}
                for job, srv in assignment.items():
                    loads[srv] = loads.get(srv, 0) + inst.job_demands[job]
                for s in range(inst.n_servers):
                    load = loads.get(s, 0)
                    cap = inst.server_capacities[s]
                    if load < cap and load > 0:
                        slack = slack_vals.get(s, 0) / q.slack_scale
                        expected_slack = cap - load
                        ok = np.isclose(slack, expected_slack, atol=1e-6)
                        st = "PASS" if ok else "FAIL"
                        print(f"  Server {s}: load={load}, cap={cap}, slack={slack}, expected={expected_slack} [{st}]")
                        underutil_checked += 1
print(f"  Checked {underutil_checked} under-utilization cases")


# Test 5: Ground state equivalence
print("\nTest 5: Ground State Equivalence")
l1, l2 = compute_penalty_weights(inst, alpha=2.0)
q2 = encode_qubo(inst, lambda_1=l1, lambda_2=l2)
best_energy = float("inf")
best_x = None
for combo in itertools.product([0, 1], repeat=n):
    x = np.array(combo, dtype=float)
    e = compute_qubo_energy(x, q2)
    if e < best_energy:
        best_energy = e
        best_x = x
assignment, _ = decode_solution(best_x, q2)
sol = AssignmentSolution(assignment=assignment)
feasible = problem.check_feasibility(sol).is_feasible
exact = solve_exact(problem)
match = np.isclose(best_energy, exact.objective, atol=0.01)
status = "PASS" if (match and feasible) else "FAIL"
print(f"  QUBO ground state: {best_energy:.4f}")
print(f"  Exact optimum: {exact.objective:.4f}")
print(f"  Feasible: {feasible}, Match: {match} [{status}]")


# Test 6: Infeasible instance
print("\nTest 6: Infeasible Instance")
inst_inf = JobAssignmentInstance(
    n_jobs=1, n_servers=1, job_demands=[10.0],
    server_capacities=[5.0], eligible_jobs=[frozenset({0})],
    energy_costs={(0, 0): 1.0},
)
q_inf = encode_qubo(inst_inf)
n_inf = q_inf.n_vars
best_inf = float("inf")
best_x_inf = None
for combo in itertools.product([0, 1], repeat=n_inf):
    x = np.array(combo, dtype=float)
    e = compute_qubo_energy(x, q_inf)
    if e < best_inf:
        best_inf = e
        best_x_inf = x
terms_inf = compute_penalty_terms(best_x_inf, q_inf)
has_penalty = terms_inf["assignment_penalty"] > 0 or terms_inf["capacity_penalty"] > 0
assignment_inf, _ = decode_solution(best_x_inf, q_inf)
# assignment_inf may be empty for infeasible instances
if len(assignment_inf) == 0:
    # No valid assignment possible - this confirms infeasibility
    inf_infeasible = True
    sol_inf = None
else:
    sol_inf = AssignmentSolution(assignment=assignment_inf)
    problem_inf = AssignmentProblem(instance=inst_inf)
    inf_infeasible = not problem_inf.check_feasibility(sol_inf).is_feasible
status = "PASS" if (has_penalty and inf_infeasible) else "FAIL"
print(f"  Penalty>0: {has_penalty}, Infeasible: {inf_infeasible} [{status}]")
print(f"  Penalties: assign={terms_inf['assignment_penalty']:.4f}, cap={terms_inf['capacity_penalty']:.4f}")


# Test 7: Penalty sufficiency bounds
print("\nTest 7: Penalty Weight Sufficiency Bounds")
tight = JobAssignmentInstance(
    n_jobs=3, n_servers=3, job_demands=[2.0, 2.0, 2.0],
    server_capacities=[4.0, 4.0, 4.0],
    eligible_jobs=[frozenset({0, 1, 2}), frozenset({0, 1, 2}), frozenset({0, 1, 2})],
    energy_costs={(0,0):1.0, (0,1):2.0, (0,2):3.0, (1,0):3.0, (1,1):1.0, (1,2):2.0, (2,0):2.0, (2,1):3.0, (2,2):1.0},
)
for name, inst_t, q_t in [("tiny", inst, q), ("tight", tight, encode_qubo(tight))]:
    l1, l2 = compute_penalty_weights(inst_t, alpha=2.0)
    suff, msg = check_penalty_sufficiency(inst_t, l1, l2)
    nt = q_t.n_vars
    best_t = float("inf")
    best_xt = None
    for combo in itertools.product([0, 1], repeat=nt):
        x = np.array(combo, dtype=float)
        e = compute_qubo_energy(x, q_t)
        if e < best_t:
            best_t = e
            best_xt = x
    assignment_t, _ = decode_solution(best_xt, q_t)
    sol_t = AssignmentSolution(assignment=assignment_t)
    problem_t = AssignmentProblem(instance=inst_t)
    feasible_t = problem_t.check_feasibility(sol_t).is_feasible
    status = "PASS" if (suff and feasible_t) else "FAIL"
    print(f"  {name}: l1={l1:.2f} l2={l2:.2f} suff={suff} feasible={feasible_t} [{status}]")


print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)

