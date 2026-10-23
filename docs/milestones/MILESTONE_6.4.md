# Milestone 6.4 — Final Checkpoint

## Status: PASS with caveats

## Implementation & Artifacts

### Files changed (uncommitted)
- `qgrid/benchmark.py` — extended aggregation, new instances, QUBO resources
- `tests/test_solvers/test_benchmark.py` — 18 new tests
- `multi_seed_benchmark.py` — new multi-seed runner (untracked)
- `results/benchmarks/multi_seed/` — 6 JSON result files (untracked)
- `results/benchmarks/` — 3 additional JSON files from backward-compat run (untracked)

### Git state
- Latest commit: `7bb2eda` (Milestone 6.2 baseline)
- Changes are **not committed**
- `git diff --check`: clean (no whitespace errors)

## Audited Results

### Six instances, 60 QAOA trials
| Instance | Exact optimum | QUBO vars | QAOA seeds |
|----------|--------------|-----------|------------|
| tiny_2job_2server | 5.0 | 11 | 10 |
| medium_2job_2server | 5.0 | 10 | 10 |
| tight_capacity_restricted | 2.0 | 7 | 10 |
| asymmetric_3job_3server | 6.0 | 17 | 10 |
| tight_capacity_3job_2server | 6.0 | 12 | 10 |
| cost_asymmetry_2job_2server | 3.0 | 10 | 10 |

### Aggregate counts (independently recomputed from raw records)
- Total QAOA trials attempted: **60**
- Total feasible QAOA trials: **59**
- Total infeasible (completed, feasible=false): **1** (seed 4, tight_capacity_restricted)
- Total optimum-matching feasible trials: **59** (within 1e-6 tolerance)
- Total proven-optimal QAOA trials: **0**
- Total proven-optimal (exact + CP-SAT): **12** across 6 instances

### Test results
- **178 passed, 0 warnings** in ~193s
- (160 existing + 18 new tests)

## Audit Findings

### Aggregate verification
- All independently computed statistics match stored aggregate fields across all 6 JSON files
- No aggregate mismatches found
- Missing objectives correctly stored as `null` (not zero)
- Failed/infeasible trials preserved in raw records (`raw_feasible`, `raw_objectives`, `raw_status`)

### Exact optimum verification
- All 6 instances: exact solver status = "optimal", is_optimal = True
- Stored exact_optimum matches exact solver's objective in every case
- Gap references correctly use original objective, not QUBO energy

### Reproducibility
- Seed lists in configuration match trial seed records exactly (`[0,1,...,9]` for all 6 instances)
- Seeded components: `np.random.seed(seed)` (init params), `StatevectorSampler(seed=seed)` (optimization + final sampling)
- **Caveat**: Cross-session reproducibility (bit-for-bit) was **not** tested — audit was read-only

### Memory estimate
- The "6.7 GB" memory figure is **unverified** — no memory measurement instrumentation exists in the codebase or saved JSON records
- No `memory`, `rss`, `peak`, or similar references found in any source file

## Limitations
- Pilot-scale: 10 seeds × 6 small instances × p=1, 512 shots
- No statistical significance claims made
- No quantum-advantage claims made
- Statevector simulation only (no real quantum hardware)
- 17-qubit instance (asymmetric_3job_3server) is computationally expensive (~5 min for 10 seeds)