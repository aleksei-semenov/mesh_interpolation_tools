# §6 Independent refinement

Spec: [Experiment List.md](../Experiment%20List.md), section 6.

## Goal

Using the smooth field, refine one thing at a time:

- **A**: subdivision k, with both meshes fixed
- **B**: source lattice 16/32/64/128, on a fine target with k large enough
- **C**: target grid 16/32/64/128, on a fixed source

## Figure 6

- A: geometric E2 vs 2^-k
- B: total E2 vs h_S, one curve per reconstruction
- C: geometric E2 vs h_T, one curve per fixed k

Slope guides are guides only.

## Status

| Item | Where | Status |
|------|-------|--------|
| sweeps A–C and figure | `suites/smooth_robustness_benchmarks` → `smooth_refinement_*.pdf` | done |
| flag where geometric error contaminates source sweep | same (circled points, 10 % threshold) | done |
| protocol sizes 16–128 | defaults are 5/8/12 and 6/11/16 | **open**: needs larger runs |

Example for a larger run (memory grows ~ N·4^k, so start small):

```bash
python run_benchmarks.py --source-n 32 --target-n 32 --source-sweep 16 32 64 \
  --target-sweep 16 32 64 --k 0 1 2 3 4 5 --refine-k 5 --out ../../s06_independent_refinement/results
```

## Open work, belongs here

- [ ] Production-size runs, possibly split per sweep so that each fits in memory.
