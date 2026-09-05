# ADR-002: Multi-Seed Statistical Evaluation

## Status

Accepted

## Context

Adversarial robustness evaluations are stochastic when using random-start
PGD or iterative optimization attacks. Single-run results can vary by
several percentage points depending on initialization.

The original artifact reported single-run results for each model. This made
it impossible to distinguish between:
- Genuine robustness differences between methods
- Run-to-run variance from unseeded attack initialization

## Decision

All canonical robustness claims are based on **multi-seed evaluation** with
pre-registered tolerances.

### Seed Coverage

| Dataset | Seeds | Models per seed | Total |
|---------|-------|-----------------|-------|
| NSL-KDD | 42, 43, 44 | 3 methods | 9 |
| CICIDS2017 | 42-54 | 3 methods | 39 |
| UNSW-NB15 | 42-51 | 3 methods | 27 |

### Tolerances

Pre-registered in `verification/compare_exh_fresh.py`:
- `n_total`, `clean_correct`, `attacked`: exact match required
- `k0_survivors`, `k1_survivors`: within ±0.75% relative
- `k1_pct_of_attacked`: within ±0.5 percentage points absolute

### Reporting

Aggregated results are reported as mean ± std across seeds. See
`canonical/consolidated_canonical_table.py`.

## Consequences

### Positive
- Distinguishes signal from noise in robustness comparisons
- Pre-registered tolerances prevent p-hacking
- Archived canonical results are stable anchors for reproduction

### Negative
- Reproduction requires running 75 model evaluations (9 + 39 + 27)
- Runtime is hours to days depending on hardware
- Reviewers cannot verify all multi-seed results in a short visit

### Neutral
- Single-run results are still meaningful for qualitative trends
- The `canonical/eval_unified.py` one-shot evaluator is retained for
  comparison but not used for canonical claims
