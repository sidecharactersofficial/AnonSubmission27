# ADR-004: Canonical vs Fresh Reproduction Outputs

## Status

Accepted

## Context

Reviewers need to verify that the shipped canonical results can be
reproduced from the shipped code and checkpoints. However, stochastic attacks
(random-start PGD) produce run-to-run variance. A naive reproduction that
overwrites the archive would destroy the original results and make it
impossible to compare fresh runs against the paper's figures.

Additionally, reviewers may run evaluations with different hardware, Python
versions, or library builds. Exact byte-identical reproduction is impossible
for stochastic attacks, but tolerance-based comparison is feasible.

## Decision

Preserve two separate output namespaces:

1. **Canonical results** (`results/`) — Archived outputs from the paper's
   original evaluation runs. These are the authoritative numbers in the
   paper. Never overwritten by reproduction scripts.
2. **Fresh results** (`results/foolbox/*.fresh.json`,
   `results/*.repro.json`) — Newly generated outputs from reviewer runs.
   Written to distinct paths so the archive remains immutable.

## Comparison Mechanism

`verification/compare_exh_fresh.py` compares fresh outputs against archives
using pre-registered tolerances. This allows reviewers to:
- Run evaluations on their own hardware
- Compare against the canonical archive
- Verify that differences are within expected stochastic variance

## File Naming Convention

| Namespace | Pattern | Example |
|-----------|---------|---------|
| Canonical | `results/foolbox/exh_k1_pgd40_nsl_kdd_curriculum_seed42.json` | Archived paper result |
| Fresh | `results/foolbox/exh_k1_pgd40_nsl_kdd_curriculum_seed42.fresh.json` | Reviewer reproduction |
| Repro | `results/unified/eval_rsc_cicids2017_seed42.repro.json` | Re-run with `--suffix .repro` |

## Consequences

### Positive
- Canonical results are immutable anchors
- Reviewers can verify without risking archive corruption
- Fresh vs canonical comparison provides quantitative reproduction metrics

### Negative
- Two namespaces increase disk usage
- Reviewers must understand the distinction to avoid confusion
- Some scripts require explicit `--suffix` flags to avoid overwriting

### Neutral
- `make clean` removes fresh/repro outputs but preserves canonical results
- The `canonical/consolidated_canonical_table.py` aggregation script reads
  only canonical results, ensuring paper tables are always generated from
  the archive
