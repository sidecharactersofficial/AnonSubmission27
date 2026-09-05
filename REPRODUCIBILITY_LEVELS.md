# Reproducibility Levels

Not every result in this artifact has the same reproducibility guarantee.
This document classifies results into four levels so reviewers can
understand what to expect.

## Level Definitions

| Level | Meaning | Examples in this artifact |
|-------|---------|---------------------------|
| R1 | Exact byte-identical reproduction | Checkpoint hashes, dataset hashes, deterministic certificates, logit-reversal trace |
| R2 | Deterministic numerical reproduction | Mixed-norm K=0/1/2 counts, CROWN/IBP certified rates, JSMA vs exhaustive comparison |
| R3 | Statistical / tolerance-based reproduction | Multi-seed EXH K=1 sweeps (random-start PGD), Section III faithful diagnostic |
| R4 | Verification using archived canonical outputs | Paper figures, aggregated canonical tables |

## R1: Exact Byte-Identical

These results are deterministic given identical inputs. Verification
requires only checking that the output matches the archive.

- `manifest/checkpoint_sha256.json` — all 104 checkpoint hashes
- `manifest/dataset_sha256.txt` — all 6 dataset tensor hashes
- `results/logit_trace_model_adv_hardened.json` — logit-reversal trace
- `results/certificates/cert_ibp_*.json` — IBP certified bounds
- `results/certificates/crown_vs_ibp_*.json` — CROWN vs IBP bounds

## R2: Deterministic Numerical

These results are numerically stable across runs but may have minor
formatting differences. Verification requires exact numerical comparison.

- Mixed-norm K=0/1/2 counts (`results/results_nsl_kdd_*.json`)
- JSMA vs exhaustive comparison (`results/jsma_vs_exhaustive_unsw_nb15_K1.json`)
- Canonical aggregated tables (`results/consolidated/canonical_both_conventions.json`)

## R3: Statistical / Tolerance-Based

These results depend on stochastic optimization (random-start PGD, C&W).
Exact reproduction is impossible; tolerance-based comparison is used.

| Experiment | Tolerance | Comparator |
|------------|-----------|------------|
| EXH K=1 multiseed sweeps | `k0/k1_survivors` ±0.75% relative; `k1_pct` ±0.5 pp absolute | `verification/compare_exh_fresh.py` |
| Section III faithful diagnostic | ±0.4 pp observed spread | Single-run qualitative verification |

## R4: Archival Verification

These outputs are generated from canonical result files and should be
verified by regenerating statistics from the archive.

- `results/consolidated/canonical_both_conventions.json` — aggregated tables
- Paper figures (not included in artifact; generated from canonical results)

## Implications for Reviewers

- Do not assume R3 results are deterministic. Use tolerance-based comparison.
- R1 and R2 results can be verified with `make verify` and `make smoke`.
- R4 verification requires reading the canonical results and confirming
  they support the paper's tables and figures.
