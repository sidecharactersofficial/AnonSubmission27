# Reproducibility Artifact

Everything needed to reproduce every table and figure of the paper from the
shipped checkpoints, scripts, and result files.

```
.
├── Makefile                     reproducible proof pipeline (see below)
├── LICENSE                      MIT
├── requirements.txt             pinned environment
├── dataset_links.md             dataset acquisition + integrity anchors
├── download_data.py             NSL-KDD fetch + preprocessing (deterministic)
├── app/                          model architecture, loaders, attack implementations
├── canonical/                    final evaluators supporting reported conclusions
├── diagnostics/                  historical implementations used to reproduce and diagnose retracted results
├── verification/                 independent integrity and soundness checks
├── models/                       trained checkpoints (SHA-256 pinned in manifest/)
├── manifest/
│   ├── checkpoint_sha256.json    SHA-256 of every model file
│   └── dataset_sha256.txt        SHA-256 of preprocessed tensors (not shipped)
└── results/
    ├── foolbox/                  multi-seed EXH K=1 sweeps (pgd40 + deepfool)
    ├── certificates/             CROWN vs IBP certified rates
    ├── consolidated/             aggregated canonical tables
    ├── section3/                 faithful AdvGuard diagnostic
    ├── cw/                       C&W survivor records
    └── *.json                    mixed-norm K=0/1/2, scalability, JSMA, trace, unified
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
. .venv/bin/activate
make setup          # fetches NSL-KDD; CICIDS2017/UNSW-NB15 per dataset_links.md
make verify         # confirm every checkpoint matches its pinned SHA-256
make smoke          # fast end-to-end check (~2 min)
make reproduce-table3
...
```

GPU is optional. All evaluations default to CPU (as archived); only
`reproduce-scalability` / `reproduce-jsma` use CUDA when available. Runtimes
cited are from the reference machine: 6 GB mobile GPU, 10-core CPU, 11 GB RAM.

## Environment

| Package   | Version (pinned) |
|-----------|------------------|
| torch     | 2.9.1+cu128      |
| numpy     | 2.4.0            |
| pandas    | 2.3.3            |
| pyarrow   | 24.0.0           |
| scipy     | 1.16.3           |
| foolbox   | 3.3.4            |
| auto-LiRPA| 0.3              |

## What reproduces exactly vs. within tolerance

| Quantity                                        | Behavior |
|-------------------------------------------------|----------|
| checkpoint SHA-256 (all 104 files)              | exact (manifest) |
| mixed-norm K=0/1/2 counts (NSL-KDD)             | exact (`ALL CHECKS PASS`) |
| certified CROWN/IBP rates (27 checkpoints)      | byte-identical |
| JSMA vs exhaustive-image comparison             | byte-identical |
| logit-reversal trace                            | byte-identical |
| canonical aggregated tables                     | exact (input-driven) |
| EXH K=1 multiseed sweeps (foolbox)              | within tolerance (below) |
| Section III faithful diagnostic                 | within noise (below) |

**EXH K=1 sweeps.** `eval_deepfool_k1.py` performs an **unseeded** random-start
PGD-40; each surviving-sample outcome can flip between runs. The comparator
(`verification/compare_exh_fresh.py`) pre-registers tolerances: `n_total`,
`clean_correct`, `attacked`, and embedded `checkpoint_sha256` are exact;
`k0/k1_survivors` within ±0.75% relative; `k1_pct_of_attacked` within ±0.5 pp.
Observed drift in validation runs was ≪ 0.1 pp.

**Section III diagnostic.** The faithful AdvGuard evaluator uses an unseeded
random-start PGD. Observed run-to-run spread ±0.4 pp. Archived draw:
hardened @ eps 0.15 = **40.36%** (paper-prose table), baseline 16.06%. The
robust claim under reproduction: hardened ≈ 40 ± 0.4% vs baseline ≈ 16%, with
the originally-reported ~29.10% not reproduced (the Section III finding).

## Script → paper mapping

| Target | Script | Paper content | Runtime (ref.) |
|--------|--------|---------------|----------------|
| `reproduce-table3` | `canonical/section3_faithful_diagnostic.py` | Table III prose: faithful AdvGuard reproduction (baseline 16.06%, hardened 40.36% @ 0.15) | ~45 s (CUDA) / ~4 min (CPU) |
| `reproduce-table-mixednorm` | `canonical/run_paper_mixednorm.py` + `canonical/eval_mixed_norm.py` | mixed-norm K=0/1/2 collapse, NSL-KDD (Hardened 17421/0/0; Curriculum 5604/9/9; Baseline 3342/0/0; RSC 18106/0) | ~26 s |
| `reproduce-multiseed-nslkdd` | `canonical/eval_deepfool_k1.py --attack pgd40` | multiseed EXH K=1, NSL-KDD, 9 models | ~10 min |
| `reproduce-multiseed-cicids` | same | multiseed EXH K=1, CICIDS2017, 39 models | ~6 h (parallelize) |
| `reproduce-multiseed-unsw` | same | multiseed EXH K=1, UNSW-NB15, 27 models | ~9–10 h (parallelize) |
| `reproduce-certified` | `canonical/run_crown_vs_ibp.py` (+ `canonical/crown_bound.py`, `canonical/certified_bound.py`) | Tables VIII–IX: CROWN vs IBP certified K=0/K=1 rates, 27 checkpoints | ~12–15 min |
| `reproduce-scalability` | `canonical/eval_scalability.py` | Section VII-G: K=1 0.0155 s/state, K=2 0.118 s/state (reference GPU) | ~2 min (GPU) |
| `reproduce-jsma` | `canonical/eval_jsma_vs_exhaustive.py` | Section VII-H: JSMA vs exhaustive-image, 500 UNSW samples | ~1 min |
| `reproduce-logit-trace` | `canonical/trace_logit_reversal_hardened.py` | Table III prose: per-sample logit reversal for the hardened NSL model | ~4 s |
| `reproduce-canonical-table` | `canonical/consolidated_canonical_table.py` | aggregated canonical summary, both K conventions | <1 s |

### Deepfool variant
`canonical/eval_deepfool_k1.py --attack deepfool` reproduces the DeepFool EXH K=1
records (`results/foolbox/exh_k1_deepfool_*`; 47 files). Use
`--suffix .fresh` and `verification/compare_exh_fresh.py --attack deepfool` to
verify. Two CICIDS2017 seed-53 records (hardened, curriculum) are intentionally
absent from the archive (evaluation budget); `--fresh` re-creates them.

### Superseded evaluator (archived for completeness)
`canonical/eval_unified.py` was used for the Section VII-B masking-gap analysis
(`results/unified/*`). It is superseded by `canonical/eval_deepfool_k1.py` +
`canonical/consolidated_canonical_table.py`; kept for provenance. Usage:
`python3 canonical/eval_unified.py --datasets cicids2017 --suffix .repro`.

## Re-generation vs. shipped results

**Shipped result files are authoritative.** `reproduce-*` targets write fresh
output under `*.repro.*` / `*.fresh.*` / `results_fresh/` paths and never
overwrite the archive, then verify. Use `make clean` to remove fresh outputs.

Confirmed reproducible (validation runs on the reference machine):
byte-identical for `reproduce-certified`, `reproduce-logit-trace`,
`reproduce-canonical-table`, `reproduce-jsma`; exact for
`reproduce-table-mixednorm`; within tolerance for the multiseed sweeps.

## Known gaps / honest limitations

- **CICIDS2017 / UNSW-NB15 preprocessing is not scripted.** Parquet
  representations are not regenerable from the shipped scripts; `dataset_links.md`
  documents the pipeline and pins the exact tensors via SHA-256.
- **One legacy NSL/UNSW mixed-norm file is stale.** Of the legacy per-dataset
  K=0/1 files produced by the pre-foolbox mixed-norm evaluator, all are full
  test-set runs except `results/results_unsw_nb15_Hardened.json`, which was a
  partial run (`total = 348,000`). It is not a source of any paper table; the
  UNSW-NB15 numbers the paper reports come from the foolbox multiseed sweeps
  (`results/foolbox/exh_k1_pgd40_*`, 27 unsw files, all `n_total = 508,010`)
  and the `consolidated/` and `unified/` tables, which use the full 508,010-row
  test set. The full-test-set twin is `results_unsw_nb15_Hardened_fgsm.json`.
- **`EPS`/attack constants** are defined inside the shipped scripts (mirroring
  `app/ml/attacks/eval_protocol.py`) so each script is self-contained.
- **Random-start attacks are unseeded** by design (faithful to the original
  pipeline); tolerances above were pre-registered before validation.
- **Certified sweeps** were budgeted to NSL-KDD (seeds 42–44) + CICIDS2017
  (seeds 42–44) + UNSW-NB15 (seeds 42–44) = 27 files; `cert_ibp_*_seed53.json`
  are auxiliary IBP-only records. The 31-file set is complete for Tables
  VIII–IX.

## Files omitted from the artifact

- Raw datasets (~700 MB, see `dataset_links.md` for integrity anchors).
- Per-sample attack mask `.pt` files (82 MB, regenerable by the corresponding
  `eval_deepfool_k1.py` run).
- Internal development summaries, logs, audit notes, and sibling-project code
  (not needed for evaluation; a couple of dev-only efficiency scripts not used
  by any table).