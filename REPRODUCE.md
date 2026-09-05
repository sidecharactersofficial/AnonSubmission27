# Reproduction

This guide covers **full reproduction** of the artifact's experiments from
scratch. For fast verification without re-running experiments, see `VERIFY.md`.

## Prerequisites

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Acquisition

### NSL-KDD (Regenerable)

```bash
python3 data_scripts/download_data.py
```

This fetches raw NSL-KDD CSVs from the public repository, applies
preprocessing, and writes `data/nsl-kdd-train.csv` and `data/nsl-kdd-test.csv`.
SHA-256 hashes are pinned in `manifest/dataset_sha256.txt`.

### CICIDS2017 and UNSW-NB15

```bash
python3 data_scripts/download_cicids2017.py
python3 data_scripts/download_unsw_nb15.py
```

These scripts download the raw data and apply the canonical preprocessing
pipeline. The shipped, hash-verified parquet files are the authoritative source
for exact reproduction; from-scratch regeneration may produce slightly different
hashes due to public mirror variations or category frequency differences.

## Model Training

Canonical models are shipped in `models/` and `models/unified/`. Training
from scratch is optional and time-consuming.

```bash
# Canonical unified training (PGD-10, mixed loss, curriculum)
python3 canonical/train_unified.py --dataset nsl-kdd
python3 canonical/train_unified.py --dataset cicids2017
python3 canonical/train_unified.py --dataset unsw-nb15
```

Expected runtime: ~50 min per dataset on CUDA, longer on CPU.

## Experiment Reproduction

### Table 3: Faithful AdvGuard Diagnostic

```bash
make reproduce-table3
```

Expected runtime: ~45 s (CUDA) / ~4 min (CPU).

Expected output: `results/section3/faithful_diagnostic_nsl_kdd.json`

Expected property: Hardened robust accuracy ≈ 40 ± 0.4% at eps=0.15.

### Mixed-Norm K=0/1/2 Collapse

```bash
make reproduce-table-mixednorm
```

Expected runtime: ~26 s.

Expected output: `results/results_nsl_kdd_Baseline.json`,
`results/results_nsl_kdd_Curriculum.json`, etc.

Expected property: ALL CHECKS PASS against archived values.

### Multi-Seed EXH K=1 Sweeps

```bash
make reproduce-multiseed-nslkdd
make reproduce-multiseed-cicids
make reproduce-multiseed-unsw
```

Expected runtime:
- NSL-KDD: ~10 min
- CICIDS2017: ~6 h
- UNSW-NB15: ~9-10 h

Expected output: `results/foolbox/exh_k1_pgd40_*.json` (fresh files with
`.fresh.json` suffix).

### Certified Bounds (CROWN vs IBP)

```bash
make reproduce-certified
```

Expected runtime: ~12-15 min for all 27 checkpoints.

Expected output: `results/certificates/cert_ibp_*.json`,
`results/certificates/crown_vs_ibp_*.json`

Expected property: Byte-identical to archive.

### Scalability

```bash
make reproduce-scalability
```

Expected runtime: ~2 min (CUDA) / ~10-15 min (CPU).

Expected output: `results/scalability_unsw_nb15_*.json`

### JSMA vs Exhaustive

```bash
make reproduce-jsma
```

Expected runtime: ~1 min.

Expected output: `results/jsma_vs_exhaustive_unsw_nb15_K1.json`

### Logit Reversal Trace

```bash
make reproduce-logit-trace
```

Expected runtime: ~4 s.

Expected output: `results/logit_trace_model_adv_hardened.json`

Expected property: Byte-identical to archive.

### Canonical Table Aggregation

```bash
make reproduce-canonical-table
```

Expected runtime: <1 s.

Expected output: `results/consolidated/canonical_both_conventions.json`

## Full Reproduction

```bash
make reproduce-all
```

This runs all experiments except the CICIDS2017 and UNSW-NB15 multiseed
sweeps (which are time-budgeted separately).

## Cleanup

```bash
make clean
```

Removes fresh outputs (`*.fresh.json`, `*.repro.json`, `results_fresh/`)
but preserves canonical archives.
