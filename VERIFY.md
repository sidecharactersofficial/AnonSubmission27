# Verification

This guide covers **fast verification** of the artifact without running
full reproduction experiments. Verification confirms that the provided files
support the paper's claims. Reproduction (`REPRODUCE.md`) covers end-to-end
regeneration.

## Prerequisites

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Verify Checkpoint Integrity

Confirm that every shipped checkpoint matches its pinned SHA-256 hash:

```bash
make verify
# or
python3 verification/verify_manifest.py
```

Expected output: `verify_manifest: PASS (104/104 checkpoints)`

## Step 2: Run Smoke Tests

Run the fastest verification targets to confirm the evaluation pipeline
executes correctly:

```bash
make smoke
```

This runs:
1. Checkpoint manifest verification
2. Mixed-norm canonical evaluation (NSL-KDD, all 4 models)
3. Logit-reversal trace
4. Canonical table aggregation

Expected runtime: ~2 minutes on CPU.

## Step 3: Inspect Canonical Results

Verify that canonical results are present and consistent:

```bash
# Check expected result directories exist
ls results/consolidated/
ls results/certificates/ | head
ls results/foolbox/ | head
ls results/section3/
ls results/logit_trace_model_adv_hardened.json
```

## Step 4: Regenerate Statistics

Regenerate the aggregated canonical table from archived results:

```bash
python3 canonical/consolidated_canonical_table.py
```

Expected output: Per-seed and mean±std tables for all 3 datasets × 3 methods
× canonical seeds. Values should match paper tables.

## Step 5: Verify Paper Tables (Time-Budgeted)

For reviewers with more time, verify individual experiments:

```bash
# Section III: Faithful AdvGuard diagnostic (~4 min CPU)
python3 canonical/section3_faithful_diagnostic.py
# Expected: baseline ~16%, hardened ~40% at eps=0.15

# Mixed-norm K=0/1/2 collapse (~26 s)
python3 canonical/run_paper_mixednorm.py --check
# Expected: ALL CHECKS PASS

# Logit reversal trace (~4 s)
python3 canonical/trace_logit_reversal_hardened.py
# Expected: byte-identical to archive
```

## Step 6: Compare Fresh vs Canonical (Optional)

If you run fresh evaluations, compare against the canonical archive:

```bash
# Run fresh EXH K=1 for NSL-KDD (subset)
python3 canonical/eval_deepfool_k1.py --attack pgd40 --dataset nsl-kdd --method hardened --seed 42 --suffix .fresh

# Compare
python3 verification/compare_exh_fresh.py --attack pgd40 --dataset nsl-kdd
# Expected: ALL PASS (within pre-registered tolerances)
```

## What Not to Expect

- **Exact byte-identical reproduction** for stochastic attacks (random-start
  PGD). Use tolerance-based comparison.
- **Full multi-seed sweep reproduction** in under 2 hours. NSL-KDD takes
  ~10 min/model, CICIDS2017 ~6 h/model, UNSW-NB15 ~9-10 h/model.
- **Dataset reconstruction** for CICIDS2017/UNSW-NB15 from raw sources.
  These require manual preprocessing documented in `dataset_links.md`.
