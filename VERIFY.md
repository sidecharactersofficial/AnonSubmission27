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

## Step 2: Verify Formal Mathematical Proofs (Lean 4 & Python)

Confirm that all theoretical foundations, bounds, and complexity counts established in the paper are sound and machine-checked:

```bash
# 1. Automated Python numerical verification suite (<1s)
make verify-math
# or
python3 leanproofs/verify_math.py

# 2. Formal machine-checked proofs in Lean 4
make verify-lean
# or
cd leanproofs && lake build && ./.lake/build/bin/blackice_proofs
```

Expected output:
- `ALL MATHEMATICAL PROOFS AND PROPOSITIONS SUCCESSFULLY VERIFIED!` (6/6 test suites pass)
- Lean 4: `Build completed successfully (8 jobs).` All 6 core theorems/propositions verified.

Detailed step-by-step mathematical expansions, Taylor remainders, and hypergeometric derivations are documented in `leanproofs/MATH_AUDIT_REPORT.md`.

## Step 3: Run Smoke Tests

Run the fastest verification targets to confirm the evaluation pipeline
executes correctly:

```bash
make smoke
```

This runs:
1. Checkpoint manifest verification
2. Mathematical verification test suite (all 6 theorems)
3. Mixed-norm canonical evaluation (NSL-KDD, all 4 models)
4. Logit-reversal trace
5. Canonical table aggregation

Expected runtime: ~2 minutes on CPU.

## Step 4: Inspect Canonical Results

Verify that canonical results are present and consistent:

```bash
# Check expected result directories exist
ls results/consolidated/
ls results/certificates/ | head
ls results/foolbox/ | head
ls results/section3/
ls results/logit_trace_model_adv_hardened.json
```

## Step 5: Regenerate Statistics

Regenerate the aggregated canonical table from archived results:

```bash
python3 canonical/consolidated_canonical_table.py
```

Expected output: Per-seed and mean±std tables for all 3 datasets × 3 methods
× canonical seeds. Values should match paper tables.

## Step 6: Verify Paper Tables (Time-Budgeted)

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

## Step 7: Compare Fresh vs Canonical (Optional)

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
- **Anything under `replication/`**. The post-freeze replication addendum
  (NF-ToN-IoT, CAA+MOEVA, Illusion/Takedown — see `replication/README.md` and
  `docs/postmortem/008`) is outside the verification surface: `make verify`
  checks the 104 canonical checkpoints only, and `make smoke` exercises no
  addendum code. The two supplementary checkpoints in
  `models/unified/model_adv_lite_nf_ton_iot*.pth` are intentionally not in
  the canonical manifest.
