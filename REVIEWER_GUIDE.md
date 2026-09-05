# Reviewer Guide

This artifact accompanies the paper on adversarial robustness of tabular
intrusion detection systems under mixed-norm threat models. The repository
contains multiple independent evaluation components. Reviewers do not need
to inspect the repository linearly.

## Quick Navigation

| If you are reviewing... | Start here |
|---|---|
| Mixed-norm threat model | `docs/threat_model.md` |
| Canonical evaluation pipeline | `canonical/eval_deepfool_k1.py` |
| Mixed-norm enumeration | `canonical/eval_mixed_norm.py` |
| Certified robustness bounds | `canonical/run_crown_vs_ibp.py` |
| Independent attack validation | `canonical/eval_jsma_vs_exhaustive.py` |
| Reproduce paper tables | `canonical/consolidated_canonical_table.py` |
| Historical methodology changes | `PROVENANCE.md` |
| Known limitations | `LIMITATIONS.md` |
| Claim-to-evidence mapping | `CLAIM_MAP.md` |
| Review verification checklist | `REVIEW_CHECKLIST.md` |
| Fast verification | `VERIFY.md` |
| Full reproduction | `REPRODUCE.md` |

## Suggested Review Paths

### 15-minute scientific sanity check
1. `CLAIM_MAP.md` — understand the claim structure
2. `canonical/eval_mixed_norm.py` — inspect canonical enumerator
3. `VERIFY.md` — run `make smoke` for end-to-end check
4. `LIMITATIONS.md` — understand boundaries

### 30-minute methodology review
1. `docs/threat_model.md` — mixed-norm threat model
2. `docs/methodology.md` — evaluation conventions
3. `docs/adr/ADR-001-canonical-exhaustive-mixed-norm-evaluation.md` — why exhaustive enumeration
4. `canonical/eval_deepfool_k1.py` — canonical evaluator
5. `canonical/run_crown_vs_ibp.py` or `canonical/eval_jsma_vs_exhaustive.py` — independent validation

### 2-hour reproducibility review
1. `VERIFY.md` — checkpoint verification, smoke tests
2. `REPRODUCE.md` — full reproduction instructions
3. Run selected `reproduce-*` targets for time-budgeted experiments
4. `verification/compare_exh_fresh.py` — compare fresh vs canonical
5. `EXPECTED_OUTPUTS.md` — verify outputs match expectations

### 30-minute provenance review
1. `PROVENANCE.md` — methodological evolution
2. `docs/adr/` — decision rationale
3. `diagnostics/` — superseded implementations
4. Compare `canonical/eval_unified.py` (superseded) with `canonical/eval_deepfool_k1.py` (canonical)

## Dataset Caveats

- **NSL-KDD** can be regenerated from public sources using `data_scripts/download_data.py`.
- **CICIDS2017** and **UNSW-NB15** can be regenerated from public sources using
  `data_scripts/download_cicids2017.py` and `data_scripts/download_unsw_nb15.py`.
  The shipped, hash-verified parquet files are the authoritative source for exact
  reproduction.
- Reviewers with limited time can verify all claims using the shipped
  preprocessed tensors and checkpoints without re-running data pipelines.
