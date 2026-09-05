# Known Limitations

This artifact is a research prototype. It does not provide production-grade
guarantees. Reviewers should understand these boundaries before interpreting
results.

## Dataset Reconstruction

**Status**: Known limitation.

**Description**: Processed tensor representations for CICIDS2017 and
UNSW-NB15 can be regenerated from raw public sources using the shipped
`data_scripts/download_cicids2017.py` and `data_scripts/download_unsw_nb15.py`
scripts. However, the output hashes may differ from the shipped manifests if
public mirrors change or category frequencies differ. The shipped parquet files
remain the authoritative source for exact reproduction.

**Impact**:
- Does not affect verification of shipped experiments using pinned checkpoints
  and manifests.
- From-scratch regeneration produces structurally compatible data (same feature
  layout and dimensions) but may not be byte-identical to shipped archives.
- Reviewers who need exact numerical reproduction should use the shipped
  preprocessed tensors.

**Mitigation**: SHA-256 hashes in `manifest/dataset_sha256.txt` pin the exact
tensors consumed by all canonical evaluations.

## Scalability

**Status**: Inherent limitation.

**Description**: Exhaustive categorical enumeration is tractable for the
evaluated state spaces but scales combinatorially with the categorical budget
$K$ and the number of categorical groups.

For UNSW-NB15 (5 categorical groups):
- K=1: 42 states per sample
- K=2: 667 states per sample
- K=3: 5,852 states per sample

**Impact**: Full K=2 exhaustive evaluation on UNSW-NB15 requires ~10-15 min
on CPU and ~2 min on reference GPU. K=3 is not evaluated in the current
artifact.

**Mitigation**: The canonical artifact evaluates K=1 for multi-seed sweeps
and K=2 for scalability characterization only.

## Continuous Optimization

**Status**: Methodological boundary.

**Description**: Exhaustive enumeration guarantees coverage of the discrete
categorical state space. The continuous inner optimization remains dependent
on the configured attack procedure (PGD-40 with random start) and does not
constitute a global certificate unless explicitly stated.

Certified bounds (`canonical/crown_bound.py`, `canonical/certified_bound.py`)
provide architecture-level lower bounds, but these bounds are loose and
conservative compared to empirical PGD-40 results.

**Impact**: Robustness claims are empirical lower bounds, not global
certificates of robustness.

**Mitigation**: Independent attack validation (C&W, DeepFool) and certified
bounds provide converging evidence but do not eliminate this limitation.

## Random-Start Attacks

**Status**: Design choice with reproducibility impact.

**Description**: Canonical EXH K=1 sweeps use unseeded random-start PGD-40.
This is intentional: it matches the original evaluation protocol and
produces realistic robustness estimates. However, it introduces run-to-run
variance.

**Impact**:
- `k0_survivors` and `k1_survivors` can vary by ±0.75% relative across runs
- Exact byte-identical reproduction is impossible for these quantities
- Reviewers must use tolerance-based comparison, not exact comparison

**Mitigation**: Pre-registered tolerances in `verification/compare_exh_fresh.py`
and multi-seed statistical reporting (mean±std) quantify this variance.

## Legacy Model Origins

**Status**: Historical artifact.

**Description**: Some shipped checkpoints (e.g., `models/model_adv_pgd_curriculum_*.pth`)
were produced by superseded training scripts (`diagnostics/train_pgd_robust.py`).
These checkpoints are retained for backward compatibility with archived
results but were not produced by the canonical training pipeline.

**Impact**: Does not affect evaluation of canonical claims, but may confuse
reviewers who inspect training scripts.

**Mitigation**: `PROVENANCE.md` documents the origin of each checkpoint
family. `canonical/train_unified.py` is the canonical trainer.

## JSMA Suitability

**Status**: Known limitation.

**Description**: JSMA is included for completeness and divergence analysis
only. On UNSW-NB15, JSMA misses 12.0% of true adversarial examples that
exhaustive K=1 correctly identifies, and is 10x slower than PGD. The
non-gradient-based iteration rule does not satisfy the same Lipschitz
convergence guarantees as PGD.

**Impact**: JSMA should not be used for robustness claims. It is retained
only to demonstrate the failure modes of greedy projection.

**Mitigation**: `canonical/eval_jsma_vs_exhaustive.py` explicitly documents
this divergence.
