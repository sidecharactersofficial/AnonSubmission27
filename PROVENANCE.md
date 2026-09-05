# Provenance

This document records the methodological evolution of the evaluation
pipeline. Superseded implementations are retained for scientific provenance
and should not be interpreted as the canonical evaluation pipeline.

## Evolution Timeline

```
Initial Evaluation (AdvGuard Legacy)
       │
       ▼
DACM Hard Snapping
       │
       ├── Identified gradient masking concerns
       ├── Identified invalid-state artifacts during optimization
       ▼
Greedy Top-K Projection / One-Shot Evaluation
       │
       ├── Identified categorical oscillation
       ├── Identified suboptimal categorical flips
       ├── Retracted result: 29.10% robust accuracy
       ▼
Canonical Exhaustive Evaluation
       │
       ├── Statistical re-evaluation (multi-seed)
       ├── Independent attack validation (C&W, DeepFool, CROWN/IBP)
       ├── C&W survivor records archived
       ├── DeepFool EXH K=1 records archived
       ▼
Final Evaluation Pipeline
```

## Superseded Components

### `canonical/eval_unified.py` (Superseded)

**Original role**: Section VII-B masking-gap analysis. One-shot gradient-
snapped K=1 evaluation.

**Why superseded**: Uses one-shot categorical projection (argmax at end of
PGD) rather than exhaustive enumeration. Does not guarantee coverage of all
valid categorical states. Produces optimistic robustness estimates due to
greedy projection artifacts.

**Replacement**: `canonical/eval_deepfool_k1.py` (exhaustive enumeration)

**Archive**: Results in `results/unified/` are retained for provenance but
are not used in canonical claims.

### `diagnostics/train_pgd_robust.py` (Superseded)

**Original role**: Legacy adversarial training with curriculum alpha_cat
schedule. Produced `models/model_adv_pgd_curriculum_*.pth`.

**Why superseded**: Training-time PGD uses steps=10 and legacy attack
implementation (`app/ml/attacks/pgd.py`) rather than the canonical unified
PGD (`app/ml/attacks/unified_pgd.py`). The legacy trainer is retained to
explain the origin of shipped `model_adv_pgd_curriculum_*.pth` files.

**Replacement**: `canonical/train_unified.py` (PGD-10, mixed loss 0.5/0.5,
curriculum alpha_cat, unified PGD)

**Archive**: Shipped checkpoints `models/model_adv_pgd_curriculum_*.pth` are
retained for evaluation but were produced by the superseded trainer.

### `app/ml/evaluation/` (Legacy)

**Original role**: Base evaluators for clean, FGSM, PGD, JSMA, and ensemble
accuracy.

**Why superseded**: These evaluators are not used by any canonical script.
Canonical evaluators implement their own evaluation loops directly. The
legacy evaluators are retained because they are imported by
`app/ml/utils/robustness_curve.py` and `app/ml/utils/test_model.py`, which
are used for development smoke tests.

**Replacement**: Inline evaluation in canonical scripts

### `app/ml/attacks/pgd.py` (Legacy)

**Original role**: Training-time PGD attack used by `diagnostics/train_pgd_robust.py`.

**Why superseded**: Uses `alpha_cat` parameter with DACM snapping. Legacy
implementation does not raise when `steps` is None, which can mask budget
drift.

**Replacement**: `app/ml/attacks/unified_pgd.py` (raises if `steps` is None,
supports RSC, unified interface)

## Archived Results

Some result files are archived but superseded:

- `results/results_*.json` — Legacy per-dataset mixed-norm results produced
  by pre-foolbox evaluators. All are full test-set runs except
  `results/results_unsw_nb15_Hardened.json` (partial run, total=348,000).
  Not used in any paper table.
- `results/unified/` — Superseded masking-gap analysis. Retained for
  provenance. Not used in canonical claims.

## Decision Records

See `docs/adr/` for the full ADR history:
- ADR-001: Canonical exhaustive mixed-norm evaluation
- ADR-002: Multi-seed statistical re-evaluation
- ADR-003: Independent attack validation
- ADR-004: Canonical vs fresh reproduction outputs
