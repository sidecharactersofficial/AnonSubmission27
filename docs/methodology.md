# Methodology

## Overview

This artifact evaluates adversarial robustness of tabular intrusion detection
models under a mixed-norm threat model. The methodology has three layers:

1. **Defense** — Adversarial training with PGD-10 and mixed loss
2. **Attack** — Mixed-norm PGD-40 with DACM snapping and exhaustive categorical enumeration
3. **Evaluation** — Multi-seed statistical validation with independent attack verification

## Training

Models are trained using Min-Max adversarial training:

```
loss = 0.5 * CE(model(x), y) + 0.5 * CE(model(PGD(x, y)), y)
```

Training uses PGD-10 with epsilon schedule (0.02 to 0.15 over first 10 epochs)
and curriculum categorical attack strength (alpha_cat ramps from 0.01 to 1.0
over epochs 10-30).

See `canonical/train_unified.py` for the canonical training pipeline.

## Attack Budgets

All evaluation-time attacks use the protocol defined in
`app/ml/attacks/eval_protocol.py`:

```python
EVAL_EPSILON = 0.15
EVAL_ALPHA_CONT = 0.01
EVAL_PGD_STEPS = 40
```

These are fixed across all canonical evaluations. Legacy training-time attacks
used different budgets (PGD-10, alpha_cat schedule) and are documented in
`diagnostics/train_pgd_robust.py`.

## Mixed-Norm Enumeration

For categorical budget $K=1$, the canonical evaluator evaluates:

1. The base continuous-only attack (K=0)
2. Every valid one-hot categorical state for each categorical group

For each state, continuous PGD-40 is run from a random start. The worst
state (highest cross-entropy loss) determines whether the sample is
classified as robust.

For $K=2$, all pairs of categorical groups are enumerated.

## Statistical Validation

Single-run adversarial evaluations are stochastic due to random PGD
initialization. The artifact uses multi-seed evaluation (seeds 42-44 for
NSL-KDD, 42-54 for CICIDS2017, 42-51 for UNSW-NB15) to compute mean±std
robustness metrics.

Tolerances for reproduction are pre-registered in
`verification/compare_exh_fresh.py`.

## Independent Validation

Findings are validated using:
- **CROWN/IBP certified bounds** — backward/forward bound propagation
- **C&W L2 attack** — Carlini-Wagner optimization in mixed-norm setting
- **JSMA comparison** — group-level saliency map attack (for divergence analysis only)

These independent methods confirm that exhaustive PGD-40 findings are not
attack-specific artifacts.

## Soundness Checks

CROWN vs IBP includes a soundness check:
```
assert crown_k1 <= ibp_k1 + 1e-6
```

This ensures backward bounds do not exceed forward bounds, which would
indicate an implementation bug.
