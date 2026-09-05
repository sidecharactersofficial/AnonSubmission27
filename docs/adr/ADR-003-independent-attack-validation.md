# ADR-003: Independent Attack Validation

## Status

Accepted

## Context

Robustness claims based on a single attack family risk being artifacts of
that attack's optimization landscape. An attack may fail to find adversarial
examples not because they do not exist, but because of local optima,
gradient masking, or implementation quirks.

The original artifact relied primarily on PGD-based evaluation. While PGD is
a standard first-order attack, it is not exhaustive and does not provide
certificates.

## Decision

Validate canonical PGD-40 findings using **independent attack methods**:

1. **C&W L2** (`app/ml/attacks/cw.py`) — second-order optimization in
   mixed-norm setting. Enumerates categorical states, optimizes continuous
   features in tanh-space. Used as an independent empirical check.
2. **DeepFool L2** (`canonical/eval_foolbox.py`) — Foolbox LinfDeepFool
   with exhaustive categorical enumeration. 47 DeepFool EXH K=1 records
   archived in `results/foolbox/exh_k1_deepfool_*.json`.
3. **CROWN/IBP certified bounds** (`canonical/crown_bound.py`,
   `canonical/certified_bound.py`) — backward and forward bound propagation
   through the exact network architecture. Provides deterministic lower
   bounds on robustness without relying on attack optimization.

## Validation Criteria

For a finding to be canonical:
- Empirical PGD-40 exhaustive results must agree with C&W within tolerance
- Certified IBP bounds must be inside the empirical PGD-40 exhaustive mask
- CROWN backward bounds must be $\le$ IBP forward bounds (soundness check)

## Consequences

### Positive
- Reduces risk that findings are attack-specific artifacts
- Certified bounds provide deterministic, architecture-level guarantees
- DeepFool and C&W explore different optimization landscapes

### Negative
- Increases artifact size (47 additional DeepFool result files)
- Certified bound computation is complex and requires careful architecture
   matching
- C&W optimization is slower than PGD and requires more tuning

### Neutral
- JSMA (`canonical/eval_jsma_vs_exhaustive.py`) is included for divergence
  analysis, not as a primary validator. JSMA misses 12.0% of true
  adversarial examples on UNSW-NB15 and is 10x slower than PGD.
