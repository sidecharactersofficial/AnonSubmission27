# Adversarial Constraint Mapping Engine (DACM)

This artifact implements a mixed-norm threat model for tabular intrusion
detection systems. The threat model assumes an adversary who can perturb:

1. **Continuous features** — bounded by $L_\infty$ norm with $\epsilon = 0.15$
2. **Categorical features** — bounded by a categorical budget $K$ (number of
   categorical groups flipped per sample)

The adversary may change any continuous feature within $[-\epsilon, \epsilon]$
and flip at most $K$ categorical groups to any valid one-hot state. This is
the **mixed-norm** threat model: continuous perturbations are additive in
$L_\infty$, categorical perturbations are discrete state transitions.

## Constraint Representation

Continuous and categorical features are concatenated into a single input
vector. Categorical features are one-hot encoded. DACM enforces constraints
by snapping perturbed inputs back to valid vertices:

- **Continuous snapping**: $\text{clamp}(x + \eta, 0, 1)$ where $\|\eta\|_\infty \le \epsilon$
- **Categorical snapping**: $\text{argmax}(\tilde{x}_{\text{cat}}) \to \text{one\_hot}$

This is implemented in `app/ml/attacks/unified_pgd.py` and
`app/ml/attacks/pgd.py`.

## Evaluation Protocols

Two evaluation conventions are present in the artifact:

**SNAP (gradient-snapped K=1)** — Single best-by-gradient flip, applied once
at the end. Used by the legacy one-shot evaluator (`diagnostics/`,
`canonical/eval_unified.py`). This is superseded.

**EXH (exhaustive K=1)** — Enumerate all valid one-hot categorical states,
run continuous PGD on each, pick the worst state. This is the canonical
evaluation protocol (`canonical/eval_deepfool_k1.py`,
`canonical/eval_mixed_norm.py`).

## Why Exhaustive Enumeration

Greedy gradient-based categorical projection can:
- Select a categorical flip that is locally optimal but globally suboptimal
- Produce invalid intermediate states during optimization
- Miss adversarial examples that require a specific categorical state

Exhaustive enumeration eliminates these failure modes by evaluating every
valid categorical configuration within budget $K$. The continuous inner
optimization remains gradient-based, but the discrete outer loop is exact.

## Datasets

All datasets are tabular intrusion detection benchmarks with mixed continuous
and categorical features:

- **NSL-KDD** — 22,543 test samples, 18 features (4 continuous, 2 categorical groups)
- **CICIDS2017** — 623,869 test samples, 80 features (77 continuous, 1 categorical group)
- **UNSW-NB15** — 508,010 test samples, 79 features (38 continuous, 5 categorical groups)

Preprocessed tensor SHAs are pinned in `manifest/dataset_sha256.txt`.
Raw dataset acquisition is documented in `dataset_links.md`.
