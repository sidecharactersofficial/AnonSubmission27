# Postmortem: Evaluation Diagnostics and Audits (October 2026)

This directory holds the post-evaluation diagnostic and audit reports produced
after the artifact was frozen. They verify the correctness of the canonical
K=0/K=1 evaluator, reconcile the diagnostic numbers with archived results,
clarify the CICIDS2017 categorical threat model, and record the findings of
the coverage and representation-sensitivity audits.

## Index

| ID | Document | Scope |
|---|---|---|
| 001 | `001-final-diagnostic-summary.md` | Executive summary of the diagnostic findings |
| 002 | `002-evaluator-correctness-and-attack-budget-diagnostic.md` | K=0/K=1 evaluation path audit and attack-budget analysis |
| 003 | `003-reconciliation-diagnostic-vs-archived-results.md` | Reconciles diagnostic numbers against archived results (signals: seed variance, aggregation, and population semantics) |
| 004 | `004-threat-model-clarification-cicids2017.md` | CICIDS2017 categorical group mapping and K=1 implications |
| 005 | `005-audits-1-3-feature-category-semantic-coverage.md` | Audits of feature coverage, category coverage, and semantic validity |
| 006 | `006-audit-4-representation-sensitivity.md` | Representation-sensitivity audit (known limitation, see `LIMITATIONS.md`) |
| 007 | `007-exh-vs-snap-canonical-comparison.md` | EXH vs SNAP canonical comparison across 75 checkpoints, plus staged K-budget evaluation on UNSW-NB15 |

## Note on paths

These reports were written against the companion evaluation workspace, so
some paths they cite (e.g., `scripts/eval_mixed_norm.py::canonical_mixed_norm_attack`,
`ml-optimizer/...`) should be read as referring to the corresponding artifact
members (`canonical/eval_mixed_norm.py`, shipped checkpointer under `models/`,
etc.). Reported checkpoint SHAs are the same pins used by
`manifest/checkpoint_sha256.json`.

## Headline findings

- **Evaluator correctness confirmed**: full single-flip categorical
  enumeration, one-hot validity preserved, AND-semantics for K=1, and
  cross-method agreement (canonical PGD K=1 vs. C&W exhaustive K=1).
- **Archived numbers reconcile**: the apparent 11.69% ± 24.9% vs. ~40% gap on
  CICIDS2017 RSC K=1 is explained by single-seed vs. multi-seed aggregation
  and population/denominator semantics — not evaluator bugs.
- **Cross-dataset inconsistency is genuine**: NSL-KDD collapses to 0%
  K=1 robustness while CICIDS2017 retains ~40%; this is dataset- and
  model-dependent, not an artifact of attack budget settings.
- **Coverage audits pass** (features, categories, semantic validity); the
  representation-sensitivity audit is a documented open limitation.
- **EXH vs SNAP gap quantified**: 66/75 checkpoints show SNAP overestimating
  K=1 robustness (mean 41.9pp among non-zero gaps; zero gap only on NSL-KDD
  where both agree at 0%); K=0 matches exactly under controlled init.
  Staged K=2 evaluation shows UNSW-NB15's residual robustness is largely a
  K=1 threat-model incompleteness artifact.
