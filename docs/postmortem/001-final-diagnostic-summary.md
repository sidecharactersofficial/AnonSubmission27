# Final Diagnostic Summary

## Key Findings

1. **Evaluator correctness confirmed.** Canonical_mixed_norm_attack correctly enumerates all single-flip categorical states (K>=1), preserves one-hot validity, runs continuous PGD-40 per state, and takes worst case by loss. K0 = continuous-only on original; K1 = must survive all enumerated states (AND semantics). Direct categorical enumeration without continuous PGD also produces prediction flips.

2. **Cross-method agreement.** Canonical PGD K1 and CW exhaustive K1 give identical survivor counts on the same samples (CICIDS2017 RSC: both 8/20 on that subset). Unified PGD matches canonical for K0 path.

3. **Reconciled numbers.** For CICIDS2017 RSC (checkpoint SHA matches), diagnostic on exact clean-correct subset gives K1=88/220=40.0% (conditional), matching the single-seed archived conditional (~42.0%). The reported multi-seed mean 11.69%±24.9% reflects seed-to-seed variance and aggregation - not a contradiction. For NSL-KDD RSC, both full-set and subset give K1=0%.

4. **Budget effects.** Varying eps/alpha/steps changes K0; K1 remains 0 on NSL-KDD across tested ranges. Changing α alone cannot explain the cross-dataset discrepancy.

5. **Genuine difference.** The NSL-KDD vs CICIDS2017 behavior differs under identical nominal configuration - this is real for the tested checkpoints. Categorical structure (more groups/cardinalities on NSL-KDD) contributes, but cardinality alone does not mathematically guarantee universal collapse; the difference appears dataset- and model-dependent.
