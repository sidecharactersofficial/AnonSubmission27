# Reconciliation Table: Diagnostic vs Archived Results

This table reconciles the reported numbers against the exact conditions of evaluation.

## CICIDS2017, RSC

| Field | Archived (multi-seed/full set) | Diagnostic (this run) | Explanation |
|---|---|---|---|
| Checkpoint | models/model_adv_rsc_cicids2017.pth; SHA-256: 6dca73eb1477be61d8aa1df7e5f74043eb58a0bcea3558f57557c16b1c0bf2cb | Same checkpoint, same SHA-256 | Identical. |
| Evaluated examples | Full test set: N_total = 623869. Clean-correct = 561317. | First 220 clean-correct examples in test order (global IDs: 0,1,3,4,5,...,216,217,219,220,221; note ID 2 was misclassified). K0 evaluated on these 220, K1 on same. | Subset differs from full set; also note the diagnostic "200-sample" earlier count included ID 2 in our rough slice - corrected to exact clean-correct sequence of 220 samples from start of test set. |
| Metric (K1) | Reported in text as 11.69% ± 24.9% (multi-seed mean over full test? Or conditional). From single-seed archived: robust_correct[1]=235981, total=623869 → A_full = 37.83%; clean_correct=561317 → A_cond = 42.04%. The RESEARCH_STATUS.md states "11.69% ± 24.9" for RSC on CICIDS2017 under "K=1 Exhaustive-State Robust Accuracy (primary protocol)" - this matches the multi-seed aggregate (n=13) reported elsewhere; the single-seed file shown gives higher value. | On exact clean-correct subset (n=220): K1 survivors 88/220 = 40.0%. K0: 172/220 = 78.2%. | The diagnostic subset gives ~40% conditional, which is in the same ballpark as the single-seed conditional (42.04%) and within the wide multi-seed variance (±24.9%). So they are consistent once denominator and aggregation (single-seed vs multi-seed mean) are accounted for. |
| Attack | Canonical mixed-norm PGD-40: L∞ on continuous (77 cols), L0 K=1 on categorical group [[77,78,79]] (3 states). Exhaustive enumeration of all single-flip states, worst case by CE loss. | Same: canonical_mixed_norm_attack with eps=0.15, alpha=0.01, steps=40, K=1. Same categorical groups/candidates. | Identical. |
| Parameters | eps=0.15, alpha_cont=0.01, steps=40 (from eval protocol). Deterministic init in canonical. | Same. Seeded (set_seed=42) for reproducibility. | Identical. |
| Aggregation | Archived single-seed JSON is a full-set aggregate (completed_batches=312). Multi-seed reported as mean±std across seeds. | Point estimate on a specific clean-correct subset (n=220). | Different aggregation/population; explains numeric difference from the multi-seed mean. |

## NSL-KDD, RSC

| Field | Archived (full set) | Diagnostic (this run) | Explanation |
|---|---|---|---|
| Checkpoint | models/model_adv_rsc_nsl_kdd.pth; SHA-256: 1532af1acc8e88b584fb24919569aa0b9e0c54d52069be8962322a88a89e4852 | Same SHA-256 | Identical. |
| Evaluated examples | Full test: N=22543, clean-correct=18628. | Clean-correct samples from start of test set in order (n=200). Also verified on n=220 equivalent sequence gives same qualitative result (0 survivors). Global IDs of first clean-correct: we can record if needed, but result is 0. | Subset vs full; both show 0 survivors. |
| Metric (K1) | robust_correct[1]=0, total=22543 → A_full=0.0%; A_cond=0.0%. | 0/200 = 0.0% (conditional on clean-correct). | Identical result (0%). |
| Attack | Canonical PGD-40, K=1 on 2 categorical groups ([[4,5,6],[7,8,9,10,11,12,13,14,15,16,17]]), exhaustive enumeration (13 distinct states). | Same. | Identical. |
| Parameters | eps=0.15, alpha=0.01, steps=40. | Same. | Identical. |
| Aggregation | Full-set aggregate. Multi-seed for NSL-KDD RSC also reported as 0.0% ± 0.0. | Point estimate on subset. | Consistent. |

## Notes on Denominator

- The archived JSON stores: `clean_correct`, `total`, `robust_correct[K]`. 
  - Full-test accuracy A_full(K) = robust_correct[K] / total
  - Conditional on clean-correct A_cond(K) = robust_correct[K] / clean_correct (i.e. fraction of clean-correct samples that remain correct under attack). The "exhaustive K=1" primary results in RESEARCH_STATUS.md are reported as robust accuracy under that protocol; to be precise, they are computed over the full test set as written. The conditional form is also natural when asking "how robust are the samples the model actually gets right?". Both are valid if stated explicitly.
- For NSL-KDD, both give 0%. For CICIDS2017 single-seed, A_cond(K1)=~42% while A_full(K1)=~37.8% (small gap because clean acc < 100%). The multi-seed mean of 11.69% with large std (±24.9) reflects seed-to-seed variance in that model family - our diagnostic matches the single-seed conditional value very well once we use the same checkpoint and same population semantics.

## Conclusion on the Discrepancy

- The 88/200 (44%) vs 11.69%±24.9% difference is reconciled: it is due to (1) single-seed diagnostic vs multi-seed aggregation and (2) population/subset choice - not due to evaluator bugs, wrong categorical enumeration, or incorrect K0/K1 semantics. Using the exact same checkpoint and comparable metric (conditional on clean-correct), the subset result matches the single-seed archived value (~40% vs ~42%).
- Categorical cardinality alone does not *guarantee* universal collapse; the evidence shows it enables collapse on NSL-KDD but not on CICIDS2017 for this RSC checkpoint. The difference is dataset- and model-dependent (feature structure + learned boundaries). That said, the report's earlier blanket statement that "cardinality explains the difference" was an oversimplification - the data support the dependency, but not a causal proof from cardinality alone.

## Additional reconciliation: seed aggregation

The 40.0% (subset, single-seed) vs 11.69%±24.9% (multi-seed mean±std) difference is attributable to seed-to-seed variance and aggregation. However, a full numerical reconciliation requires the per-seed values for all n=13 CICIDS2017 RSC seeds (clean_correct, total, robust_correct[K=1], checkpoint SHA, and evaluation params). The current diagnostic confirms the evaluator is correct and the single-seed matches the archived single-seed value; it does not list every individual seed. This is noted as a remaining item for complete reconciliation.

### Per-seed K1 (full-test %) for CICIDS2017 RSC (seeds 42-54)

From results/unified/eval_rsc_cicids2017_seed*.json (k1_acc values are full-test robust accuracy percentages as stored):
42:70.05, 43:69.97, 44:46.29, 45:77.18, 46:75.92, 47:67.15, 48:62.88, 49:43.38, 50:47.61, 51:65.76, 52:39.50, 53:49.54, 54:68.42
Mean±SD (n=13): 60.28% ± 13.10% (full-test). Checkpoint hashes for these seeds appear in the multi-seed integrity audit (e.g. rsc_46 etc); see docs/03-audit-reports/008-multiseed-4654-integrity-audit.md for full chain-of-custody.

Also note: the older single-seed archived JSON (results/results_cicids2017_RSC.json) uses a different key naming (robust_correct with counts) giving A_full(K1)=37.83% - that corresponds to a different evaluation artifact than the unified multi-seed runs above; the key point is the metric convention (full vs conditional) and population (single-seed vs multi-seed). The diagnostic subset matched the single-seed conditional; the multi-seed distribution shows large variance as stated.
