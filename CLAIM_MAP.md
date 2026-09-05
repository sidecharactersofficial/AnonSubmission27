# Scientific Claim Map

Stable identifiers for every scientific claim in the artifact. Each entry
links a paper claim to its evidence, generating script, and output file.

| ID | Paper Claim | Manifest | Evidence | Script | Output |
|---|---|---|---|---|---|
| C1 | Legacy one-shot gradient-snapped evaluation produces unreliable robustness estimates | `experiment_manifests/E-C1-LEGACY-DIAGNOSTIC.json` | Faithful reproduction of original AdvGuard protocol yields 16.06% baseline / 40.36% hardened vs. retracted 29.10% | `canonical/section3_faithful_diagnostic.py` | `results/section3/faithful_diagnostic_nsl_kdd.json` |
| C2 | Exhaustive discrete enumeration avoids invalid categorical states | `experiment_manifests/E-C2-EXHAUSTIVE-MIXEDNORM.json` | Mixed-norm K=0/1/2 counts show zero invalid states; all survivors respect exactly K categorical changes | `canonical/eval_mixed_norm.py` | `results/results_nsl_kdd_*.json` |
| C3 | Canonical exhaustive K=1 evaluation accurately measures robustness | `experiment_manifests/E-C3-EXHAUSTIVE-K1.json` | Multi-seed EXH K=1 sweeps across 3 datasets (75 models) with pre-registered tolerances | `canonical/eval_deepfool_k1.py` | `results/foolbox/exh_k1_*.json` |
| C4 | Greedy projection introduces optimization artifacts | `experiment_manifests/E-C4-JSMA-COMPARISON.json` | JSMA misses 12.0% of true adversarial examples that exhaustive K=1 identifies; JSMA is 10x slower | `canonical/eval_jsma_vs_exhaustive.py` | `results/jsma_vs_exhaustive_unsw_nb15_K1.json` |
| C5 | Observed differences survive statistical testing | `experiment_manifests/E-C5-MULTISEED-STATS.json` | Multi-seed mean±std across NSL-KDD (seeds 42-44), CICIDS2017 (seeds 42-54), UNSW-NB15 (seeds 42-51) | `canonical/consolidated_canonical_table.py` | `results/consolidated/canonical_both_conventions.json` |
| C6 | Findings are not attack-specific | `experiment_manifests/E-C6-CERTIFIED-BOUNDS.json` | CROWN/IBP certified bounds align with empirical PGD-40 exhaustive masks; C&W L2 and DeepFool L2 agree within tolerance | `canonical/run_crown_vs_ibp.py`, `app/ml/attacks/cw.py` | `results/certificates/cert_ibp_*.json`, `results/foolbox/exh_k1_deepfool_*.json` |
| C7 | Legacy FGSM-hardened models exhibit logit-reversal behavior | `experiment_manifests/E-C7-LOGIT-TRACE.json` | Per-sample logit margins flip under exhaustive K=1 categorical state changes | `canonical/trace_logit_reversal_hardened.py` | `results/logit_trace_model_adv_hardened.json` |
| C8 | Exhaustive evaluation is tractable for evaluated state spaces | `experiment_manifests/E-C8-SCALABILITY.json` | Wall-clock timing: K=1 (42 states) 0.0155 s/state, K=2 (667 states) 0.118 s/state on reference GPU | `canonical/eval_scalability.py` | `results/scalability_unsw_nb15_*.json` |

## Claim Dependencies

```
C1 (faithful diagnostic)
    └── identifies retracted result
        └── motivates C2 (exhaustive enumeration)
            └── enables C3 (accurate measurement)
                ├── validated by C4 (JSMA divergence)
                ├── validated by C6 (independent attacks)
                ├── confirmed by C7 (logit reversal)
                └── scaled by C8 (tractability)
C5 (statistical) validates C3 across seeds
```

## Cross-References

- **Threat model**: `docs/threat_model.md`
- **Methodology**: `docs/methodology.md`
- **ADRs**: `docs/adr/`
- **Provenance**: `PROVENANCE.md`
- **Limitations**: `LIMITATIONS.md`
