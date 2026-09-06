# Reviewer Verification Checklist

Use this checklist to verify the artifact systematically. Each section maps
to a claim or reproducibility requirement.

## Environment Setup
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Checkpoint manifest passes (`make verify`)
- [ ] Smoke tests pass (`make smoke`)

## Claim C1 — Legacy evaluation unreliability
- [ ] `canonical/section3_faithful_diagnostic.py` executes without error
- [ ] Baseline robust accuracy ≈ 16.06% at ε=0.15
- [ ] Hardened robust accuracy ≈ 40.36% at ε=0.15
- [ ] Retracted 29.10% figure is identified as unreliable

## Claim C2 — Exhaustive enumeration validity
- [ ] `canonical/run_paper_mixednorm.py --check` passes
- [ ] No invalid categorical states in mixed-norm K=0/1/2 counts
- [ ] Archived counts match paper tables

## Claim C3 — Canonical exhaustive K=1 accuracy
- [ ] Fresh evaluation generates `results/foolbox/exh_k1_*.json`
- [ ] `n_total` matches archive exactly
- [ ] `clean_correct` matches archive exactly
- [ ] `k0_survivors`, `k1_survivors` within ±0.75% relative
- [ ] `k1_pct_of_attacked` within ±0.5 pp absolute

## Claim C4 — Greedy projection artifacts
- [ ] `canonical/eval_jsma_vs_exhaustive.py` executes
- [ ] JSMA misses ~12.0% of true adversarial examples
- [ ] JSMA runtime is ~10x slower than PGD

## Claim C5 — Statistical validation
- [ ] `canonical/consolidated_canonical_table.py` runs successfully
- [ ] Aggregated mean±std tables are produced
- [ ] Multi-seed coverage matches manifest (NSL-KDD: 3 seeds, CICIDS2017: 13 seeds, UNSW-NB15: 9 seeds)

## Claim C6 — Independent attack validation
- [ ] `canonical/run_crown_vs_ibp.py` executes without soundness assertion failures
- [ ] Certified bounds align with empirical PGD-40 exhaustive masks
- [ ] C&W survivor records present in `results/cw/`
- [ ] DeepFool EXH K=1 records present in `results/foolbox/`

## Claim C7 — Logit reversal
- [ ] `canonical/trace_logit_reversal_hardened.py` executes
- [ ] Output is byte-identical to `results/logit_trace_model_adv_hardened.json`
- [ ] Per-sample logit margins flip under exhaustive K=1 categorical state changes

## Claim C8 — Scalability
- [ ] `canonical/eval_scalability.py` executes
- [ ] K=1 timing ≈ 0.0155 s/state on reference GPU
- [ ] K=2 timing ≈ 0.118 s/state on reference GPU

## Provenance
- [ ] `PROVENANCE.md` documents methodological evolution
- [ ] `diagnostics/` contains superseded implementations
- [ ] `docs/adr/` contains ADR-001 through ADR-004
- [ ] Canonical vs superseded evaluators are clearly distinguished

## Limitations
- [ ] Dataset reconstruction limitation understood (CICIDS2017/UNSW-NB15)
- [ ] Continuous optimization limitation understood (no global certificate)
- [ ] Random-start variance limitation understood (tolerance-based comparison)
- [ ] Legacy model provenance understood (some checkpoints from superseded pipelines)

## Artifact Integrity
- [ ] All 104 checkpoints verify against `manifest/checkpoint_sha256.json`
- [ ] Canonical results are present in `results/`
- [ ] No author-identifying information in committed files
- [ ] Remote repository is accessible via the anonymized 4open.science proxy link provided in the submission
