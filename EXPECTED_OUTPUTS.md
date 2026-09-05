# Expected Outputs

This document describes what happens when you run each major experiment.
Use this to verify that your environment is correctly configured.

## Checkpoint Verification

```bash
make verify
```

**Expected runtime**: <1 min

**Expected output**:
```
verify_manifest: PASS (104/104 checkpoints)
```

**Expected properties**:
- All 104 `.pth` files under `models/` and `models/unified/` are present
- SHA-256 hashes match `manifest/checkpoint_sha256.json`

## Smoke Tests

```bash
make smoke
```

**Expected runtime**: ~2 min

**Expected output**:
```
verify_manifest: PASS
run_paper_mixednorm: ALL CHECKS PASS
trace_logit_reversal_hardened: OK
consolidated_canonical_table: <printed tables>
```

**Expected properties**:
- No errors or assertions
- Canonical table aggregation completes successfully

## Section III: Faithful AdvGuard Diagnostic

```bash
python3 canonical/section3_faithful_diagnostic.py
```

**Expected runtime**: ~45 s (CUDA) / ~4 min (CPU)

**Expected files**:
- `results/section3/faithful_diagnostic_nsl_kdd.json`

**Expected properties**:
- Baseline robust accuracy ≈ 16.06% at eps=0.15
- Hardened robust accuracy ≈ 40.36% at eps=0.15
- Run-to-run spread: ±0.4 pp

## Mixed-Norm K=0/1/2 Collapse

```bash
python3 canonical/run_paper_mixednorm.py --check
```

**Expected runtime**: ~26 s

**Expected files**:
- `results/results_nsl_kdd_Baseline.json`
- `results/results_nsl_kdd_Curriculum.json`
- `results/results_nsl_kdd_Hardened.json`
- `results/results_nsl_kdd_RSC.json`

**Expected properties**:
- Baseline: 3342/0/0 (K=0/K=1/K=2 survivors)
- Curriculum: 5604/9/9
- Hardened: 17421/0/0
- RSC: 18106/0
- ALL CHECKS PASS against archive

## Multi-Seed EXH K=1 Sweeps (NSL-KDD)

```bash
python3 canonical/eval_deepfool_k1.py --attack pgd40 --dataset nsl-kdd --method hardened --seed 42 --suffix .fresh
```

**Expected runtime**: ~1 min per model

**Expected files**:
- `results/foolbox/exh_k1_pgd40_nsl_kdd_hardened_seed42.fresh.json`

**Expected properties**:
- `n_total` = 22,543 (exact)
- `clean_correct` matches archive (exact)
- `checkpoint_sha256` matches archive (exact)
- `k0_survivors`, `k1_survivors` within ±0.75% relative of archive
- `k1_pct_of_attacked` within ±0.5 pp absolute of archive

## CROWN vs IBP Certified Bounds

```bash
python3 canonical/run_crown_vs_ibp.py --dataset nsl-kdd --method hardened --seed 42
```

**Expected runtime**: ~5 s

**Expected files**:
- `results/certificates/crown_vs_ibp_nsl_kdd_hardened_seed42.json`
- `results/certificates/cert_ibp_nsl_kdd_hardened_seed42.json`

**Expected properties**:
- Certified counts are deterministic and byte-identical to archive
- Soundness check passes: `crown_k1 <= ibp_k1 + 1e-6`

## Scalability

```bash
python3 canonical/eval_scalability.py --dataset unsw_nb15 --method rsc --seed 42 --K 1 --eligible-groups 0,1,2,3,4 --num-samples 500 --out results/scalability_unsw_nb15_K1_g0_1_2_3_4.repro.json
```

**Expected runtime**: ~2 min (CUDA) / ~10-15 min (CPU)

**Expected files**:
- `results/scalability_unsw_nb15_K1_g0_1_2_3_4.repro.json`

**Expected properties**:
- K=1 (42 states): ~0.0155 s/state on reference GPU
- K=2 (667 states): ~0.118 s/state on reference GPU

## JSMA vs Exhaustive

```bash
python3 canonical/eval_jsma_vs_exhaustive.py --dataset unsw_nb15 --method rsc --seed 42 --K 1 --num-samples 500 --out results/jsma_vs_exhaustive_unsw_nb15_K1.repro.json
```

**Expected runtime**: ~1 min

**Expected files**:
- `results/jsma_vs_exhaustive_unsw_nb15_K1.repro.json`

**Expected properties**:
- JSMA misses ~12.0% of true adversarial examples
- JSMA is ~10x slower than PGD

## Logit Reversal Trace

```bash
python3 canonical/trace_logit_reversal_hardened.py
```

**Expected runtime**: ~4 s

**Expected files**:
- `results/logit_trace_model_adv_hardened.json`

**Expected properties**:
- Byte-identical to archive
- Confirms logit margins flip under exhaustive K=1 categorical state changes
