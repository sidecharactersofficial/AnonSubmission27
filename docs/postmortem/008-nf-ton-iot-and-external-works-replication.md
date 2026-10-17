# Postmortem 8: NF-ToN-IoT Dataset Extension and External-Works Replication

**Scope**: Post-freeze replication addendum (`replication/`). Not part of the
canonical C1-C8 evaluation; provided for provenance and completeness.

## Overview

After the artifact baseline was frozen, three replication thrusts were run
against the frozen pipeline. All reuse the canonical evaluator protocol
(ADR-001: L-inf PGD-40 + exhaustive K=1 categorical enumeration) on
checkpoints *outside* the paper's original model set.

## 1. NF-ToN-IoT dataset extension

NF-ToN-IoT is a network intrusion detection dataset derived from the
ToN-IoT testbed, evaluated here in two preprocessing variants:

- **`nf_ton_iot`** (native encoding) and **`nf_ton_iot_oh`** (one-hot
  categorical encoding), each with its own config under `app/ml/data/`.
- Compact "lite" checkpoints were trained with
  `replication/nf_ton_iot/train_nf_ton_iot_lite.py`:
  - `models/unified/model_adv_lite_nf_ton_iot.pth`
    (SHA-256 `596f0d00...90beb019`)
  - `models/unified/model_adv_lite_nf_ton_iot_oh.pth`
    (SHA-256 `3d1783ce...563e595a`)

### Headline numbers (500-sample eval, eps=0.15, 40 PGD steps, ADR-001 protocol)

| Variant | Clean | K=0 robust (% of clean-correct) | K=1 robust (% of clean-correct) |
|---|---|---|---|
| `nf_ton_iot` (native) | 67.2% | 87.5% | **87.5%** (K=1 == K=0: no categorical attack surface) |
| `nf_ton_iot_oh` (one-hot) | 78.8% | 83.2% | **0.0%** (≈85% of K=0 survivors collapse under K=1: 328 → 0) |

**Principal finding**: encoding choice *materially changes* the apparent
K=1 robustness — with one-hot categorical groups the model collapses to 0%
under exhaustive K=1 enumeration, while the native encoding variant retains
full K=0 performance. This is consistent with, and extends, the
representation-sensitivity concern documented in `006-audit-4-representation-sensitivity.md`:
robust-accuracy measurements are encoding-dependent measurements, and
one-hot definitions open a categorical surface that other encodings leave
closed. Full results:
`replication/results/eval_lite_nf_ton_iot{,_oh}.json`.

## 2. External-works replication (CAA + MOEVA, Illusion/Takedown)

- `replication/external_works/eval_caa_moeva.py` — CAA and MOEVA attacks
  run against the curriculum / hardened / rsc checkpoints across NSL-KDD,
  CICIDS2017, and UNSW-NB15.
- `replication/external_works/replicate_illusion_and_takedown.py` —
  trigger-detection style attacks replicating the "Illusion" and "Takedown"
  threat models, useful as an independent, out-of-family attack comparator
  (analogous to the role of C&W/DeepFool validation in claim C6, but for
  external works).

Results: `replication/results/eval_curriculum_*`, `eval_hardened_*`,
`eval_rsc_*` (39 files) across `feature_squeezing`,
`gumbel_softmax`, and `tabnet_embedding` attack variants; original
constraint references in `replication/results/constraint_map.json`.

## Provenance / status

- **Canonical status**: not canonical. None of these artifacts are in
  `manifest/checkpoint_sha256.json`; all outputs live under `replication/`
  so reviewers can distinguish them from archived C1-C8 evidence.
- **Not verified**: the artifact's `make verify` / `make smoke` targets do
  not exercise this directory.
- `app/ml/data/loader.py` gained four lines routing `nf_ton_iot` /
  `nf_ton_iot_oh` to their configs; no behavior of the canonical three
  datasets is changed. Verified by diff against the pre-addendum version
  (`diff` shows the guard-clause block only).

## What still to do if this ever moves into the main story

- Add experiments to `experiment_manifests/` with stable claim IDs + link to
  `CLAIM_MAP.md`.
- Pin checkpoint SHAs into `manifest/checkpoint_sha256.json` once finalized.
- Multi-seed, not just the 500-sample slices reported here.
- Add NSL-KDD/CICIDS2017/UNSW-NB15 baselines for the CAA+MOEVA / Illusion /
  Takedown column comparisons so they can sit in the same tables as C6.
