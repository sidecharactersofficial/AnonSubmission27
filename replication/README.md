# Replication Addendum

**Scope disclaimer:** Everything in this directory is a *post-submission
replication extension*. It is not part of the canonical evaluation pipeline
and none of it supports the paper claims C1-C8 in `CLAIM_MAP.md`. Files here
were produced later than the artifact freeze (`commit c200e28`, the last
"Anonymous" baseline) and are provided for completeness and scientific
provenance, not for claim verification.

## Contents

```
replication/
├── nf_ton_iot/            Dataset extension: NF-ToN-IoT replication study
│   ├── prep_nf_ton_iot.py         Dataset preparation (original encoding)
│   ├── prep_nf_ton_iot_onehot.py  Dataset preparation (one-hot encoding)
│   └── train_nf_ton_iot_lite.py   Model training ("lite" model family)
├── external_works/        Independent third-party attack replication
│   ├── eval_caa_moeva.py          CAA + MOEVA adversarial attacks
│   ├── replicate_illusion_and_takedown.py
│   └──                            Replicates "Illusion" and "Takedown"
│                                   trigger-detection attacks on curriculum/
│                                   hardened/rsc checkpoints
└── results/               Archived outputs (39 JSON files)
    ├── eval_curriculum_*          Per-model, per-attack variant results
    ├── eval_hardened_*            (NSL-KDD, CICIDS2017, UNSW-NB15)
    ├── eval_rsc_*
    ├── eval_lite_nf_ton_iot.json  NF-ToN-IoT "lite" checkpoints
    ├── eval_lite_nf_ton_iot_oh.json
    └── constraint_map.json        Per-experiment constraint reference
```

Related additions:

- `app/ml/data/nf_ton_iot_config.py` / `nf_ton_iot_oh_config.py` — preprocessing
  configs for the NF-ToN-IoT dataset variants; the two variants differ mainly
  in categorical-feature encoding (native vs. one-hot).
- `app/ml/data/loader.py` — a four-line addition routing the two new dataset
  names (`nf_ton_iot`, `nf_ton_iot_oh`) to their respective configs. It does
  not modify behavior for the original three datasets.
- `models/unified/model_adv_lite_nf_ton_iot.pth` /
  `model_adv_lite_nf_ton_iot_oh.pth` — compact ("lite") checkpoints produced
  by `train_nf_ton_iot_lite.py`. These are *supplementary* — not part of the
  canonical checkpoint manifest, no paper claim depends on them.

## Companion document

A narrative description of how this replication was run and what it shows is
in `docs/postmortem/008-nf-ton-iot-and-external-works-replication.md`.

## Reviewer note

If you are verifying C1-C8, you can stop here — nothing in `replication/`
affects the canonical results tables reproduced by
`canonical/consolidated_canonical_table.py` or checked by
`verification/compare_exh_fresh.py`.
