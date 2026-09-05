# Reviewer Guide

This artifact accompanies the paper on adversarial robustness of tabular
intrusion detection systems under mixed-norm threat models. The repository
contains multiple independent evaluation components. Reviewers do not need
to inspect the repository linearly.

## Quick Navigation

| If you are reviewing... | Start here |
|---|---|
| Mixed-norm threat model | `docs/threat_model.md` |
| Canonical evaluation pipeline | `canonical/eval_deepfool_k1.py` |
| Mixed-norm enumeration | `canonical/eval_mixed_norm.py` |
| Certified robustness bounds | `canonical/run_crown_vs_ibp.py` |
| Independent attack validation | `canonical/eval_jsma_vs_exhaustive.py` |
| Reproduce paper tables | `canonical/consolidated_canonical_table.py` |
| Historical methodology changes | `PROVENANCE.md` |
| Known limitations | `LIMITATIONS.md` |
| Claim-to-evidence mapping | `CLAIM_MAP.md` |
| Fast verification | `VERIFY.md` |
| Full reproduction | `REPRODUCE.md` |

## Repository Structure

```
.
├── README.md                     Overview and quick start
├── REVIEWER_GUIDE.md             This file — start here
├── CLAIM_MAP.md                  Scientific claim → evidence navigation
├── VERIFY.md                     Fast verification without full reproduction
├── REPRODUCE.md                  Full reproduction instructions
├── LIMITATIONS.md                Explicit boundaries of the artifact
├── PROVENANCE.md                 Methodological history and superseded code
├── REPRODUCIBILITY_LEVELS.md     Classification of reproduction guarantees
├── EXPECTED_OUTPUTS.md           What to expect when running each experiment
├── CITATION_TO_ARTIFACT.md       Versioning and citation information
├── Makefile                      Reproducibility pipeline
├── requirements.txt              Pinned Python environment
├── download_data.py              NSL-KDD dataset download and preprocessing
├── dataset_links.md              Dataset acquisition and integrity anchors
├── app/                          Shared library: models, attacks, loaders, training
├── canonical/                    Final evaluators supporting reported conclusions
├── diagnostics/                  Historical implementations for provenance
├── verification/                 Integrity checks and fresh-vs-archive comparison
├── models/                       Trained checkpoints (SHA-256 pinned)
├── manifest/                     SHA-256 manifests for datasets and checkpoints
├── results/                      Archived experimental outputs
│   ├── foolbox/                  Multi-seed EXH K=1 sweeps
│   ├── certificates/             CROWN vs IBP certified rates
│   ├── consolidated/             Aggregated canonical tables
│   ├── section3/                 Faithful AdvGuard diagnostic
│   ├── cw/                       C&W survivor records
│   └── unified/                  Superseded masking-gap analysis
└── docs/
    ├── threat_model.md           Mixed-norm threat model specification
    ├── methodology.md            Evaluation methodology and conventions
    └── adr/                      Architectural Decision Records
        ├── ADR-001-canonical-exhaustive-mixed-norm-evaluation.md
        ├── ADR-002-statistical-re-evaluation.md
        ├── ADR-003-independent-attack-validation.md
        └── ADR-004-canonical-vs-fresh-reproduction.md
```

## Suggested Review Paths

### Fast review (~15 minutes)
1. Read `README.md` methodology section
2. Read `CLAIM_MAP.md`
3. Inspect one canonical evaluator (`canonical/eval_deepfool_k1.py` or `canonical/eval_mixed_norm.py`)
4. Inspect corresponding canonical results (`results/consolidated/`, `results/foolbox/`)

### Methodology review (~1 hour)
1. Read `docs/threat_model.md`
2. Read `docs/methodology.md`
3. Read `docs/adr/ADR-001-canonical-exhaustive-mixed-norm-evaluation.md`
4. Inspect exhaustive evaluator (`canonical/eval_deepfool_k1.py`)
5. Inspect independent validation (`canonical/run_crown_vs_ibp.py` or `canonical/eval_jsma_vs_exhaustive.py`)

### Reproducibility review (~2 hours)
1. Read `VERIFY.md`
2. Run `make verify` to confirm checkpoint integrity
3. Run `make smoke` for fast end-to-end verification
4. Read `REPRODUCE.md`
5. Run selected `reproduce-*` targets for time-budgeted experiments
6. Compare fresh outputs against canonical archive using `verification/compare_exh_fresh.py`

### Provenance review (~30 minutes)
1. Read `PROVENANCE.md`
2. Inspect `diagnostics/` for superseded implementations
3. Read `docs/adr/` for decision rationale
4. Compare `canonical/eval_unified.py` (superseded) with `canonical/eval_deepfool_k1.py` (canonical)

## Dataset Caveats

- **NSL-KDD** can be regenerated from public sources using `download_data.py`.
- **CICIDS2017** and **UNSW-NB15** preprocessing is not scripted. The raw
  data acquisition pipeline is documented in `dataset_links.md`. Processed
  tensor SHAs are pinned in `manifest/dataset_sha256.txt`. The script does
  not include dataset-specific preprocessing for these two datasets because
  the preprocessing steps involve proprietary or manual transformations that
  cannot be fully automated from publicly available raw data.
- Reviewers with limited time can verify all claims using the shipped
  preprocessed tensors and checkpoints without re-running data pipelines.
