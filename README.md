# Reproducibility Artifact

Everything needed to reproduce every table and figure of the paper from the
shipped checkpoints, scripts, formal proofs, and result files.

## Artifact Navigation

```
                    ARTIFACT NAVIGATION
                          │
          ┌───────────────┼────────────────┐
          │               │                │
       CLAIMS          EVALUATION       PROVENANCE
          │               │                │
     CLAIM_MAP.md     one-shot         ADRs
          │           exhaustive       postmortems
          │           C&W              superseded
          │           certificates     artifacts_results
          │
     Paper Sections
          │
      Evidence
```

| If you are reviewing... | Start here |
|---|---|
| Formal mathematical proofs (Lean 4) | `leanproofs/README.md` |
| Step-by-step mathematical audit | `leanproofs/MATH_AUDIT_REPORT.md` |
| Canonical vs. non-required results taxonomy | [Results Taxonomy & Manuscript Evidence Mapping](#results-taxonomy--manuscript-evidence-mapping) |
| Supplementary / past-version artifacts | `artifacts_results/README.md` |
| Mixed-norm threat model | `docs/threat_model.md` |
| Canonical evaluation pipeline | `canonical/eval_deepfool_k1.py` |
| Mixed-norm enumeration | `canonical/eval_mixed_norm.py` |
| Certified robustness bounds | `canonical/run_crown_vs_ibp.py` |
| Independent attack validation | `canonical/eval_jsma_vs_exhaustive.py` |
| Review verification checklist | `REVIEW_CHECKLIST.md` |
| Reproduce paper tables | `canonical/consolidated_canonical_table.py` |
| Historical methodology changes | `PROVENANCE.md` |
| Evaluation diagnostics & audits | `docs/postmortem/` |
| Known limitations | `LIMITATIONS.md` |
| Claim-to-evidence mapping | `CLAIM_MAP.md` |
| Fast verification | `VERIFY.md` |
| Full reproduction | `REPRODUCE.md` |

```
.
├── README.md                     Overview, quick start, and results taxonomy
├── REVIEWER_GUIDE.md             Entry point for reviewers
├── REVIEW_CHECKLIST.md            Bounded verification checklist
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
├── leanproofs/                   Formal verification in Lean 4 and audit test suite
│   ├── BlackiceProofs/Basic.lean Formal Lean 4 proofs (Propositions 1-4, Theorem 1, Eq. 6)
│   ├── MATH_AUDIT_REPORT.md      Line-by-line step-by-step mathematical audit
│   ├── verify_math.py            Automated test suite verifying algebraic bounds
│   ├── lakefile.toml             Lake project definition (Lean 4)
│   └── README.md                 Lean 4 compilation and verification guide
├── data_scripts/                  Dataset download and preprocessing scripts
│   ├── download_data.py           NSL-KDD
│   ├── download_cicids2017.py     CICIDS2017
│   └── download_unsw_nb15.py      UNSW-NB15
├── dataset_links.md              Dataset acquisition and integrity anchors
├── app/                          Shared library: models, attacks, loaders, training
├── canonical/                    Final evaluators supporting reported conclusions
├── replication/                  Post-freeze replication addendum
│   ├── nf_ton_iot/               NF-ToN-IoT dataset prep and training scripts
│   ├── external_works/           CAA+MOEVA and Illusion/Takedown replication
│   └── results/                  Archived replication outputs (39 JSONs)
├── diagnostics/                  Historical implementations for provenance
├── verification/                 Integrity checks and fresh-vs-archive comparison
├── models/                       Trained checkpoints (SHA-256 pinned)
├── manifest/                     SHA-256 manifests for datasets and checkpoints
├── experiment_manifests/         Machine-readable experiment manifests (C1-C9)
├── results/                      Archived canonical experimental outputs
│   ├── foolbox/                  Multi-seed EXH K=1 sweeps (JSONs & sample masks)
│   ├── certificates/             CROWN vs IBP certified rates (36 models)
│   ├── consolidated/             Aggregated canonical tables across 75 checkpoints
│   ├── section3/                 Faithful AdvGuard diagnostic
│   ├── cw/                       C&W survivor records & K=1 final evaluations
│   └── unified/                  Reconciled masking-gap analysis
├── artifacts_results/            Supplementary / non-required results & raw masks
│   │                             (aliased via symlink as artifacts_restults/)
│   ├── README.md                 Detailed explanatory guide for supplementary files
│   ├── raw_tensor_masks/         Raw PyTorch boolean tensor masks (127 .pt files)
│   ├── legacy_exploratory_runs/  Unseeded, single-checkpoint preliminary runs (Aug 2026)
│   ├── exploratory_probes_and_traces/ Sensitivity probes, gradient traces & nesting logs
│   ├── cluster_batch_runs/       Distributed cluster execution logs and runner dumps
│   └── replication_sota_raw/     Raw population sweeps for MoEvA2 & CAA attacks
└── docs/
    ├── threat_model.md           Mixed-norm threat model specification
    ├── methodology.md            Evaluation methodology and conventions
    ├── postmortem/               Post-evaluation diagnostics and audits
    │   ├── 001-final-diagnostic-summary.md
    │   ├── 002-evaluator-correctness-and-attack-budget-diagnostic.md
    │   ├── 003-reconciliation-diagnostic-vs-archived-results.md
    │   ├── 004-threat-model-clarification-cicids2017.md
    │   ├── 005-audits-1-3-feature-category-semantic-coverage.md
    │   ├── 006-audit-4-representation-sensitivity.md
    │   ├── 007-exh-vs-snap-canonical-comparison.md
    │   ├── 008-nf-ton-iot-and-external-works-replication.md
    │   └── README.md
    └── adr/                      Architectural Decision Records
        ├── ADR-001-canonical-exhaustive-mixed-norm-evaluation.md
        ├── ADR-002-statistical-re-evaluation.md
        ├── ADR-003-independent-attack-validation.md
        └── ADR-004-canonical-vs-fresh-reproduction.md
```

---

## Results Taxonomy & Manuscript Evidence Mapping

To ensure total transparency, all result files across the repository are explicitly classified into **Required / Canonical Results** (which directly support the manuscript claims, tables, and figures) and **Supplementary / Non-Required Artifacts** (which are archived in `artifacts_results/` for auditability and provenance, but are not needed to reproduce any paper claim).

### 1. Canonical Results Supporting Manuscript Tables & Figures

Every table and figure in the manuscript maps to an authoritative canonical result file:

| Paper Item | Content / Description | Canonical Data File(s) | Reproduction Script / Target | Status |
|---|---|---|---|---|
| **Table 1** | Reconstruction & audit of 5 prior-work constrained tabular paradigms on NSL-KDD (Section 3) | `results/section3/faithful_diagnostic_nsl_kdd.json` | `make reproduce-table3` (`canonical/section3_faithful_diagnostic.py`) | **Canonical** |
| **Table 2** | Five-point unit-fixture verification audit of reference EXH evaluator (Section 4) | `verification/fixtures/fixture_test_results.json` | `python3 verification/fixtures/test_exh_fixtures.py` | **Canonical** |
| **Table 3** | Taxonomy of evaluation and training failure modes (Pitfalls P1–P6) (Section 6) | Conceptual synthesis across Sections 3–6 | Main text synthesis | **Conceptual** |
| **Table 4** | Canonical EXH vs SNAP robust accuracy across 75 checkpoints (Section 7.1) | `results/consolidated/canonical_both_conventions.json`<br>`results/foolbox/exh_k1_pgd40_*.json` | `make reproduce-canonical-table`<br>`make reproduce-multiseed-*` | **Canonical** |
| **Table 5** | NSL-KDD raw survivor counts across all 9 checkpoints under EXH K=1 (Section 7.2) | `results/foolbox/exh_k1_pgd40_nsl_kdd_*.json` | `make reproduce-multiseed-nslkdd` | **Canonical** |
| **Table 6** | Editability regime, causal macro-groups, and staged K=2 audit on UNSW-NB15 (Section 7.3) | `verification/staged/regime_b_unsw_nb15_rsc_seed42.json`<br>`verification/staged/unsw_nb15_macro_hypothesis_test.json`<br>`verification/staged/unsw_nb15_rsc_seed42_k012.json` | `python3 verification/staged/eval_regime_b.py` | **Canonical** |
| **Table 7** | NF-ToN-IoT-v2 encoding ablation (Standard vs One-Hot) across K=0/1 (Section 7.4) | `replication/results/eval_lite_nf_ton_iot.json`<br>`replication/results/eval_lite_nf_ton_iot_oh.json` | `python3 replication/nf_ton_iot/eval_ton_ablation.py` | **Canonical** |
| **Table 8** | Controlled initialization comparison (deterministic PGD start) across 3 datasets (Section 7.5) | `verification/pilots/reconciled_comparison.json`<br>`verification/pilots/exh_vs_snap_all_datasets.json` | `python3 verification/pilots/reconcile_controlled_init.py` | **Canonical** |
| **Table 9** | Reconciled K=1 masking gap distribution across all 75 checkpoints (Section 7.6) | `results/unified/eval_*.json`<br>`results/consolidated/canonical_both_conventions.json` | `python3 canonical/consolidated_canonical_table.py` | **Canonical** |
| **Table 10** | SOTA Tabular Attack Benchmark (Cat-PGD, MoEvA2, CAA, JSMA vs EXH) (Section 7.6.1) | `verification/staged/attack_table_results.json`<br>`results/jsma_vs_exhaustive_unsw_nb15_K1.json`<br>`replication/results/sota/*` | `make reproduce-jsma`<br>`python3 verification/staged/eval_sota_benchmark.py` | **Canonical** |
| **Table 11** | Seed-level EXH K=1 robust accuracy across 13 RSC checkpoints on CICIDS2017 (Section 7.6.2) | `results/foolbox/exh_k1_pgd40_cicids2017_rsc_seed*.json`<br>`results/foolbox/rsc_outlier_forensics_cicids2017.json` | `make reproduce-multiseed-cicids` | **Canonical** |
| **Table 12** | Defense Comparison: Feature Squeezing & RanMASK vs RSC under EXH K=1 (Section 7.6.2) | `verification/staged/defense_table_results.json`<br>`replication/results/*feature_squeezing.json`<br>`replication/results/*gumbel_softmax.json` | `python3 verification/staged/eval_defense_comparison.py` | **Canonical** |
| **Table 13** | Evaluator wall-clock scaling on UNSW-NB15 across feature groups g and K=1,2 (Section 7.8) | `results/scalability_unsw_nb15_K1_*.json`<br>`results/scalability_unsw_nb15_K2_*.json`<br>`results/scalability_unsw_nb15_summary.json` | `make reproduce-scalability` | **Canonical** |
| **Table 14** | Consolidated mixed-norm certified robustness lower bounds (IBP vs CROWN) (Section 7.7) | `results/certificates/crown_vs_ibp_*.json` (36 files across eps=0.01 and eps=0.03) | `make reproduce-certified` (`canonical/run_crown_vs_ibp.py`) | **Canonical** |
| **Figure 2(a)** | Snap-semantics masking gap across defenses (n=13 seeds) | `results/consolidated/canonical_both_conventions.json` | `make reproduce-canonical-table` | **Canonical** |
| **Figure 2(b)** | Bimodal unmasked training distribution across 13 RSC seeds | `results/foolbox/exh_k1_pgd40_cicids2017_rsc_seed*.json` | `make reproduce-multiseed-cicids` | **Canonical** |
| **Figure 2(c)** | Cross-optimizer convergence (PGD, C&W, DeepFool on RSC Seed 53) | `results/foolbox/EXH_K1_METHODS_seed53_cicids2017.json`<br>`results/cw/K1_FINAL_cicids2017_rsc_seed53.json` | `python3 canonical/eval_deepfool_k1.py` | **Canonical** |
| **Formal Proofs** | Machine-checked verification of Propositions 1–4, Theorem 1, and Equation (6) | `leanproofs/BlackiceProofs/Basic.lean`<br>`leanproofs/verify_math.py` | `make verify-lean`<br>`make verify-math` | **Machine-Checked** |

### 2. Supplementary / Non-Required Artifacts (`artifacts_results/` / `artifacts_restults/`)

The directory `artifacts_results/` (accessible also via symlink `artifacts_restults/`) contains supplementary files that are **NOT required** to reproduce any manuscript claim. They are preserved for completeness:

1. **`artifacts_results/raw_tensor_masks/` (127 PyTorch `.pt` files, ~84.7 MB):**
   - Contains binary boolean tensor masks for each attacked sample across the foolbox and Carlini-Wagner (C&W) runs.
   - *Why Not Required:* All aggregate percentages, sample counts, and break rates are already fully serialized in the canonical `.json` files. The raw `.pt` files provide bitwise auditability of the per-sample Boolean vectors, but are not needed for table reproduction.
2. **`artifacts_results/legacy_exploratory_runs/` (18 JSON files):**
   - Unseeded single-model evaluations (`results_Baseline.json`, `results_Legacy_FGSM-Hardened.json`, `results_cicids2017_*.json`, `results_unsw_nb15_*.json`) logged during early exploratory work (Aug 16–22, 2026).
   - *Why Not Required:* Superseded by the rigorous 13-seed protocol (seeds 42–54) established in Section 7.1 and stored in `results/consolidated/canonical_both_conventions.json`.
3. **`artifacts_results/exploratory_probes_and_traces/` (8 JSON files):**
   - Diagnostic probes (`df_set_convergence_probe_rsc_seed53.json`, `exh_k1_deepfool_cicids2017_rsc_seed53_probe150.json`, `k1_nesting_perbatch_*.json`) used during evaluator development.
   - *Why Not Required:* Internal engineering checks confirming gradient saturation and batch nesting.
4. **`artifacts_results/cluster_batch_runs/` (Cluster Execution Logs & Dumps):**
   - Slurm/cluster execution logs (`full_run_execution/logs/`) and preliminary runner outputs (`unified_full/` from August 17, 2026).
   - *Why Not Required:* Intermediate task execution traces.
5. **`artifacts_results/replication_sota_raw/` (13 JSON files):**
   - Raw parameter sweep outputs from external attacks (MoEvA2 and CAA) across varying population sizes ($N \in \{20, 50, 100, 1000\}$).
   - *Why Not Required:* Table 10 reports the consolidated official benchmark compiled in `verification/staged/attack_table_results.json`.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
. .venv/bin/activate
make setup          # fetches all datasets (NSL-KDD, CICIDS2017, UNSW-NB15)
make verify         # confirm every checkpoint matches its pinned SHA-256
make verify-math    # verify all 6 mathematical theorems with Python test suite (<1s)
make verify-lean    # formal machine-checked verification via Lean 4
make smoke          # fast end-to-end check (~2 min)
make reproduce-table3
...
```

GPU is optional. All evaluations default to CPU (as archived); only
`reproduce-scalability` / `reproduce-jsma` use CUDA when available. Runtimes
cited are from the reference machine: 6 GB mobile GPU, 10-core CPU, 11 GB RAM.

## Environment

| Package   | Version (pinned) |
|-----------|------------------|
| torch     | 2.9.1+cu128      |
| numpy     | 2.4.0            |
| pandas    | 2.3.3            |
| pyarrow   | 24.0.0           |
| scipy     | 1.16.3           |
| foolbox   | 3.3.4            |
| auto-LiRPA| 0.3              |

## What reproduces exactly vs. within tolerance

| Quantity                                        | Behavior |
|-------------------------------------------------|----------|
| checkpoint SHA-256 (all 106 files)              | exact (manifest) |
| mathematical soundness & bounds (all 6 results) | exact / machine-checked (`verify_math.py`, `lake build`) |
| mixed-norm K=0/1/2 counts (NSL-KDD)             | exact (`ALL CHECKS PASS`) |
| certified CROWN/IBP rates (36 checkpoints)      | byte-identical |
| JSMA vs exhaustive-image comparison             | byte-identical |
| logit-reversal trace                            | byte-identical |
| canonical aggregated tables                     | exact (input-driven) |
| EXH K=1 multiseed sweeps (foolbox)              | within tolerance (below) |
| Section 3 faithful diagnostic                   | within noise (below) |

**EXH K=1 sweeps.** `eval_deepfool_k1.py` performs an **unseeded** random-start
PGD-40; each surviving-sample outcome can flip between runs. The comparator
(`verification/compare_exh_fresh.py`) pre-registers tolerances: `n_total`,
`clean_correct`, `attacked`, and embedded `checkpoint_sha256` are exact;
`k0/k1_survivors` within ±0.75% relative; `k1_pct_of_attacked` within ±0.5 pp.
Observed drift in validation runs was ≪ 0.1 pp.

**Section 3 diagnostic.** The faithful AdvGuard evaluator uses an unseeded
random-start PGD. Observed run-to-run spread ±0.4 pp. Archived draw:
hardened @ eps 0.15 = **40.36%** (Table 1), baseline 16.06%. The
robust claim under reproduction: hardened ≈ 40 ± 0.4% vs baseline ≈ 16%, with
the originally-reported ~29.10% not reproduced (the Section 3 finding).

## Script → paper mapping

| Target | Script | Paper content | Runtime (ref.) |
|--------|--------|---------------|----------------|
| `reproduce-table3` | `canonical/section3_faithful_diagnostic.py` | Table 1: faithful AdvGuard reproduction (baseline 16.06%, hardened 40.36% @ 0.15) | ~45 s (CUDA) / ~4 min (CPU) |
| `reproduce-table-mixednorm` | `canonical/run_paper_mixednorm.py` + `canonical/eval_mixed_norm.py` | mixed-norm K=0/1/2 collapse, NSL-KDD (Hardened 17421/0/0; Curriculum 5604/9/9; Baseline 3342/0/0; RSC 18106/0) | ~26 s |
| `reproduce-multiseed-nslkdd` | `canonical/eval_deepfool_k1.py --attack pgd40` | multiseed EXH K=1, NSL-KDD, 9 models (Table 4, 5) | ~10 min |
| `reproduce-multiseed-cicids` | same | multiseed EXH K=1, CICIDS2017, 39 models (Table 4, 11) | ~6 h (parallelize) |
| `reproduce-multiseed-unsw` | same | multiseed EXH K=1, UNSW-NB15, 27 models (Table 4, 6) | ~9–10 h (parallelize) |
| `reproduce-certified` | `canonical/run_crown_vs_ibp.py` (+ `canonical/crown_bound.py`, `canonical/certified_bound.py`) | Table 14: CROWN vs IBP certified K=0/K=1 rates, 36 models | ~12–15 min |
| `reproduce-scalability` | `canonical/eval_scalability.py` | Table 13: K=1 0.0155 s/state, K=2 0.118 s/state (reference GPU) | ~2 min (GPU) |
| `reproduce-jsma` | `canonical/eval_jsma_vs_exhaustive.py` | Table 10: JSMA vs exhaustive-image, 500 UNSW samples | ~1 min |
| `reproduce-logit-trace` | `canonical/trace_logit_reversal_hardened.py` | Table 3 prose: per-sample logit reversal for the hardened NSL model | ~4 s |
| `reproduce-canonical-table` | `canonical/consolidated_canonical_table.py` | aggregated canonical summary across all 75 checkpoints | <1 s |

### Deepfool variant
`canonical/eval_deepfool_k1.py --attack deepfool` reproduces the DeepFool EXH K=1
records (`results/foolbox/exh_k1_deepfool_*`; 47 files). Use
`--suffix .fresh` and `verification/compare_exh_fresh.py --attack deepfool` to
verify. Two CICIDS2017 seed-53 records (hardened, curriculum) are intentionally
absent from the archive (evaluation budget); `--fresh` re-creates them.

### Superseded evaluator (archived for completeness)
`canonical/eval_unified.py` was used for the Section 7.6 masking-gap analysis
(`results/unified/*`). It is superseded by `canonical/eval_deepfool_k1.py` +
`canonical/consolidated_canonical_table.py`; kept for provenance. Usage:
`python3 canonical/eval_unified.py --datasets cicids2017 --suffix .repro`.

## Re-generation vs. shipped results

**Shipped result files are authoritative.** `reproduce-*` targets write fresh
output under `*.repro.*` / `*.fresh.*` / `results_fresh/` paths and never
overwrite the archive, then verify. Use `make clean` to remove fresh outputs.

Confirmed reproducible (validation runs on the reference machine):
byte-identical for `reproduce-certified`, `reproduce-logit-trace`,
`reproduce-canonical-table`, `reproduce-jsma`; exact for
`reproduce-table-mixednorm`; within tolerance for the multiseed sweeps.

## Known gaps / honest limitations

- **CICIDS2017 / UNSW-NB15 from-scratch regeneration is scripted but not
  guaranteed byte-identical.** `data_scripts/download_cicids2017.py` and
  `data_scripts/download_unsw_nb15.py` implement the canonical preprocessing
  pipelines. However, public mirrors may change and category frequencies may
  differ slightly, so output hashes may not match the shipped manifests.
  The shipped `data/*.parquet` files are the authoritative source for exact
  reproduction.
- **One legacy NSL/UNSW mixed-norm file is stale.** Of the legacy per-dataset
  K=0/1 files produced by the pre-foolbox mixed-norm evaluator, all are full
  test-set runs except `results/results_unsw_nb15_Hardened.json`, which was a
  partial run (`total = 348,000`). It is not a source of any paper table; the
  UNSW-NB15 numbers the paper reports come from the foolbox multiseed sweeps
  (`results/foolbox/exh_k1_pgd40_*`, 27 unsw files, all `n_total = 508,010`)
  and the `consolidated/` and `unified/` tables, which use the full 508,010-row
  test set. The full-test-set twin is `results_unsw_nb15_Hardened_fgsm.json`.
- **`EPS`/attack constants** are defined inside the shipped scripts (mirroring
  `app/ml/attacks/eval_protocol.py`) so each script is self-contained.
- **Random-start attacks are unseeded** by design (faithful to the original
  pipeline); tolerances above were pre-registered before validation.
- **Certified sweeps** cover NSL-KDD (seeds 42–44), CICIDS2017 (seeds 42–44),
  and UNSW-NB15 (seeds 42–44) across $\epsilon \in \{0.01, 0.03\}$, totaling
  36 canonical certification models in `results/certificates/crown_vs_ibp_*.json`.

## Reviewer Documentation

This artifact includes a complete reviewer documentation suite. Start with
`REVIEWER_GUIDE.md` for navigation guidance. Key documents:

- `CLAIM_MAP.md` — Every paper claim mapped to exact code, output, and evidence
- `PROVENANCE.md` — Methodological evolution and superseded implementations
- `LIMITATIONS.md` — Explicit boundaries and dataset caveats
- `VERIFY.md` — Fast verification without full reproduction
- `REPRODUCE.md` — Full reproduction instructions
- `REPRODUCIBILITY_LEVELS.md` — Classification of reproduction guarantees
- `EXPECTED_OUTPUTS.md` — What to expect when running each experiment
- `CITATION_TO_ARTIFACT.md` — Versioning and citation information
- `docs/adr/` — Architectural Decision Records explaining key design choices
- `experiment_manifests/` — Machine-readable experiment manifests (C1–C9)

## Files omitted from the artifact

- **Raw datasets (~700 MB):** Not shipped due to size limits; fetched automatically via `make setup` (see `dataset_links.md` for cryptographic integrity anchors).
- **Paper LaTeX source and compiled PDF (`paper/`, `*.tex`, `*.pdf`):** Strictly excluded to preserve double-blind submission anonymity and adhere to conference artifact guidelines.
- **Per-sample attack mask `.pt` files (84.7 MB):** Now **fully included** under `artifacts_results/raw_tensor_masks/` and `results/foolbox/` / `results/cw/` for complete bitwise auditability.
