# Supplementary Results & Non-Required Artifacts (`artifacts_results/`)

> [!NOTE]
> **Notice to Reviewers:** None of the files in this directory are required to reproduce any of the 14 tables, figures, or scientific claims presented in the paper. All canonical evidence required for manuscript reproduction is contained in `results/`, `verification/`, `leanproofs/`, and `replication/results/` as indexed in the root [`README.md`](../README.md) and [`CLAIM_MAP.md`](../CLAIM_MAP.md).

This directory (`artifacts_results/`, aliased via symlink as `artifacts_restults/`) archives raw tensor masks, early exploratory runs, intermediate diagnostic probes, cluster execution batch logs, and raw heuristic attack sweeps. They are included here for 100% bitwise auditability, complete methodological provenance, and archival integrity.

---

## Directory Organization

```
artifacts_results/
├── README.md                      This explanatory guide
├── raw_tensor_masks/              Raw PyTorch boolean tensor masks (.pt files)
│   ├── foolbox/                   124 per-seed attack survivor tensor masks
│   └── cw/                        3 Carlini-Wagner (C&W) L2-optimizer survivor masks
├── legacy_exploratory_runs/       Unseeded, single-checkpoint preliminary runs (Aug 16–22, 2026)
│   ├── results_Baseline.json
│   ├── results_Legacy_FGSM-Hardened.json
│   ├── results_Legacy_FGSM-Hardened_K0-Anchor.json
│   ├── results_New_Curriculum_PGD-Hardened.json
│   ├── results_cicids2017_*.json  (Baseline, Curriculum, Hardened, Hardened_fgsm, RSC)
│   ├── results_unsw_nb15_*.json   (Baseline, Curriculum, Hardened, Hardened_fgsm, RSC)
│   ├── results_nsl_kdd_*.json     (Archival copies; active copies in results/ for manifests)
│   ├── prior_work_reconstruction.json
│   └── consolidated_ibp_crown_matrix.json
├── exploratory_probes_and_traces/ Diagnostic probes and gradient traces
│   ├── logit_trace_nsl_kdd.json
│   ├── df_set_convergence_probe_rsc_seed53.json
│   ├── exh_k1_deepfool_cicids2017_rsc_seed53_probe150.json
│   ├── anchor_backup_eval_rsc_cicids2017_seed53.json
│   ├── k1_nesting_perbatch_cicids2017_rsc_seed53.json
│   ├── k1_nesting_perbatch_bs2000_cicids2017_rsc_seed53.json
│   ├── survivor_comparison_nsl_kdd_rsc_seed42.json
│   └── nsl_kdd_pgd40_repass.json
├── cluster_batch_runs/            Cluster sweep execution dumps and logs
│   ├── full_run_execution/        Slurm/cluster batch execution task logs & summaries
│   └── unified_full/              Initial cluster runner dumps (Aug 17, 2026)
└── replication_sota_raw/          Raw heuristic attack logs across varying population sizes
    ├── sota_moeva_*.json          MoEvA2 genetic algorithm runs (N=20, 50, 100, 1000)
    ├── sota_caa_*.json            CAA combinatorial attack runs (N=50, 100, 1000)
    └── consolidated_attack_benchmark.json
```

---

## Detailed Contents & Purpose

### 1. `raw_tensor_masks/` (127 PyTorch `.pt` Files, ~84.7 MB uncompressed)
- **Foolbox (`raw_tensor_masks/foolbox/`, 124 files):** Contains serialized PyTorch boolean tensors recording sample-level survival masks across 100 attacked samples for each optimizer run (PGD-40, DeepFool) across seeds 42–54.
- **Carlini-Wagner (`raw_tensor_masks/cw/`, 3 files):** Contains PyTorch boolean tensor masks for C&W optimization on CICIDS2017 RSC Seed 53 (`masks_cicids2017_rsc_seed53.pt`, `cw_masks_flat_cicids2017_rsc_seed53.pt`) and NSL-KDD RSC Seed 42 (`masks_nsl_kdd_rsc_seed42_repass.pt`).
- **Why Not Required for Paper Claims:** The parsed summary statistics (robust accuracy percentages, clean accuracy, break rates, survivor counts, and standard errors) are already fully serialized in the canonical `.json` files in `results/foolbox/` and `results/cw/`. These `.pt` files represent internal binary tensors used during evaluation to compute intersections between attack survivor sets. They are preserved for binary bitwise auditability and sample-level inspection.

### 2. `legacy_exploratory_runs/` (18 JSON Files)
- **Early Single-Model Evaluations:** During early development (August 16–22, 2026), single-checkpoint evaluations (`results_Baseline.json`, `results_Legacy_FGSM-Hardened.json`, `results_cicids2017_*.json`, `results_unsw_nb15_*.json`) were logged before establishing the full 13-seed protocol (seeds 42–54) described in Section 7.1.
- **Why Not Required for Paper Claims:** These early single-seed runs are completely superseded by the multi-seed canonical benchmark across 75 checkpoints (`results/consolidated/canonical_both_conventions.json` and `results/foolbox/exh_k1_pgd40_*.json`). They are archived here to document the research iteration history.

### 3. `exploratory_probes_and_traces/` (8 JSON Files)
- **Sensitivity Probes & Nesting Diagnostics:** Contains intermediate gradient sensitivity experiments, such as `df_set_convergence_probe_rsc_seed53.json` and `exh_k1_deepfool_cicids2017_rsc_seed53_probe150.json` (testing whether increasing DeepFool steps beyond 150 yields additional categorical flips), and `k1_nesting_perbatch_*.json` (validating batch-level nesting of continuous perturbation sets inside discrete candidate fibers).
- **Why Not Required for Paper Claims:** These were diagnostic safety checks run during the verification phase. The definitive findings are summarized in Section 7.6.2 and Figure 2(c), supported by `results/cw/K1_FINAL_cicids2017_rsc_seed53.json` and `results/foolbox/EXH_K1_FINAL_cicids2017_rsc_seed53.json`.

### 4. `cluster_batch_runs/` (Cluster Execution Logs & Dumps)
- **Execution Artifacts:** Contains task logs (`full_run_execution/logs/`) from cluster certification, attack benchmarking, and preliminary sweeps (`unified_full/` from August 17, 2026).
- **Why Not Required for Paper Claims:** These are execution trace dumps from distributed cluster nodes. The final authoritative outputs are compiled into `results/` and verified by `canonical/consolidated_canonical_table.py`.

### 5. `replication_sota_raw/` (13 JSON Files)
- **Raw Attack Population Sweeps:** Contains individual run logs from the baseline comparison against external tabular attacks (MoEvA2 and CAA) across varying population sizes ($N \in \{20, 50, 100, 1000\}$).
- **Why Not Required for Paper Claims:** Table 10 reports the consolidated full-dataset benchmark using the official staged results in `verification/staged/attack_table_results.json` and `results/jsma_vs_exhaustive_unsw_nb15_K1.json`. These raw files show the intermediate population sweep progression.
