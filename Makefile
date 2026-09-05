# Reproducibility targets for all paper tables/figures.
#
# Default Python: `python3`. GPU is *not* required for any target; every
# evaluation runs on CPU by design (matching the archived numbers) except
# `reproduce-scalability` / `reproduce-jsma`, which use CUDA when present.
#
# Every `reproduce-*` target re-runs the evaluation, writes fresh outputs under
# a `*.repro.*` / `*.fresh.*` / `results_fresh/` path (never over the shipped
# archive), then verifies against the archived file and prints a PASS/FAIL line.
#
# Runtimes quoted below were measured on the reference machine (6 GB mobile GPU,
# 10-core CPU, 11 GB RAM). Expect ~2x on a desktop CPU.

PY ?= python3

DATASETS        := nsl-kdd cicids2017 unsw_nb15
METHODS         := hardened curriculum rsc
NSL_SEEDS       := 42 43 44
CICIDS_SEEDS    := 42 43 44 45 46 47 48 49 50 51 52 53 54
UNSW_SEEDS      := 42 43 44 46 47 48 49 50 51

.PHONY: help setup verify \
        reproduce-table3 reproduce-table-mixednorm \
        reproduce-multiseed-nslkdd reproduce-multiseed-cicids reproduce-multiseed-unsw \
        reproduce-certified reproduce-scalability reproduce-jsma \
        reproduce-logit-trace reproduce-canonical-table \
        smoke reproduce-all clean

help:
	@echo "Targets:"
	@echo "  make setup                          create venv, pin deps, fetch NSL-KDD"
	@echo "  make verify                         checkpoint SHA-256 manifest check"
	@echo "  make reproduce-table3               Section III faithful diagnostic (Table 3, AdvGuard)"
	@echo "  make reproduce-table-mixednorm      mixed-norm K=0/1/2 collapse (Tab.MixedNorm, NSL-KDD)"
	@echo "  make reproduce-multiseed-nslkdd     EXH K=1 pgd40 sweep, NSL-KDD, 9 models"
	@echo "  make reproduce-multiseed-cicids     EXH K=1 pgd40 sweep, CICIDS2017, 39 models"
	@echo "  make reproduce-multiseed-unsw       EXH K=1 pgd40 sweep, UNSW-NB15, 27 models"
	@echo "  make reproduce-certified            CROWN vs IBP certified rates, 27 checkpoints"
	@echo "  make reproduce-scalability          per-state wall-clock timing, UNSW-NB15 K=1/K=2"
	@echo "  make reproduce-jsma                 JSMA vs exhaustive-image attack comparison"
	@echo "  make reproduce-logit-trace          logit-reversal trace on the hardened NSL model"
	@echo "  make reproduce-canonical-table      aggregate summary across all sweeps"
	@echo "  make smoke                          quick self-test (skips heavy sweeps)"

# ---------------------------------------------------------------------------
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	$(PY) download_data.py        # fetches + preprocesses NSL-KDD (~120 MB download)

verify:
	$(PY) scripts/verify_manifest.py

# --- Table 3 (AdvGuard faithful diagnostic) ------------------------------
# ~45 s (CUDA) / ~4 min (CPU). NOTE: faithful protocol uses an UNSEEDED
# random-start PGD; single-run results jitter ~±0.4 pp. Archive value for the
# supported-claims row (hardened @ eps=0.15) is 40.36%; the qualitative claim
# (baseline ~16%, hardened ~40%, vs the originally-reported ~29.1%) is the
# object of reproduction.
reproduce-table3:
	$(PY) scripts/section3_faithful_diagnostic.py --device cuda

# --- Mixed-norm K=0/1/2 collapse (canonical exhaustive evaluator) ---------
# ~26 s. Archival values (NSL-KDD, 22,543): Baseline 3342/0/0,
# Curriculum 5604/9/9, Hardened 17421/0/0, RSC 18106/0.
reproduce-table-mixednorm:
	$(PY) scripts/run_paper_mixednorm.py --check

# --- Multi-seed EXH K=1 sweeps (eval_deepfool_k1.py, CPU, random-start PGD) --
# Each (dataset,method,seed) derives candidates for every one-hot state and
# records K=0/K=1 survivors with checkpoint SHA-256 embedded. Run-to-run
# variance from unseeded random PGD init is ~±0.5 pp on k1_pct_of_attacked;
# the comparator tolerates it.
# Runtimes (reference machine, CPU): NSL-KDD ~1 min/model, CICIDS2017 ~8 min/model,
# UNSW-NB15 ~20 min/model. Serial totals: make ...-nslkdd ~10 min,
# -cicids ~6 h, -unsw ~9-10 h. Parallelize with e.g. xargs -P.
reproduce-multiseed-nslkdd:
	@set -e; for m in $(METHODS); do for s in $(NSL_SEEDS); do \
	  $(PY) scripts/eval_deepfool_k1.py --attack pgd40 --dataset nsl-kdd --method $$m --seed $$s --suffix .fresh; \
	done; done
	$(PY) scripts/compare_exh_fresh.py --attack pgd40 --dataset nsl-kdd

reproduce-multiseed-cicids:
	@set -e; for m in $(METHODS); do for s in $(CICIDS_SEEDS); do \
	  $(PY) scripts/eval_deepfool_k1.py --attack pgd40 --dataset cicids2017 --method $$m --seed $$s --suffix .fresh; \
	done; done
	$(PY) scripts/compare_exh_fresh.py --attack pgd40 --dataset cicids2017

reproduce-multiseed-unsw:
	@set -e; for m in $(METHODS); do for s in $(UNSW_SEEDS); do \
	  $(PY) scripts/eval_deepfool_k1.py --attack pgd40 --dataset unsw_nb15 --method $$m --seed $$s --suffix .fresh; \
	done; done
	$(PY) scripts/compare_exh_fresh.py --attack pgd40 --dataset unsw_nb15

# --- Certified (Tables VIII-IX): CROWN vs IBP, K=0 and K=1 -----------------
# ~5 s (NSL-KDD) / ~35 s (CICIDS2017/UNSW-NB15) per checkpoint; full 27 =
# ~12-15 min. Certified counts are deterministic and byte-identical to archive.
reproduce-certified:
	@set -e; for d in $(DATASETS); do for m in $(METHODS); do for s in $(NSL_SEEDS); do \
	  $(PY) scripts/run_crown_vs_ibp.py --dataset $$d --method $$m --seed $$s; \
	done; done; done

# --- Scalability (Section VII-G): per-state wall-clock timing --------------
# ~2 min with CUDA, ~10-15 min on CPU. Values measured: K=1 (42 states)
# 0.0155 s/state, K=2 (667 states) 0.118 s/state on the reference GPU.
reproduce-scalability:
	$(PY) scripts/eval_scalability.py --dataset unsw_nb15 --method rsc --seed 42 \
	  --K 1 --eligible-groups 0,1,2,3,4 --num-samples 500 --out results/scalability_unsw_nb15_K1_g0_1_2_3_4.repro.json
	$(PY) scripts/eval_scalability.py --dataset unsw_nb15 --method rsc --seed 42 \
	  --K 2 --eligible-groups 0,1,2,3,4 --num-samples 500 --out results/scalability_unsw_nb15_K2_g0_1_2_3_4.repro.json

# --- JSMA vs exhaustive-image (Section VII-H) -------------------------------
# ~1 min. UNSW-NB15, rsc seed 42, first 500 clean-correct test samples.
reproduce-jsma:
	$(PY) scripts/eval_jsma_vs_exhaustive.py --dataset unsw_nb15 --method rsc --seed 42 \
	  --K 1 --num-samples 500 --out results/jsma_vs_exhaustive_unsw_nb15_K1.repro.json

# --- Logit reversal trace (Table III prose) ---------------------------------
# ~4 s. Deterministic; reproduces the shipped trace byte-for-byte.
reproduce-logit-trace:
	$(PY) scripts/trace_logit_reversal_hardened.py

# --- Aggregated canonical table (all-sweep summary) -------------------------
# <1 s. Pure aggregation over the shipped results JSONs.
reproduce-canonical-table:
	$(PY) scripts/consolidated_canonical_table.py

# --- Smoke: fastest coverage of all verification machinery ------------------
smoke:
	$(PY) scripts/verify_manifest.py
	$(PY) scripts/run_paper_mixednorm.py --check
	$(PY) scripts/trace_logit_reversal_hardened.py
	$(PY) scripts/consolidated_canonical_table.py

reproduce-all: reproduce-table3 reproduce-table-mixednorm \
	reproduce-multiseed-nslkdd reproduce-certified reproduce-scalability \
	reproduce-jsma reproduce-logit-trace reproduce-canonical-table

clean:
	rm -rf results_fresh results/*.repro.* results/foolbox/*.fresh* __pycache__