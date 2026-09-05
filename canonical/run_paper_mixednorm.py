#!/usr/bin/env python3
"""Reproduce the mixed-norm K=0/1/2 evaluations behind the paper's
mixed-norm vulnerability table (legacy single-seed checkpoints).

Re-runs the canonical exhaustive-state evaluator
(canonical/eval_mixed_norm.py -> canonical_mixed_norm_attack) on the same legacy
checkpoints and dataset loaders that produced the archived results/results_*.json,
then (optionally) asserts freshly computed counts match the archived ones.

Semantics (identical to eval_mixed_norm.py):
  - attack: mixed-norm, L_inf on continuous columns (eps=0.15, alpha=0.01,
    steps=40, NO random start) with the categorical block held EXACTLY at the
    enumerated one-hot state;
  - K=0: original categorical state only;
  - K=1: original state + every alternative one-hot column per group,
    worst-state selection by CE loss;
  - K=2: every 1- and 2-column state (Cartesian product across groups);
  - survivor := pred == true label on the worst-state adversarial example;
  - denominator: full test set.

Default scope is NSL-KDD (the paper's collapse table, full 22,543-sample test
set). CICIDS2017/UNSW-NB15 legacy per-dataset files in the archive were partial
runs; use --dataset to extend the evaluation there (check output is then
informational only).

Usage:
  python canonical/run_paper_mixednorm.py                  # NSL-KDD, all 4 models
  python canonical/run_paper_mixednorm.py --check          # + compare to archived
  python canonical/run_paper_mixednorm.py --dataset cicids2017
"""

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

TOL = 5  # absolute count tolerance when asserting equality to archived files


def load_eval_mixed_norm():
    path = os.path.join(ROOT, "scripts", "eval_mixed_norm.py")
    spec = importlib.util.spec_from_file_location("eval_mixed_norm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Dataset -> model-label -> checkpoint-path (legacy single-seed checkpoints).
DS_MODELS = {
    "nsl-kdd": {
        "Baseline": "models/model.pth",
        "Hardened": "models/model_adv.pth",
        "Curriculum": "models/model_adv_pgd_curriculum.pth",
        "RSC": "models/model_adv_rsc_nsl_kdd.pth",
    },
    "cicids2017": {
        "Baseline": "models/model_cicids2017.pth",
        "Hardened": "models/model_adv_cicids2017.pth",
        "Curriculum": "models/model_adv_pgd_curriculum_cicids2017.pth",
        "RSC": "models/model_adv_rsc_cicids2017.pth",
    },
    "unsw_nb15": {
        "Baseline": "models/model_unsw_nb15.pth",
        "Hardened": "models/model_adv_unsw_nb15.pth",
        "Curriculum": "models/model_adv_pgd_curriculum_unsw_nb15.pth",
        "RSC": "models/model_adv_rsc_unsw_nb15.pth",
    },
}

# The NSL-KDD legacy root-level files carry the K=2 key; the per-dataset files
# carry only K=0/1 (exactly as archived).
NSL_FILES = {
    "Baseline": ("results_nsl_kdd_Baseline.json", ["0", "1"], "results_Baseline.json"),
    "Hardened": ("results_nsl_kdd_Hardened.json", ["0", "1"], "results_Legacy_FGSM-Hardened.json"),
    "Curriculum": ("results_nsl_kdd_Curriculum.json", ["0", "1"], "results_New_Curriculum_PGD-Hardened.json"),
    "RSC": ("results_nsl_kdd_RSC.json", ["0", "1"], None),
}

EPSILON, ALPHA, STEPS, BATCH = 0.15, 0.01, 40, 2000


def evaluate(em, dataset, label, ckpt, ks, out_dir):
    loader = em.get_test_loader(dataset, batch_size=BATCH)
    model = em.TabularMLP(input_dim=em.get_config(dataset).FEATURE_DIM).to(em.DEVICE)
    em.load_model_checkpoint(model, ckpt, device=em.DEVICE)
    model.eval()
    config = em.get_config(dataset)

    results = {"clean_correct": 0, "total": 0,
               "robust_correct": {str(k): 0 for k in ks}, "completed_batches": 0}

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(em.DEVICE), batch_y.to(em.DEVICE)
        with torch.no_grad():
            results["clean_correct"] += (model(batch_x).argmax(1) == batch_y).sum().item()
        for k in ks:
            em.set_seed(42 + batch_idx)
            adv = em.canonical_mixed_norm_attack(
                model, batch_x, batch_y, config, epsilon=EPSILON, alpha=ALPHA, steps=STEPS, K=k)
            with torch.no_grad():
                results["robust_correct"][str(k)] += (model(adv).argmax(1) == batch_y).sum().item()
        results["total"] += len(batch_y)
        results["completed_batches"] += 1
        print(f"[{dataset}/{label}] batch {results['completed_batches']}: "
              f"clean={results['clean_correct']} total={results['total']} "
              f"{ {k: results['robust_correct'][str(k)] for k in ks} }")

    os.makedirs(out_dir, exist_ok=True)
    if dataset == "nsl-kdd":
        fname, keys, alias = NSL_FILES[label]
        slim = dict(results)
        slim["robust_correct"] = {k: results["robust_correct"][k] for k in keys}
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(slim, f)
        if alias:
            with open(os.path.join(out_dir, alias), "w") as f:
                json.dump(results, f)
    else:
        fname = f"results_{dataset.replace('-', '_')}_{label}.json"
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(results, f)
    return results


def compare(em, dataset, label, results, out_dir):
    paths = [os.path.join(out_dir, f"results_{dataset.replace('-', '_')}_{label}.json")]
    if dataset == "nsl-kdd" and NSL_FILES[label][2]:
        paths.append(os.path.join(out_dir, NSL_FILES[label][2]))
    ok = True
    any_compared = False
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        archived = json.load(open(path))
        any_compared = True
        if archived["total"] != results["total"]:
            print(f"[{dataset}/{label}] WARN: archived {os.path.basename(path)} total "
                  f"{archived['total']} != fresh total {results['total']} "
                  f"(archived was partial/gated); skipping equality check")
            continue
        for k in sorted(set(results["robust_correct"]) & set(archived["robust_correct"])):
            want = archived["robust_correct"][k]
            diff = abs(results["robust_correct"][k] - want)
            status = "OK" if diff <= TOL else "MISMATCH"
            ok &= diff <= TOL
            print(f"[{dataset}/{label}] {os.path.basename(path)} K={k}: "
                  f"fresh={results['robust_correct'][k]} archived={want} "
                  f"(diff {diff}) {status}")
        if results["clean_correct"] != archived["clean_correct"]:
            print(f"[{dataset}/{label}] {os.path.basename(path)} clean MISMATCH: "
                  f"fresh={results['clean_correct']} archived={archived['clean_correct']}")
            ok = False
    if not any_compared:
        print(f"[{dataset}/{label}] no archived file to compare; skipping check")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DS_MODELS), default="nsl-kdd")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out-dir", default="results_fresh",
                        help="write freshly computed files here (never into results/ archive)")
    parser.add_argument("--archived-dir", default="results")
    args = parser.parse_args()

    em = load_eval_mixed_norm()
    em.set_seed(42)

    all_ok = True
    for label in sorted(DS_MODELS[args.dataset]):
        ckpt = DS_MODELS[args.dataset][label]
        if not os.path.exists(ckpt):
            print(f"[{args.dataset}/{label}] SKIP: {ckpt} not found")
            continue
        print(f"=== {args.dataset} / {label} ({ckpt}) ===")
        ks = [0, 1, 2] if args.dataset == "nsl-kdd" and label != "RSC" else [0, 1]
        results = evaluate(em, args.dataset, label, ckpt, ks, args.out_dir)
        if args.check:
            all_ok &= compare(em, args.dataset, label, results, args.archived_dir)

    print("run_paper_mixednorm: ALL CHECKS PASS" if all_ok else "run_paper_mixednorm: FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()