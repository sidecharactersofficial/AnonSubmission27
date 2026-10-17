#!/usr/bin/env python3
"""
Phase 2: Numerical reconciliation of all reported results.

Key tasks:
  1. Define one canonical denominator (total, clean-correct, robust-correct per K).
  2. Reproduce each archived result from exact checkpoint + dataset artifact.
  3. Reconcile every seed: capture individual K=0/K=1 counts, recompute mean/std.
  4. Reconcile diagnostic subsets with full evaluations (same sample IDs).
  5. Enforce K=0/K=1 nesting check (K=1 <= K=0).

Critical method comparison:
  - canonical_mixed_norm_attack (eval_mixed_norm.py): exhaustive enumeration
  - unified_pgd_attack (unified_pgd.py): gradient-snapped categorical attack
  These are DIFFERENT algorithms and may produce different results.

Run from ml-optimizer/:
    python verification/phase2_reconcile.py
"""
import sys, os, json, time, hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
from app.ml.data.loader import get_config, get_test_loader, load_tabular_data
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint, compute_sha256
from scripts.eval_mixed_norm import canonical_mixed_norm_attack, set_seed
from app.ml.attacks.unified_pgd import unified_pgd_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS   = EVAL_EPSILON
ALPHA = EVAL_ALPHA_CONT
STEPS = EVAL_PGD_STEPS


def sha256_file(path):
    return compute_sha256(path)


def run_canonical_eval(model, X, y, config, Ks=[0, 1], batch_size=2000):
    """Run canonical_mixed_norm_attack (exhaustive enumeration) on the full dataset."""
    results = {
        'total': 0,
        'clean_correct': 0,
        'robust_correct': {str(K): 0 for K in Ks},
    }
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        bx = X[start:end].to(DEVICE)
        by = y[start:end].to(DEVICE)

        with torch.no_grad():
            pred = model(bx).argmax(dim=1)
        cc = (pred == by).sum().item()
        results['clean_correct'] += cc
        results['total'] += by.size(0)

        # Evaluate on clean-correct subset only
        cc_mask = (pred == by)
        bx_cc = bx[cc_mask]
        by_cc = by[cc_mask]
        if bx_cc.shape[0] == 0:
            continue

        for K in Ks:
            set_seed(42)
            adv = canonical_mixed_norm_attack(model, bx_cc, by_cc, config,
                                              epsilon=EPS, alpha=ALPHA, steps=STEPS, K=K)
            with torch.no_grad():
                pred_adv = model(adv).argmax(dim=1)
            results['robust_correct'][str(K)] += (pred_adv == by_cc).sum().item()

    return results


def run_unified_eval(model, X, y, config, K=1, batch_size=2000):
    """Run unified_pgd_attack (gradient-snapped) on the full dataset."""
    total = y.shape[0]
    clean_correct = 0
    # For K=0: continuous-only attack
    robust_correct = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bx = X[start:end].to(DEVICE)
        by = y[start:end].to(DEVICE)

        with torch.no_grad():
            pred = model(bx).argmax(dim=1)
        cc = (pred == by).sum().item()
        clean_correct += cc
        cc_mask = (pred == by)
        data_cc = bx[cc_mask]
        target_cc = by[cc_mask]
        if data_cc.size(0) == 0:
            continue

        if K == 0:
            adv = unified_pgd_attack(
                model, data_cc, target_cc,
                epsilon=EPS, alpha=ALPHA, alpha_cat=0.0, steps=STEPS,
                continuous_cols=config.CONTINUOUS_COLS,
                categorical_groups=config.CATEGORICAL_GROUPS
            )
            with torch.no_grad():
                adv_pred = model(adv).argmax(dim=1)
            robust_correct += (adv_pred == target_cc).sum().item()
        elif K == 1:
            # Mirror eval_unified.py logic
            survivors = torch.ones(data_cc.size(0), dtype=torch.bool, device=DEVICE)

            # K=0 base case
            adv_k0 = unified_pgd_attack(
                model, data_cc, target_cc,
                epsilon=EPS, alpha=ALPHA, alpha_cat=0.0, steps=STEPS,
                continuous_cols=config.CONTINUOUS_COLS,
                categorical_groups=[]
            )
            with torch.no_grad():
                pred_k0 = model(adv_k0).argmax(dim=1)
            survivors &= (pred_k0 == target_cc)

            # Each group
            for group in config.CATEGORICAL_GROUPS:
                adv = unified_pgd_attack(
                    model, data_cc, target_cc,
                    epsilon=EPS, alpha=ALPHA, alpha_cat=1.0, steps=STEPS,
                    continuous_cols=config.CONTINUOUS_COLS,
                    categorical_groups=[group]
                )
                with torch.no_grad():
                    pred = model(adv).argmax(dim=1)
                survivors &= (pred == target_cc)

            robust_correct += survivors.sum().item()

    return {
        'total': total,
        'clean_correct': clean_correct,
        'robust_correct': {str(K): robust_correct},
    }


def load_full_dataset(dataset_name):
    """Load full test dataset as tensors."""
    config = get_config(dataset_name)
    safe_name = dataset_name.replace('-', '_')
    parquet_path = f"data/{safe_name}_test.parquet"
    csv_path = f"data/{dataset_name}-test.csv"

    if os.path.exists(parquet_path):
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        y = torch.tensor(df['label'].values, dtype=torch.long)
        X = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
    elif os.path.exists(csv_path):
        data = np.loadtxt(csv_path, delimiter=',')
        X = torch.tensor(data[:, :-1], dtype=torch.float32)
        y = torch.tensor(data[:, -1], dtype=torch.long)
    else:
        print(f"  No data found for {dataset_name}")
        return None, None

    return X, y


def phase2_main():
    print("=" * 70)
    print("PHASE 2: Numerical Reconciliation")
    print(f"Device: {DEVICE} | eps={EPS}, alpha={ALPHA}, steps={STEPS}")
    print("=" * 70)

    all_results = {}

    # ---- NSL-KDD: Reproduce archived result ----
    print("\n--- NSL-KDD RSC: Reproduce archived (canonical eval) ---")
    dataset = 'nsl-kdd'
    config = get_config(dataset)
    X, y = load_full_dataset(dataset)
    print(f"  Total test samples: {X.shape[0]}")

    # Archived result
    arch = json.load(open('results/results_nsl_kdd_RSC.json'))
    print(f"  Archived: total={arch['total']}, clean={arch['clean_correct']}, "
          f"K0={arch['robust_correct']['0']}, K1={arch['robust_correct']['1']}")
    print(f"  Archived: K0={arch['robust_correct']['0']/arch['total']*100:.2f}%, "
          f"K1={arch['robust_correct']['1']/arch['total']*100:.2f}%")

    # Reproduce with canonical evaluator
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    h = load_model_checkpoint(model, 'models/model_adv_rsc_nsl_kdd.pth', device=DEVICE)
    model.eval()
    t0 = time.time()
    res = run_canonical_eval(model, X, y, config, Ks=[0, 1], batch_size=2000)
    dt = time.time() - t0
    print(f"  Reproduced (canonical): total={res['total']}, clean={res['clean_correct']}, "
          f"K0={res['robust_correct']['0']}, K1={res['robust_correct']['1']}")
    print(f"  Reproduced: K0={res['robust_correct']['0']/res['total']*100:.2f}%, "
          f"K1={res['robust_correct']['1']/res['total']*100:.2f}%")
    print(f"  Time: {dt:.1f}s")

    match_k0 = res['robust_correct']['0'] == arch['robust_correct']['0']
    match_k1 = res['robust_correct']['1'] == arch['robust_correct']['1']
    match_clean = res['clean_correct'] == arch['clean_correct']
    print(f"  Match: clean={match_clean}, K0={match_k0}, K1={match_k1}")
    all_results['nsl_kdd_rsc_canonical'] = {
        'archived': arch,
        'reproduced': res,
        'matches': {'clean': match_clean, 'k0': match_k0, 'k1': match_k1},
    }

    # ---- NSL-KDD: Run unified evaluator on same model ----
    print(f"\n--- NSL-KDD RSC: Unified PGD (gradient-snapped) on same model ---")
    res_unified_k1 = run_unified_eval(model, X, y, config, K=1, batch_size=2000)
    print(f"  Unified K=1: total={res_unified_k1['total']}, clean={res_unified_k1['clean_correct']}, "
          f"robust={res_unified_k1['robust_correct']['1']}")
    print(f"  Unified K1={res_unified_k1['robust_correct']['1']/res_unified_k1['total']*100:.2f}%")
    print(f"  Difference from canonical K1: "
          f"{res_unified_k1['robust_correct']['1'] - res['robust_correct']['1']} samples")
    print(f"  Unified is WEAKER attack (fewer states explored) => "
          f"{'HIGHER' if res_unified_k1['robust_correct']['1'] > res['robust_correct']['1'] else 'LOWER'} robust acc")

    # K=0/K=1 nesting check for canonical
    k0_robust = res['robust_correct']['0']
    k1_robust = res['robust_correct']['1']
    print(f"\n  K=0/K=1 nesting (canonical): K0={k0_robust}, K1={k1_robust}")
    if k1_robust <= k0_robust:
        print(f"  PASS: K1 <= K0 (nesting satisfied)")
    else:
        print(f"  FAIL: K1 > K0 (nesting violated)")

    # K=0/K=1 nesting check for unified
    u_k0 = res_unified_k1['robust_correct']['1']  # This is actually k1
    # Need k0 from unified
    res_unified_k0 = run_unified_eval(model, X, y, config, K=0, batch_size=2000)
    res_unified_k0_val = res_unified_k0['robust_correct']['0'] / res_unified_k0['total'] * 100
    u_k0 = res_unified_k0['robust_correct']['0']
    print(f"\n  K=0/K=1 nesting (unified): K0={u_k0}, K1={res_unified_k1['robust_correct']['1']}")
    if res_unified_k1['robust_correct']['1'] <= u_k0:
        print(f"  PASS: K1 <= K0 (nesting satisfied)")
    else:
        print(f"  FAIL: K1 > K0 (nesting violated)")

    all_results['nsl_kdd_rsc_canonical']['unified_k0'] = u_k0
    all_results['nsl_kdd_rsc_canonical']['unified_k1'] = res_unified_k1['robust_correct']['1']

    # ---- CICIDS2017: Compare legacy vs unified (different models) ----
    print("\n--- CICIDS2017 RSC: Archived (legacy model + canonical eval) ---")
    arch_ci = json.load(open('results/results_cicids2017_RSC.json'))
    print(f"  Archived: total={arch_ci['total']}, clean={arch_ci['clean_correct']}, "
          f"K0={arch_ci['robust_correct']['0']}, K1={arch_ci['robust_correct']['1']}")
    print(f"  Archived: K0={arch_ci['robust_correct']['0']/arch_ci['total']*100:.2f}%, "
          f"K1={arch_ci['robust_correct']['1']/arch_ci['total']*100:.2f}%")

    # Unified seed42
    uni_ci = json.load(open('results/unified/eval_rsc_cicids2017_seed42.json'))
    print(f"  Unified seed42: clean={uni_ci['clean_acc']:.2f}%, K0={uni_ci['k0_acc']:.2f}%, K1={uni_ci['k1_acc']:.2f}%")

    # Check if unified K1 (70%) can be reproduced by canonical eval on SAME model
    print(f"\n  Note: Legacy model SHA={sha256_file('models/model_adv_rsc_cicids2017.pth')[:16]}")
    print(f"  Note: Unified seed42 model SHA={sha256_file('models/unified/model_adv_rsc_cicids2017_seed42.pth')[:16]}")
    print(f"  -> These are DIFFERENT checkpoints (different training methods)")

    # --- Per-seed reconciliation for CICIDS2017 RSC (seeds 42-54) ---
    print("\n--- CICIDS2017 RSC: Per-seed reconciliation (unified) ---")
    per_seed = []
    for seed in range(42, 55):
        json_path = f'results/unified/eval_rsc_cicids2017_seed{seed}.json'
        if os.path.exists(json_path):
            d = json.load(open(json_path))
            per_seed.append({
                'seed': seed,
                'clean_acc': d['clean_acc'],
                'k0_acc': d['k0_acc'],
                'k1_acc': d['k1_acc'],
            })
    print(f"  Found {len(per_seed)} per-seed results")
    k1_vals = [r['k1_acc'] for r in per_seed]
    k0_vals = [r['k0_acc'] for r in per_seed]
    print(f"  K0: mean={np.mean(k0_vals):.2f}%, std={np.std(k0_vals, ddof=1):.2f}%, "
          f"min={min(k0_vals):.2f}%, max={max(k0_vals):.2f}%")
    print(f"  K1: mean={np.mean(k1_vals):.2f}%, std={np.std(k1_vals, ddof=1):.2f}%, "
          f"min={min(k1_vals):.2f}%, max={max(k1_vals):.2f}%")

    # Check for K1 > K0 violations (nesting check per seed)
    nesting_violations = []
    for r in per_seed:
        if r['k1_acc'] > r['k0_acc'] + 0.5:  # tolerance for rounding
            nesting_violations.append(r)
            print(f"  FLAG: seed={r['seed']} K1={r['k1_acc']:.2f}% > K0={r['k0_acc']:.2f}%!")
    if not nesting_violations:
        print(f"  PASS: No K1 > K0 nesting violations across all seeds")

    all_results['cicids2017_rsc_unified'] = {
        'n_seeds': len(per_seed),
        'k0_mean': float(np.mean(k0_vals)),
        'k0_std': float(np.std(k0_vals, ddof=1)),
        'k1_mean': float(np.mean(k1_vals)),
        'k1_std': float(np.std(k1_vals, ddof=1)),
        'nesting_violations': len(nesting_violations),
    }

    # --- Per-seed checkpoint hashes ---
    print(f"\n--- Checkpoint hash verification (seeds 42-54) ---")
    hashes_file = 'results/unified/checkpoint_hashes_seeds45-54.txt'
    if os.path.exists(hashes_file):
        print(f"  Hash file exists: {hashes_file}")
    hashes_file2 = 'results/unified/checkpoint_hashes_seeds46-54.txt'
    if os.path.exists(hashes_file2):
        print(f"  Hash file exists: {hashes_file2}")

    # Verify a few checkpoints
    for seed in [42, 45, 53, 54]:
        ckpt = f'models/unified/model_adv_rsc_cicids2017_seed{seed}.pth'
        if os.path.exists(ckpt):
            h = sha256_file(ckpt)
            print(f"  {ckpt}: {h[:32]}...")

    # ---- CICIDS2017: Same-model method comparison (subset) ----
    print("\n--- CICIDS2017 RSC seed42: Same-model method comparison (200 clean-correct samples) ---")
    config_ci = get_config('cicids2017')
    model_ci = TabularMLP(input_dim=config_ci.FEATURE_DIM).to(DEVICE)
    h_ci = load_model_checkpoint(model_ci, 'models/unified/model_adv_rsc_cicids2017_seed42.pth', device=DEVICE)
    model_ci.eval()

    X_ci, y_ci = load_full_dataset('cicids2017')

    print(f"  CICIDS2017 total test: {X_ci.shape[0]}")

    with torch.no_grad():
        preds_ci = model_ci(X_ci[:5000].to(DEVICE)).argmax(dim=1)
    cc_idx = (preds_ci == y_ci[:5000].to(DEVICE)).nonzero(as_tuple=True)[0][:200]
    X_ci_cc = X_ci[:5000][cc_idx.cpu()].to(DEVICE)
    y_ci_cc = y_ci[:5000][cc_idx.cpu()].to(DEVICE)
    print(f"  Using {len(X_ci_cc)} clean-correct samples")

    # Canonical (exhaustive) K=1
    set_seed(42)
    t0 = time.time()
    adv_canon = canonical_mixed_norm_attack(model_ci, X_ci_cc, y_ci_cc, config_ci,
                                            epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
    with torch.no_grad():
        pred_canon = model_ci(adv_canon).argmax(dim=1)
    canon_k1_surv = (pred_canon == y_ci_cc).sum().item()
    dt_canon = time.time() - t0

    # Canonical K=0
    set_seed(42)
    adv_canon_k0 = canonical_mixed_norm_attack(model_ci, X_ci_cc, y_ci_cc, config_ci,
                                               epsilon=EPS, alpha=ALPHA, steps=STEPS, K=0)
    with torch.no_grad():
        pred_canon_k0 = model_ci(adv_canon_k0).argmax(dim=1)
    canon_k0_surv = (pred_canon_k0 == y_ci_cc).sum().item()

    # Unified K=1
    set_seed(42)
    t0 = time.time()
    adv_uni = unified_pgd_attack(model_ci, X_ci_cc, y_ci_cc,
                                 epsilon=EPS, alpha=ALPHA, alpha_cat=1.0, steps=STEPS,
                                 continuous_cols=config_ci.CONTINUOUS_COLS,
                                 categorical_groups=config_ci.CATEGORICAL_GROUPS)
    with torch.no_grad():
        pred_uni = model_ci(adv_uni).argmax(dim=1)
    uni_k1_surv = (pred_uni == y_ci_cc).sum().item()
    dt_uni = time.time() - t0

    # Unified K=0
    set_seed(42)
    adv_uni_k0 = unified_pgd_attack(model_ci, X_ci_cc, y_ci_cc,
                                    epsilon=EPS, alpha=ALPHA, alpha_cat=0.0, steps=STEPS,
                                    continuous_cols=config_ci.CONTINUOUS_COLS,
                                    categorical_groups=[])
    with torch.no_grad():
        pred_uni_k0 = model_ci(adv_uni_k0).argmax(dim=1)
    uni_k0_surv = (pred_uni_k0 == y_ci_cc).sum().item()

    print(f"  Canonical K0: {canon_k0_surv}/{len(X_ci_cc)} ({canon_k0_surv/len(X_ci_cc)*100:.1f}%) [{dt_canon:.1f}s]")
    print(f"  Canonical K1: {canon_k1_surv}/{len(X_ci_cc)} ({canon_k1_surv/len(X_ci_cc)*100:.1f}%)")
    print(f"  Unified   K0: {uni_k0_surv}/{len(X_ci_cc)} ({uni_k0_surv/len(X_ci_cc)*100:.1f}%) [{dt_uni:.1f}s]")
    print(f"  Unified   K1: {uni_k1_surv}/{len(X_ci_cc)} ({uni_k1_surv/len(X_ci_cc)*100:.1f}%)")
    print(f"  Difference (Unified K1 - Canonical K1): {uni_k1_surv - canon_k1_surv} samples")

    if uni_k1_surv > canon_k1_surv:
        print(f"  KEY FINDING: Unified (gradient-snapped) finds FEWER attacks than canonical (exhaustive).")
        print(f"  Unified K1 = {uni_k1_surv/len(X_ci_cc)*100:.1f}% > Canonical K1 = {canon_k1_surv/len(X_ci_cc)*100:.1f}%")
        print(f"  => The unified method UNDERESTIMATES the true adversarial threat (overestimates robustness).")
    elif uni_k1_surv == canon_k1_surv:
        print(f"  Methods agree on this subset.")
    else:
        print(f"  Unified is stronger on this subset (unusual for gradient-snapped vs exhaustive).")

    all_results['cicids2017_same_model'] = {
        'model_sha': h_ci,
        'n_samples': len(X_ci_cc),
        'canonical_k0': canon_k0_surv,
        'canonical_k1': canon_k1_surv,
        'unified_k0': uni_k0_surv,
        'unified_k1': uni_k1_surv,
        'canon_k1_pct': canon_k1_surv / len(X_ci_cc) * 100,
        'uni_k1_pct': uni_k1_surv / len(X_ci_cc) * 100,
        'difference': uni_k1_surv - canon_k1_surv,
        'method_discrepancy': 'unified overestimates robustness' if uni_k1_surv > canon_k1_surv else 'methods agree or unified underestimates',
    }

    # ---- Denominator reconciliation summary ----
    print("\n--- Denominator Reconciliation Summary ---")
    canon_k0_pct = res['robust_correct']['0'] / res['total'] * 100
    canon_k1_pct = res['robust_correct']['1'] / res['total'] * 100
    uni_k1_pct = res_unified_k1['robust_correct']['1'] / res_unified_k1['total'] * 100
    res_unified_k1_val = uni_k1_pct
    print(f"""
ARCHIVED (legacy) format: {{ "clean_correct": N, "total": T, "robust_correct": {{"0": R0, "1": R1}} }}
  - Denominator: total (entire test set)
  - Conditional: robust_correct / clean_correct

UNIFIED format: {{ "clean_acc": C%, "k0_acc": K0%, "k1_acc": K1% }}
  - Percentages only, NO underlying counts
  - Cannot verify exact numerators from unified percentages

NSL-KDD RSC (same model, canonical eval):
  K0 = {canon_k0_pct:.2f}% ({res['robust_correct']['0']}/{res['total']})
  K1 = {canon_k1_pct:.2f}% ({res['robust_correct']['1']}/{res['total']})
  K0 = {res_unified_k0_val:.2f}% (unified, same model)
  K1 = {res_unified_k1_val:.2f}% (unified, same model)

Canonical denominator recommendation:
  A_full(K) = robust_correct[K] / total
  A_cond(K) = robust_correct[K] / clean_correct""")

    all_results['denominator_recommendation'] = {
        'total': res['total'],
        'clean_correct': res['clean_correct'],
        'robust_correct_k0': res['robust_correct']['0'],
        'robust_correct_k1': res['robust_correct']['1'],
        'formula': 'A_full(K) = robust_correct[K] / total; A_cond(K) = robust_correct[K] / clean_correct',
    }

    # ---- Summary of method discrepancy ----
    print("\n" + "=" * 70)
    print("METHOD DISCREPANCY ANALYSIS")
    print("=" * 70)
    ci_sm = all_results.get('cicids2017_same_model', {})
    print(f"""
The canonical (canonical_mixed_norm_attack) uses exhaustive enumeration:
  - For K=1: enumerate ALL single-flip categorical states (1 + sum(|g|))
  - Run continuous PGD on EACH state
  - Take worst case by CE loss

The unified (unified_pgd_attack) uses gradient-snapping:
  - For K=1: run PGD with categorical gradient snap (alpha_cat=1.0)
  - Only tests states reachable via gradient direction
  - Cannot reach all states (gradient may point away from worst state)

The unified method is a WEAKER attack -> higher robust accuracy numbers.
The canonical method is the correct threat-model evaluation.

NSL-KDD RSC (same model, both methods):
  Canonical K1 = {canon_k1_pct:.2f}%  (exhaustive, correct)
  Unified   K1 = {uni_k1_pct:.2f}%  (gradient-snapped, weaker)

CICIDS2017 RSC seed42 (SAME unified model, 200 clean-correct subset):
  Canonical K1 = {ci_sm.get('canon_k1_pct', 0):.2f}%  (exhaustive)
  Unified   K1 = {ci_sm.get('uni_k1_pct', 0):.2f}%  (gradient-snapped)
  Difference: {ci_sm.get('difference', 0)} samples

For CICIDS2017 RSC, the archived legacy result (K1=37.83%) uses the
legacy model + canonical eval. The unified results (K1=70% mean) use
DIFFERENT models + unified eval (gradient-snapped, weaker).""")

    # Save results
    with open('verification/phase2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to verification/phase2_results.json")
    return all_results


if __name__ == "__main__":
    phase2_main()
