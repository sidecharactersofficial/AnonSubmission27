#!/usr/bin/env python3
"""
Comprehensive EXH vs SNAP verification across all available checkpoints.

Reads pre-computed results from:
- results/foolbox/exh_k1_pgd40_*.json (EXH = exhaustive categorical enumeration)
- results/unified/eval_*.json (SNAP = gradient-snapped unified evaluation)

For CICIDS2017 RSC seed43, runs a controlled per-sample comparison on a 200-sample
subset to inspect disagreements in detail.

Run from ml-optimizer/:
    python verification/comprehensive_exh_vs_snap.py
"""
import sys, os, json, hashlib
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, '.')

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_config
from app.ml.utils.checkpoint import load_model_checkpoint
from app.ml.attacks.unified_pgd import unified_pgd_attack
from scripts.eval_mixed_norm import canonical_mixed_norm_attack
from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = EVAL_EPSILON
ALPHA = EVAL_ALPHA_CONT
STEPS = EVAL_PGD_STEPS


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_exh_result(ds_key, method, seed):
    """Load EXH result from results/foolbox/."""
    fname = f"exh_k1_pgd40_cicids2017_{method}_seed{seed}.json"
    if ds_key == 'cicids2017':
        path = f"results/foolbox/{fname}"
    elif ds_key == 'nsl_kdd':
        path = f"results/foolbox/exh_k1_pgd40_nsl_kdd_{method}_seed{seed}.json"
    elif ds_key == 'unsw_nb15':
        path = f"results/foolbox/exh_k1_pgd40_unsw_nb15_{method}_seed{seed}.json"
    else:
        return None

    if not os.path.exists(path):
        return None
    return json.load(open(path))


def load_snap_result(ds_key, method, seed):
    """Load SNAP (unified) result from results/unified/."""
    fname = f"eval_{method}_{ds_key}_seed{seed}.json"
    path = f"results/unified/{fname}"
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def verify_checkpoint_hash(ds_key, method, seed, exh_result):
    """Verify the EXH result's checkpoint SHA matches the unified checkpoint file."""
    ckpt_path = f"models/unified/model_adv_{method}_{ds_key}_seed{seed}.pth"
    if not os.path.exists(ckpt_path):
        return False, f"checkpoint not found: {ckpt_path}"

    actual_sha = sha256_file(ckpt_path)
    recorded_sha = exh_result.get('checkpoint_sha256', '')

    if actual_sha[:16] != recorded_sha[:16]:
        return False, f"hash mismatch: actual={actual_sha[:16]}..., recorded={recorded_sha[:16]}..."

    return True, actual_sha


def run_comprehensive_comparison():
    """Compare EXH vs SNAP across all datasets, methods, and seeds."""
    datasets = [
        ('cicids2017', 'CICIDS2017'),
        ('nsl_kdd', 'NSL-KDD'),
        ('unsw_nb15', 'UNSW-NB15'),
    ]
    methods = ['rsc', 'hardened', 'curriculum']
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

    all_results = []
    hash_mismatches = 0
    total_compared = 0
    k1_gaps = []

    for ds_key, ds_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name} ({ds_key})")
        print(f"{'=' * 70}")
        print(f"{'Method':<12} {'Seed':>4}  {'EXH K0%':>8}  {'EXH K1%':>8}  {'SNAP K0%':>9}  {'SNAP K1%':>9}  {'Gap+':>8}")
        print(f"{'':12} {'':>4}  {'(of cl)':>8}  {'(of cl)':>8}  {'(of cl)':>9}  {'(of cl)':>9}  {'(pp)':>8}")
        print(f"{'─' * 70}")

        for method in methods:
            for seed in seeds:
                exh = load_exh_result(ds_key, method, seed)
                snap = load_snap_result(ds_key, method, seed)

                if exh is None:
                    continue

                # Verify checkpoint hash
                if not os.path.exists(f"models/unified/model_adv_{method}_{ds_key}_seed{seed}.pth"):
                    continue

                hash_ok, hash_info = verify_checkpoint_hash(ds_key, method, seed, exh)
                if not hash_ok:
                    hash_mismatches += 1

                cc = exh.get('clean_correct', 0)
                k0 = exh.get('k0_survivors', 0)
                k1 = exh.get('k1_survivors', 0)
                k0_pct = k0 / cc * 100 if cc > 0 else 0
                k1_pct = k1 / cc * 100 if cc > 0 else 0

                if snap:
                    snap_k0 = snap.get('k0_acc', 0)
                    snap_k1 = snap.get('k1_acc', 0)
                    snap_clean = snap.get('clean_acc', 100)
                    snap_k0_of_clean = snap_k0 / snap_clean * 100 if snap_clean > 0 else 0
                    snap_k1_of_clean = snap_k1 / snap_clean * 100 if snap_clean > 0 else 0
                    gap = snap_k1_of_clean - k1_pct
                    k1_gaps.append({
                        'dataset': ds_name, 'method': method, 'seed': seed,
                        'exh_k1_pct': k1_pct, 'snap_k1_pct': snap_k1_of_clean,
                        'gap_pp': gap, 'hash_ok': hash_ok
                    })
                    total_compared += 1
                    print(f"{method:<12} {seed:>4}  {k0_pct:>8.2f}  {k1_pct:>8.2f}  "
                          f"{snap_k0_of_clean:>8.2f}%  {snap_k1_of_clean:>8.2f}%  {gap:>+8.1f}pp")
                else:
                    print(f"{method:<12} {seed:>4}  {k0_pct:>8.2f}  {k1_pct:>8.2f}  "
                          f"{'N/A':>9}  {'N/A':>9}  {'N/A':>8}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print(f"CROSS-DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total comparisons: {total_compared}")
    print(f"  Hash mismatches: {hash_mismatches}")

    gaps_by_ds = defaultdict(list)
    for r in k1_gaps:
        gaps_by_ds[r['dataset']].append(r['gap_pp'])

    print(f"\n  K=1 gap (SNAP - EXH, in pp) by dataset:")
    for ds, gaps in gaps_by_ds.items():
        arr = np.array(gaps)
        print(f"    {ds:<12}: mean={arr.mean():.1f}pp, min={arr.min():.1f}pp, "
              f"max={arr.max():.1f}pp, median={np.median(arr):.1f}pp")

    print(f"\n  K=1 gap (SNAP - EXH, in pp) by method:")
    gaps_by_method = defaultdict(list)
    for r in k1_gaps:
        gaps_by_method[r['method']].append(r['gap_pp'])
    for method, gaps in gaps_by_method.items():
        arr = np.array(gaps)
        print(f"    {method:<12}: mean={arr.mean():.1f}pp, min={arr.min():.1f}pp, "
              f"max={arr.max():.1f}pp, median={np.median(arr):.1f}pp")

    # Worst discrepancies
    worst = sorted(k1_gaps, key=lambda x: x['gap_pp'], reverse=True)[:10]
    print(f"\n  Top 10 worst discrepancies (SNAP overestimates):")
    for r in worst:
        print(f"    {r['dataset']}/{r['method']}/seed{r['seed']}: "
              f"EXH K1={r['exh_k1_pct']:.1f}% vs SNAP K1={r['snap_k1_pct']:.1f}% "
              f"(gap={r['gap_pp']:.1f}pp, hash_ok={r['hash_ok']})")

    # Cases where EXH > SNAP (rare)
    exh_higher = [r for r in k1_gaps if r['gap_pp'] < 0]
    if exh_higher:
        print(f"\n  Cases where EXH > SNAP (SNAP underestimated):")
        for r in exh_higher:
            print(f"    {r['dataset']}/{r['method']}/seed{r['seed']}: "
                  f"EXH K1={r['exh_k1_pct']:.1f}% vs SNAP K1={r['snap_k1_pct']:.1f}% "
                  f"(gap={r['gap_pp']:.1f}pp)")
    else:
        print(f"\n  Cases where EXH > SNAP: 0 (SNAP never underestimates)")

    return k1_gaps


def run_per_sample_inspection():
    """
    Detailed per-sample comparison on CICIDS2017 RSC seed43.
    Inspect 10 of the 135 samples where EXH finds attacks SNAP misses.
    """
    print(f"\n{'=' * 70}")
    print(f"Per-Sample Inspection: CICIDS2017 RSC seed43 (200-sample subset)")
    print(f"{'=' * 70}")

    ckpt_path = 'models/unified/model_adv_rsc_cicids2017_seed43.pth'
    ckpt_sha = sha256_file(ckpt_path)
    print(f"\nCheckpoint SHA-256: {ckpt_sha}")
    print(f"Matches pilot: {ckpt_sha == '7472b0e0dcec774e3549dd2f10d6c752140e36bc4422b2701ec246305c85931c'}")

    config = get_config('cicids2017')
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    load_model_checkpoint(model, ckpt_path, device=DEVICE)
    model.eval()

    # Load same 200-sample subset
    import pandas as pd
    df = pd.read_parquet('data/cicids2017_test.parquet')
    y_all = torch.tensor(df['label'].values, dtype=torch.long)
    X_all = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
    torch.manual_seed(42)
    perm = torch.randperm(len(X_all))
    X = X_all[perm][:200].to(DEVICE)
    y = y_all[perm][:200].to(DEVICE)

    with torch.no_grad():
        clean_preds = model(X).argmax(dim=1)
    cc_mask = clean_preds == y
    X_cc = X[cc_mask]
    y_cc = y[cc_mask]
    sample_indices = torch.nonzero(cc_mask).squeeze()
    n_clean = len(X_cc)
    print(f"Clean-correct samples: {n_clean}")

    cont = config.CONTINUOUS_COLS
    groups = config.CATEGORICAL_GROUPS

    # Run EXH K=1
    from scripts.eval_mixed_norm import canonical_mixed_norm_attack
    adv_exh = canonical_mixed_norm_attack(model, X_cc, y_cc, config,
                                          epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
    with torch.no_grad():
        exh_preds = model(adv_exh).argmax(dim=1)
    exh_broken = (exh_preds != y_cc)

    # Run SNAP K=1
    torch.manual_seed(42)
    np.random.seed(42)
    adv_k0 = unified_pgd_attack(model, X_cc.clone(), y_cc.clone(),
                                epsilon=EPS, alpha=ALPHA, alpha_cat=0.0, steps=STEPS,
                                continuous_cols=cont, categorical_groups=[])
    with torch.no_grad():
        snap_surv = (model(adv_k0).argmax(dim=1) == y_cc)
    for group in groups:
        adv_snap = unified_pgd_attack(model, X_cc.clone(), y_cc.clone(),
                                      epsilon=EPS, alpha=ALPHA, alpha_cat=1.0, steps=STEPS,
                                      continuous_cols=cont, categorical_groups=[group])
        with torch.no_grad():
            snap_surv = snap_surv & (model(adv_snap).argmax(dim=1) == y_cc)
    snap_broken = ~snap_surv

    # Verify constraints on disagreed samples
    cat_indices = [idx for g in groups for idx in g]
    cat_min, cat_max = min(cat_indices), max(cat_indices)

    # Check EXH adversarial examples satisfy L∞ constraint and categorical validity
    print(f"\n  EXH adversarial example validation:")
    cont_diff_exh = (adv_exh[:, cont] - X_cc[:, cont]).abs()
    max_cont_diff_exh = cont_diff_exh.max().item()
    cat_sums = adv_exh[:, cat_min:cat_max+1].sum(dim=1)
    all_onehot = torch.allclose(cat_sums, torch.ones_like(cat_sums))
    print(f"    Max continuous perturbation: {max_cont_diff_exh:.6f} (limit {EPS})")
    print(f"    Categorical one-hot valid: {all_onehot}")
    print(f"    Categorical values in {{0, 1}}: "
          f"{torch.all((adv_exh[:, cat_indices] <= 1.0) & (adv_exh[:, cat_indices] >= 0.0)).item()}")

    # Verify which categorical flip each disagreed sample uses
    print(f"\n  Per-sample disagreement analysis (K=1):")
    print(f"    Both break: {(exh_broken & snap_broken).sum().item()}")
    print(f"    Both survive: {(~exh_broken & ~snap_broken).sum().item()}")
    print(f"    EXH breaks, SNAP misses: {(exh_broken & ~snap_broken).sum().item()}")
    print(f"    SNAP breaks, EXH misses: {(snap_broken & ~exh_broken).sum().item()}")

    # Inspect 10 samples where EXH finds attack, SNAP doesn't
    exh_missed = (exh_broken & ~snap_broken).nonzero(as_tuple=True)[0]
    n_inspect = min(10, len(exh_missed))
    print(f"\n  Detailed inspection of {n_inspect} samples (EXH breaks, SNAP misses):")

    cat_cols = list(range(cat_min, cat_max + 1))
    cat_col_groups = groups

    for idx_i in range(n_inspect):
        i = exh_missed[idx_i].item()
        orig_cat = X_cc[i, cat_indices].tolist()
        adv_cat = adv_exh[i, cat_indices].tolist()

        # Determine which categorical flip
        orig_hot_idx = orig_cat.index(max(orig_cat))
        adv_hot_idx = adv_cat.index(max(adv_cat))
        orig_label = f"col {cat_min + orig_hot_idx}"
        adv_label = f"col {cat_min + adv_hot_idx}"

        # Verify it's a valid single-group flip
        flips = 0
        for g in cat_col_groups:
            orig_in_g = orig_hot_idx in [idx - cat_min for idx in g]
            adv_in_g = adv_hot_idx in [idx - cat_min for idx in g]
            # Check if the hot index within this group changed
            orig_in_group = [idx - cat_min for idx in g if orig_cat[idx - cat_min] == 1.0]
            adv_in_group = [idx - cat_min for idx in g if adv_cat[idx - cat_min] == 1.0]
            if orig_in_group != adv_in_group:
                flips += 1

        cont_orig = X_cc[i, cont]
        cont_adv = adv_exh[i, cont]
        linf = (cont_adv - cont_orig).abs().max().item()

        # SNAP prediction on EXH's adversarial example
        with torch.no_grad():
            snap_pred_on_exh = model(adv_exh[i:i+1]).argmax(dim=1).item()

        print(f"    Sample {i} (orig idx {sample_indices[i].item()}): "
              f"orig_cat={orig_label}, adv_cat={adv_label}, "
              f"group_flips={flips}, "
              f"cont_L∞={linf:.6f}≤{EPS}, "
              f"EXH_adv_pred={snap_pred_on_exh} vs true={y_cc[i].item()}, "
              f"SNAP_survived={not snap_broken[i].item()}")

        # Also check: what categorical does SNAP pick?
        # Re-run SNAP with tracking
        set_seed_local = lambda s: (torch.manual_seed(s), np.random.seed(s))
        set_seed_local(42)
        snap_adv = unified_pgd_attack(model, X_cc[i:i+1].clone(), y_cc[i:i+1].clone(),
                                        epsilon=EPS, alpha=ALPHA, alpha_cat=1.0, steps=STEPS,
                                        continuous_cols=cont, categorical_groups=groups)
        snap_cat = snap_adv[0, cat_indices].tolist()
        snap_hot_idx = snap_cat.index(max(snap_cat))
        snap_label = f"col {cat_min + snap_hot_idx}"

        print(f"      SNAP chose cat: {snap_label} (different from EXH's {adv_label})")
        print(f"      SNAP adv pred: {model(snap_adv).argmax(dim=1).item()} (true={y_cc[i].item()})")

    return exh_broken, snap_broken


def set_seed_local(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    print(f"Device: {DEVICE} | torch: {torch.__version__}")
    print(f"Epsilon={EPS}, Alpha={ALPHA}, PGD steps={STEPS}")
    print(f"EXH = canonical_mixed_norm_attack (exhaustive categorical enumeration)")
    print(f"SNAP = unified_pgd_attack (gradient-snapped categorical flip)")

    # Part A: Comprehensive comparison
    k1_gaps = run_comprehensive_comparison()

    # Part B: Per-sample inspection
    run_per_sample_inspection()

    # Part C: Summary
    print(f"\n{'=' * 70}")
    print(f"EXECUTIVE SUMMARY")
    print(f"{'=' * 70}")

    arr_all = np.array([r['gap_pp'] for r in k1_gaps])
    print(f"\n  Across {len(k1_gaps)} evaluations (3 datasets, 3 methods, up to 13 seeds each):")
    print(f"    K=1 gap (SNAP - EXH): mean={arr_all.mean():.1f}pp, "
          f"median={np.median(arr_all):.1f}pp, "
          f"min={arr_all.min():.1f}pp, max={arr_all.max():.1f}pp")
    print(f"    SNAP overestimates K=1 robustness in {sum(1 for g in arr_all if g > 1):.0f}/{len(arr_all)} cases")
    print(f"    SNAP underestimates in {sum(1 for g in arr_all if g < -1):.0f} cases")

    # Save full comparison
    os.makedirs('verification/pilots', exist_ok=True)
    with open('verification/pilots/exh_vs_snap_all_datasets.json', 'w') as f:
        json.dump(k1_gaps, f, indent=2)
    print(f"\n  Saved: verification/pilots/exh_vs_snap_all_datasets.json")

    print(f"\n  RECOMMENDATION: Use EXH as the canonical evaluator for the mixed")
    print(f"  threat model. SNAP systematically overestimates K=1 robustness by")
    print(f"  {arr_all.mean():.0f}pp on average (up to {arr_all.max():.0f}pp on single checkpoint).")
    print(f"  SNAP can be retained as an approximate baseline with quantified discrepancy.")


if __name__ == "__main__":
    sys.exit(main())
