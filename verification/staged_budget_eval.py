#!/usr/bin/env python3
"""
Staged categorical budget evaluation for UNSW-NB15.

Tests the hypothesis: UNSW's higher K=1 robustness is because K=1 only
flips ONE of FIVE categorical groups, leaving 4 untouched.

Runs K=0, 1, 2 (and optionally higher) on representative UNSW-NB15 checkpoints
using the canonical EXH evaluator with exhaustive categorical enumeration.

Per-sample state counts:
  K=0: 1 state (original)
  K=1: 1 + sum(size(g)-1 for g in groups) = 1 + 36 = 37 states
  K=2: 1 + 36 + sum_{i<j} (size(gi)-1)*(size(gj)-1) = 1 + 36 + 471 = 508 states
  K=3: C(5,3) triples = 10, with product of (size-1) for each = much larger

Run from ml-optimizer/:
    python verification/staged_budget_eval.py [--kmax 2]
"""
import sys, os, json, time, argparse, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_config, get_test_loader
from app.ml.utils.checkpoint import load_model_checkpoint
from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = EVAL_EPSILON
ALPHA = EVAL_ALPHA_CONT
STEPS = EVAL_PGD_STEPS


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enumerate_k_states(orig_cat, rel_groups, K):
    """
    Enumerate all categorical states reachable by flipping at most K groups.

    Each group must be in a valid one-hot state (exactly one 1).

    For K=0: just the original state (1 state)
    For K=1: original + all single-group alternative states
    For K=2: original + all single-group + all two-group combination states
    ...

    Returns a list of tensors, each of shape (cat_width,).
    """
    states = [orig_cat.clone()]  # K=0: original state

    if K < 1:
        return states

    # For each group, enumerate its alternative states (flip to another value)
    group_alts = []  # group_alts[i] = list of alternative one-hot states for group i
    for rel_group in rel_groups:
        alts = []
        for i in range(len(rel_group)):
            alt = torch.zeros(len(rel_group), device=orig_cat.device)
            alt[i] = 1.0
            alts.append(alt)
        group_alts.append(alts)

    # K=1: enumerate all single-group flips
    k1_states = []
    for gi in range(len(rel_groups)):
        for alt in group_alts[gi]:
            new_state = orig_cat.clone()
            new_state[rel_groups[gi]] = alt
            k1_states.append(new_state)
    states.extend(k1_states)

    if K < 2:
        return states

    # K=2: enumerate all pairs of group flips
    k2_states = []
    for gi, gj in itertools.combinations(range(len(rel_groups)), 2):
        for alt_i in group_alts[gi]:
            for alt_j in group_alts[gj]:
                new_state = orig_cat.clone()
                new_state[rel_groups[gi]] = alt_i
                new_state[rel_groups[gj]] = alt_j
                k2_states.append(new_state)
    states.extend(k2_states)

    if K < 3:
        return states

    # K=3: enumerate all triples of group flips
    k3_states = []
    for gi, gj, gk in itertools.combinations(range(len(rel_groups)), 3):
        for alt_i in group_alts[gi]:
            for alt_j in group_alts[gj]:
                for alt_k in group_alts[gk]:
                    new_state = orig_cat.clone()
                    new_state[rel_groups[gi]] = alt_i
                    new_state[rel_groups[gj]] = alt_j
                    new_state[rel_groups[gk]] = alt_k
                    k3_states.append(new_state)
    states.extend(k3_states)

    if K < 4:
        return states

    # K=4: enumerate all quadruples
    k4_states = []
    for gi, gj, gk, gl in itertools.combinations(range(len(rel_groups)), 4):
        for alt_i in group_alts[gi]:
            for alt_j in group_alts[gj]:
                for alt_k in group_alts[gk]:
                    for alt_l in group_alts[gl]:
                        new_state = orig_cat.clone()
                        new_state[rel_groups[gi]] = alt_i
                        new_state[rel_groups[gj]] = alt_j
                        new_state[rel_groups[gk]] = alt_k
                        new_state[rel_groups[gl]] = alt_l
                        k4_states.append(new_state)
    states.extend(k4_states)

    if K < 5:
        return states

    # K=5: enumerate all quintuples (all groups)
    k5_states = []
    for alt_combo in itertools.product(*[range(len(g)) for g in group_alts]):
        new_state = orig_cat.clone()
        for gi in range(len(rel_groups)):
            new_state[rel_groups[gi]] = group_alts[gi][alt_combo[gi]]
        k5_states.append(new_state)
    states.extend(k5_states)

    return states


def canonical_mixed_norm_attack_k(model, images, labels, config, epsilon, alpha, steps, K=0):
    """
    Canonical EXH attack with arbitrary K budget.

    Enumerates all categorical states where at most K groups are flipped from
    the original, runs independent PGD on continuous features for each state,
    and returns the adversarial example with maximum loss.

    Categorical assignments are frozen (no PGD on categorical logits).
    States are generated PER-SAMPLE (matching the original canonical evaluator).
    """
    batch_size = images.shape[0]
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    if not config.CATEGORICAL_GROUPS:
        adv = images.clone().detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            out = model(adv)
            loss = loss_fn(out, labels)
            model.zero_grad()
            loss.mean().backward()
            with torch.no_grad():
                grad = adv.grad
                adv_cont = adv[:, config.CONTINUOUS_COLS] + alpha * grad[:, config.CONTINUOUS_COLS].sign()
                eta = (adv_cont - images[:, config.CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
                adv.data[:, config.CONTINUOUS_COLS] = (images[:, config.CONTINUOUS_COLS] + eta).clamp(0.0, 1.0)
            adv = adv.detach()
        return adv

    cat_indices = [idx for group in config.CATEGORICAL_GROUPS for idx in group]
    cat_min = min(cat_indices)
    cat_max = max(cat_indices)
    cat_slice = slice(cat_min, cat_max + 1)
    rel_groups = [[idx - cat_min for idx in group] for group in config.CATEGORICAL_GROUPS]
    cont_cols = config.CONTINUOUS_COLS

    orig_cat = images[:, cat_slice].clone()  # (batch_size, cat_width)

    # Generate per-sample state tensors
    states_per_sample = [orig_cat.clone()]  # K=0: original

    if K >= 1:
        for rel_group in rel_groups:
            for i in range(len(rel_group)):
                new_state = orig_cat.clone()
                new_state[:, rel_group] = 0.0
                new_state[:, rel_group[i]] = 1.0
                states_per_sample.append(new_state)

    if K >= 2:
        for gi, gj in itertools.combinations(range(len(rel_groups)), 2):
            for i in range(len(rel_groups[gi])):
                for j in range(len(rel_groups[gj])):
                    new_state = orig_cat.clone()
                    new_state[:, rel_groups[gi]] = 0.0
                    new_state[:, rel_groups[gi][i]] = 1.0
                    new_state[:, rel_groups[gj]] = 0.0
                    new_state[:, rel_groups[gj][j]] = 1.0
                    states_per_sample.append(new_state)

    if K >= 3:
        for gi, gj, gk in itertools.combinations(range(len(rel_groups)), 3):
            for i in range(len(rel_groups[gi])):
                for j in range(len(rel_groups[gj])):
                    for k in range(len(rel_groups[gk])):
                        new_state = orig_cat.clone()
                        new_state[:, rel_groups[gi]] = 0.0
                        new_state[:, rel_groups[gi][i]] = 1.0
                        new_state[:, rel_groups[gj]] = 0.0
                        new_state[:, rel_groups[gj][j]] = 1.0
                        new_state[:, rel_groups[gk]] = 0.0
                        new_state[:, rel_groups[gk][k]] = 1.0
                        states_per_sample.append(new_state)

    all_states = torch.stack(states_per_sample, dim=0)  # (num_states, batch_size, cat_width)
    num_states = all_states.shape[0]

    expanded = images.unsqueeze(0).expand(num_states, -1, -1).clone()
    expanded[:, :, cat_slice] = all_states
    expanded = expanded.reshape(num_states * batch_size, -1).detach()
    expanded_labels = labels.unsqueeze(0).expand(num_states, -1).reshape(-1)

    orig_expanded_cont = expanded[:, cont_cols] if cont_cols else None
    chunk_size = max(1, 16000 // num_states)

    final_adv = []
    final_losses = []

    for start in range(0, expanded.shape[0], chunk_size):
        end = min(start + chunk_size, expanded.shape[0])
        adv_chunk = expanded[start:end].clone().detach()
        labels_chunk = expanded_labels[start:end]
        orig_cont_chunk = orig_expanded_cont[start:end]

        for _ in range(steps):
            adv_chunk.requires_grad_(True)
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
            model.zero_grad()
            loss.sum().backward()
            with torch.no_grad():
                grad = adv_chunk.grad
                adv_cont = adv_chunk[:, cont_cols] + alpha * grad[:, cont_cols].sign()
                eta = (adv_cont - orig_cont_chunk).clamp(-epsilon, epsilon)
                adv_chunk.data[:, cont_cols] = (orig_cont_chunk + eta).clamp(0.0, 1.0)
            adv_chunk = adv_chunk.detach()

        with torch.no_grad():
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
        final_adv.append(adv_chunk)
        final_losses.append(loss)

    adv = torch.cat(final_adv, dim=0)
    losses = torch.cat(final_losses, dim=0)

    losses = losses.view(num_states, batch_size)
    adv = adv.view(num_states, batch_size, -1)

    best_loss, best_idx = losses.max(dim=0)
    best_adv = adv[best_idx, torch.arange(batch_size)]

    return best_adv


def evaluate_k(model, loader, config, K):
    """Evaluate model at budget K, return counts."""
    total = 0
    clean_correct = 0
    robust_correct = 0
    state_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        with torch.no_grad():
            clean_pred = model(batch_x).argmax(dim=1)
        cc_mask = clean_pred == batch_y
        total += len(batch_y)
        clean_correct += cc_mask.sum().item()

        if cc_mask.sum() == 0:
            continue

        X_cc = batch_x[cc_mask]
        y_cc = batch_y[cc_mask]

        if state_count == 0 and K <= 2:
            # Estimate state count from first batch
            cat_indices = [idx for g in config.CATEGORICAL_GROUPS for idx in g]
            cat_min, cat_max = min(cat_indices), max(cat_indices)
            rel_groups = [[idx - cat_min for idx in g] for g in config.CATEGORICAL_GROUPS]
            orig_cat = torch.zeros(cat_max - cat_min + 1, device=DEVICE)
            orig_cat[0] = 1.0  # Fake original state (just for counting)
            states = enumerate_k_states(orig_cat, rel_groups, K)
            state_count = len(states)
            print(f"  K={K}: {state_count} categorical states per sample")

        set_seed(42)
        adv = canonical_mixed_norm_attack_k(model, X_cc, y_cc, config, EPS, ALPHA, STEPS, K=K)
        with torch.no_grad():
            adv_pred = model(adv).argmax(dim=1)
        robust_correct += (adv_pred == y_cc).sum().item()

    return {
        'n_total': total,
        'clean_correct': clean_correct,
        'robust_correct': robust_correct,
        'k1_survivors_pct': robust_correct / clean_correct * 100 if clean_correct > 0 else 0,
        'state_count': state_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmax', type=int, default=2, choices=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()

    print(f"Device: {DEVICE} | torch: {torch.__version__}")
    print(f"Epsilon={EPS}, Alpha={ALPHA}, PGD steps={STEPS}")
    print(f"Max K budget: {args.kmax}")

    config = get_config('unsw_nb15')
    print(f"\nUNSW-NB15 config:")
    print(f"  Categorical groups: {len(config.CATEGORICAL_GROUPS)} groups")
    print(f"  Group sizes: {[len(g) for g in config.CATEGORICAL_GROUPS]}")
    print(f"  Continuous cols: {len(config.CONTINUOUS_COLS)}")
    print(f"  Feature dim: {config.FEATURE_DIM}")

    # Compute state counts
    print(f"\nCategorical state space per sample:")
    print(f"\nCategorical state space per sample (UNSW-NB15, 5 groups [11,11,13,2,4]):")
    group_sizes = [len(g) for g in config.CATEGORICAL_GROUPS]
    for K in range(args.kmax + 1):
        if K == 0:
            total_states = 1
        elif K == 1:
            total_states = 1 + sum(s - 1 for s in group_sizes)
        elif K == 2:
            k1_alts = sum(s - 1 for s in group_sizes)
            k2_alts = sum((group_sizes[i]-1)*(group_sizes[j]-1)
                         for i, j in itertools.combinations(range(len(group_sizes)), 2))
            total_states = 1 + k1_alts + k2_alts
        elif K == 3:
            k1_alts = sum(s - 1 for s in group_sizes)
            k2_alts = sum((group_sizes[i]-1)*(group_sizes[j]-1)
                         for i, j in itertools.combinations(range(len(group_sizes)), 2))
            k3_alts = sum((group_sizes[i]-1)*(group_sizes[j]-1)*(group_sizes[kk]-1)
                         for i, j, kk in itertools.combinations(range(len(group_sizes)), 3))
            total_states = 1 + k1_alts + k2_alts + k3_alts
        elif K == 4:
            k1_alts = sum(s - 1 for s in group_sizes)
            k2_alts = sum((group_sizes[i]-1)*(group_sizes[j]-1)
                         for i, j in itertools.combinations(range(len(group_sizes)), 2))
            k3_alts = sum((group_sizes[i]-1)*(group_sizes[j]-1)*(group_sizes[kk]-1)
                         for i, j, kk in itertools.combinations(range(len(group_sizes)), 3))
            k4_alts = sum((group_sizes[i]-1)*(group_sizes[j]-1)*(group_sizes[kk]-1)*(group_sizes[ll]-1)
                         for i, j, kk, ll in itertools.combinations(range(len(group_sizes)), 4))
            total_states = 1 + k1_alts + k2_alts + k3_alts + k4_alts
        elif K == 5:
            total_states = 1 + sum(s-1 for s in group_sizes)  # K<=1
            total_states += sum((group_sizes[i]-1)*(group_sizes[j]-1)
                               for i, j in itertools.combinations(range(len(group_sizes)), 2))  # K=2
            total_states += sum((group_sizes[i]-1)*(group_sizes[j]-1)*(group_sizes[kk]-1)
                               for i, j, kk in itertools.combinations(range(len(group_sizes)), 3))  # K=3
            total_states += sum((group_sizes[i]-1)*(group_sizes[j]-1)*(group_sizes[kk]-1)*(group_sizes[ll]-1)
                               for i, j, kk, ll in itertools.combinations(range(len(group_sizes)), 4))  # K=4
            total_states += int(np.prod([s - 1 for s in group_sizes]))  # K=5
        print(f"  K={K}: {total_states} states")

    # Use same checkpoint as the canonical K=1 evaluation
    ckpt_path = "models/unified/model_adv_rsc_unsw_nb15_seed42.pth"
    ckpt_sha = hashlib_sha256(ckpt_path) if os.path.exists(ckpt_path) else "N/A"

    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    load_model_checkpoint(model, ckpt_path, device=DEVICE)
    model.eval()
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"SHA-256: {ckpt_sha}")

    # Same batch as canonical evaluation
    set_seed(42)
    loader = get_test_loader('unsw_nb15', batch_size=2000)

    results = {}
    print(f"\n{'=' * 60}")
    print(f"Staged Budget Evaluation (UNSW-NB15, RSC, seed42)")
    print(f"Epsilon={EPS}, Alpha={ALPHA}, PGD steps={STEPS}")
    print(f"{'=' * 60}")

    for K in range(args.kmax + 1):
        print(f"\n--- K={K} ---")
        t0 = time.time()
        r = evaluate_k(model, loader, config, K)
        dt = time.time() - t0
        print(f"  Total: {r['n_total']}, Clean-correct: {r['clean_correct']}, "
              f"Robust at K={K}: {r['robust_correct']}")
        print(f"  K={K} robustness: {r['robust_correct']/r['clean_correct']*100:.2f}% of clean-correct")
        print(f"  Time: {dt:.1f}s")
        if K > 0:
            prev_k1 = r['robust_correct'] / r['clean_correct'] * 100
            prev = results.get(K-1, {})
            prev_pct = prev.get('k1_survivors_pct', 100)
            print(f"  Collapse from K={K-1}: {prev_pct:.2f}% → {prev_k1:.2f}% (drop: {prev_pct - prev_k1:.2f}pp)")
        results[K] = r

    # Check monotonicity
    print(f"\n{'=' * 60}")
    print(f"Monotonicity check (robustness should be non-increasing in K):")
    prev = 100.0
    for K in range(args.kmax + 1):
        pct = results[K]['robust_correct'] / results[K]['clean_correct'] * 100
        ok = pct <= prev + 0.01  # Allow tiny floating point errors
        print(f"  K={K}: {pct:.2f}% {'✓' if ok else '✗ VIOLATION'}")
        prev = pct

    # Save results
    os.makedirs("verification/staged", exist_ok=True)
    save = {
        'dataset': 'unsw_nb15',
        'config': 'rsc',
        'seed': 42,
        'checkpoint_sha256': ckpt_sha,
        'epsilon': EPS,
        'alpha': ALPHA,
        'pgd_steps': STEPS,
        'results': {str(K): {
            'n_total': results[K]['n_total'],
            'clean_correct': results[K]['clean_correct'],
            'robust_correct': results[K]['robust_correct'],
            'k1_survivors_pct': results[K]['k1_survivors_pct'],
            'state_count': results[K]['state_count'],
        } for K in range(args.kmax + 1)},
    }
    with open("verification/staged/unsw_nb15_rsc_seed42_k0_to_k5.json", 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: verification/staged/unsw_nb15_rsc_seed42_k0_to_k5.json")

    return 0

import hashlib
def hashlib_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    sys.exit(main())
