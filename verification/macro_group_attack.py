#!/usr/bin/env python3
"""
Macro-group attack on UNSW-NB15.

Case 2: The 5 categorical groups are causally coupled into 2 macro-groups:
  Macro-A (Transport & Session): {proto(11), state(13), is_sm_ips_ports(2)}
  Macro-B (Application & Service): {service(11), is_ftp_login(4)}

A single macro-group flip (K_macro=1) jointly enumerates all sub-group combinations
within one macro-group, removing the artificial redundancy that protects UNSW-NB15.

Run from ml-optimizer/:
    python verification/macro_group_attack.py
"""
import sys, os, time, json, hashlib
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


# UNSW-NB15 group definitions
GROUP_DEFS = {
    0: (38, 48, 'proto'),
    1: (49, 59, 'service'),
    2: (60, 72, 'state'),
    3: (73, 74, 'is_sm_ips_ports'),
    4: (75, 78, 'is_ftp_login'),
}

# Macro-group partition
MACRO_A = [0, 2, 3]  # proto, state, is_sm_ips_ports
MACRO_B = [1, 4]     # service, is_ftp_login


class MacroGroupConfig:
    """UNSW-NB15 config with 2 macro-groups instead of 5 sub-groups."""
    CONTINUOUS_COLS = list(range(38))
    # Macro-A: cols 38-74 (proto 38-48, state 60-72, is_sm_ips_ports 73-74)
    #   But these are NOT contiguous! proto(38-48), state(60-72), is_sm(73-74)
    # Macro-B: cols 49-59 (service), 75-78 (is_ftp_login)
    #   Also not contiguous!
    MACRO_GROUPS = [
        # Macro-A: proto + state + is_sm_ips_ports — non-contiguous
        list(range(38, 49)) + list(range(60, 73)) + list(range(73, 75)),
        # Macro-B: service + is_ftp_login — non-contiguous
        list(range(49, 60)) + list(range(75, 79)),
    ]
    SUB_GROUPS = {
        0: (38, 48, 11, 'proto'),
        2: (60, 72, 13, 'state'),
        3: (73, 74, 2, 'is_sm_ips_ports'),
        1: (49, 59, 11, 'service'),
        4: (75, 78, 4, 'is_ftp_login'),
    }
    FEATURE_DIM = 79
    MIN_VAL = 0.0
    MAX_VAL = 1.0
    CATEGORICAL_GROUPS = MACRO_GROUPS


def get_one_hot_states(cols, num_classes):
    """Generate all valid one-hot states for a set of columns."""
    states = []
    for i in range(num_classes):
        state = torch.zeros(len(cols), dtype=torch.float32, device=DEVICE)
        state[i] = 1.0
        states.append(state)
    return states


def macro_attack(model, images, labels, epsilon, alpha, steps):
    """
    Macro-group attack: K=1 on 2 macro-groups.

    For each macro-group, enumerate all valid joint one-hot combinations
    of its sub-groups, run PGD on continuous features for each combination,
    and select the worst-case (max loss) across all macro-group alternatives.

    This removes the artificial redundancy where ignoring a derived group
    leaves it frozen at its clean value.
    """
    batch_size = images.shape[0]
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    cont_cols = list(range(38))

    # Get original categorical values for each macro-group
    # Macro-A: cols 38-48 (proto, 11), 60-72 (state, 13), 73-74 (is_sm, 2)
    # Macro-B: cols 49-59 (service, 11), 75-78 (is_ftp, 4)
    proto_cols = list(range(38, 49))
    service_cols = list(range(49, 60))
    state_cols = list(range(60, 73))
    is_sm_cols = list(range(73, 75))
    is_ftp_cols = list(range(75, 79))

    # Build all states: original + macro-A alternatives + macro-B alternatives
    # Each state is a full (batch_size, 79) tensor with modified categorical columns
    states = []

    # Original state (K=0)
    orig = images.clone()
    states.append(orig)

    # Macro-A: 11 × 13 × 2 = 286 combinations
    proto_states = get_one_hot_states(proto_cols, 11)
    state_states = get_one_hot_states(state_cols, 13)
    is_sm_states = get_one_hot_states(is_sm_cols, 2)

    for p_state in proto_states:
        for s_state in state_states:
            for sm_state in is_sm_states:
                new_state = orig.clone()
                new_state[:, proto_cols] = p_state
                new_state[:, state_cols] = s_state
                new_state[:, is_sm_cols] = sm_state
                states.append(new_state)

    # Macro-B: 11 × 4 = 44 combinations
    service_states = get_one_hot_states(service_cols, 11)
    is_ftp_states = get_one_hot_states(is_ftp_cols, 4)

    for svc_state in service_states:
        for ftp_state in is_ftp_states:
            new_state = orig.clone()
            new_state[:, service_cols] = svc_state
            new_state[:, is_ftp_cols] = ftp_state
            states.append(new_state)

    num_states = len(states)
    print(f"  Total states: {num_states} (1 original + {len(proto_states)*len(state_states)*len(is_sm_states)-1} macro-A + {len(service_states)*len(is_ftp_states)-1} macro-B)")

    # Run PGD on each state's continuous features, select worst-case
    all_adv = []
    all_losses = []

    # Expand: stack all states for all samples
    # Each state is (batch_size, 79), stack gives (num_states, batch_size, 79)
    expanded_images = torch.stack(states, dim=0).reshape(num_states * batch_size, -1).detach()
    expanded_labels = labels.unsqueeze(0).expand(num_states, -1).reshape(-1)
    orig_cont = expanded_images[:, cont_cols]

    chunk_size = max(1, 16000)
    final_adv = []
    final_losses = []

    for start in range(0, expanded_images.shape[0], chunk_size):
        end = min(start + chunk_size, expanded_images.shape[0])
        adv_chunk = expanded_images[start:end].clone().detach()
        labels_chunk = expanded_labels[start:end]
        orig_cont_chunk = orig_cont[start:end]

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
    best_loss, best_idx = losses.max(dim=0)
    best_adv = adv[best_idx * batch_size + torch.arange(batch_size, device=adv.device)]

    return best_adv, best_idx


def main():
    print(f"Device: {DEVICE} | torch: {torch.__version__}")
    print(f"Epsilon={EPS}, Alpha={ALPHA}, PGD steps={STEPS}")

    model = TabularMLP(input_dim=79, num_classes=2).to(DEVICE)
    ckpt = "models/unified/model_adv_rsc_unsw_nb15_seed42.pth"
    sbe = __import__('verification.staged_budget_eval', fromlist=['set_seed']).set_seed if 'verification.staged_budget_eval' in sys.modules else None

    from app.ml.utils.checkpoint import load_model_checkpoint
    load_model_checkpoint(model, ckpt, device=DEVICE)
    model.eval()
    print(f"Checkpoint: {ckpt}")
    print(f"SHA-256: {hashlib.sha256(open(ckpt, 'rb').read()).hexdigest()}")

    config = get_config('unsw_nb15')

    # Load subset
    import pandas as pd
    df = pd.read_parquet('data/unsw_nb15_test.parquet')
    y_full = torch.tensor(df['label'].values, dtype=torch.long)
    X_full = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
    set_seed(42)
    perm = torch.randperm(len(X_full))
    X = X_full[perm][:200].to(DEVICE)
    y = y_full[perm][:200].to(DEVICE)

    with torch.no_grad():
        cp = model(X).argmax(dim=1)
    cc = cp == y
    X_cc = X[cc]
    y_cc = y[cc]
    n = len(y_cc)
    print(f"\nClean-correct: {n}/200")

    # K=0 (continuous-only PGD)
    set_seed(42)
    t0 = time.time()
    adv_k0 = torch.zeros_like(X_cc)
    for _ in range(STEPS):
        adv_k0.requires_grad_(True)
        out = model(adv_k0)
        loss = nn.CrossEntropyLoss(reduction='none')(out, y_cc)
        model.zero_grad()
        loss.mean().backward()
        with torch.no_grad():
            grad = adv_k0.grad
            adv_cont = adv_k0[:, list(range(38))] + ALPHA * grad[:, list(range(38))].sign()
            eta = (adv_cont - X_cc[:, list(range(38))]).clamp(-EPS, EPS)
            adv_k0.data[:, list(range(38))] = (X_cc[:, list(range(38))] + eta).clamp(0.0, 1.0)
        adv_k0 = adv_k0.detach()
    with torch.no_grad():
        pred_k0 = model(adv_k0).argmax(dim=1)
    r_k0 = (pred_k0 == y_cc).sum().item()
    print(f"K=0 (continuous-only): {r_k0}/{n} ({r_k0/n*100:.2f}%) time={time.time()-t0:.1f}s")

    # K=1 (standard EXH, 2-group enumeration)
    from scripts.eval_mixed_norm import canonical_mixed_norm_attack
    set_seed(42)
    t0 = time.time()
    adv_k1 = canonical_mixed_norm_attack(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
    dt_k1 = time.time() - t0
    with torch.no_grad():
        pred_k1 = model(adv_k1).argmax(dim=1)
    r_k1 = (pred_k1 == y_cc).sum().item()
    print(f"K=1 (standard EXH, 37 states): {r_k1}/{n} ({r_k1/n*100:.2f}%) time={dt_k1:.1f}s")

    # K=1 macro (2 macro-groups, 329 states)
    set_seed(42)
    t0 = time.time()
    adv_macro, best_idx = macro_attack(model, X_cc, y_cc, EPS, ALPHA, STEPS)
    dt_macro = time.time() - t0
    with torch.no_grad():
        pred_macro = model(adv_macro).argmax(dim=1)
    r_macro = (pred_macro == y_cc).sum().item()
    print(f"K=macro (macro-group, 329 states): {r_macro}/{n} ({r_macro/n*100:.2f}%) time={dt_macro:.1f}s")

    print(f"\nStaged robustness (UNSW-NB15 RSC seed42, {n} samples):")
    print(f"  K=0 (continuous-only): {r_k0/n*100:.2f}%")
    print(f"  K=1 (standard, 37 states): {r_k1/n*100:.2f}%  (collapse: {(r_k0-r_k1)/n*100:.2f}pp)")
    print(f"  K=macro (329 states): {r_macro/n*100:.2f}%  (additional collapse: {(r_k1-r_macro)/n*100:.2f}pp)")
    print(f"  K=0 → K=macro total: {(r_k0-r_macro)/n*100:.2f}pp")

    # Verify constraints
    cat_indices = [idx for g in config.CATEGORICAL_GROUPS for idx in g]
    cat_min, cat_max = min(cat_indices), max(cat_indices)
    cont_cols = list(range(38))
    pert = (adv_macro[:, cont_cols] - X_cc[:, cont_cols]).abs()
    max_pert = pert.max().item()

    # Check one-hot for each group
    all_valid = True
    for g in config.CATEGORICAL_GROUPS:
        sums = adv_macro[:, g].sum(dim=1)
        valid = torch.allclose(sums, torch.ones(len(sums), device=DEVICE), atol=1e-4)
        all_valid = all_valid and valid

    print(f"\n  Constraint validation (macro-group K=1):")
    print(f"  max_cont_Linf = {max_pert:.6f} (limit={EPS})")
    print(f"  one_hot_valid = {all_valid}")

    # Determine which macro-group was selected per sample
    macro_a_groups = [0, 2, 3]
    proto_cols = list(range(38, 49))
    state_cols = list(range(60, 73))
    is_sm_cols = list(range(73, 75))
    svc_cols = list(range(49, 60))
    is_ftp_cols = list(range(75, 79))

    # Count macro-A vs macro-B selections
    macro_a_count = 0
    macro_b_count = 0
    total = len(y_cc)

    # For each sample, compare original vs adversarial categorical
    adv_cat_a = adv_macro[:, proto_cols + state_cols + is_sm_cols]
    orig_cat_a = X_cc[:, proto_cols + state_cols + is_sm_cols]
    adv_cat_b = adv_macro[:, svc_cols + is_ftp_cols]
    orig_cat_b = X_cc[:, svc_cols + is_ftp_cols]

    a_flipped = ~torch.allclose(adv_cat_a, orig_cat_a, atol=1e-5, equal_nan=False)
    b_flipped = ~torch.allclose(adv_cat_b, orig_cat_b, atol=1e-5, equal_nan=False)

    macro_a_flipped = a_flipped.view(-1).any(dim=1)  # Actually we need to check each sample
    # Better: check which macro-group changed per sample
    changed_a = (adv_cat_a != orig_cat_a).any(dim=1)
    changed_b = (adv_cat_b != orig_cat_b).any(dim=1)
    changed_either = changed_a | changed_b
    both_changed = changed_a & changed_b

    print(f"\n  Macro-group selection per sample:")
    print(f"    Samples where Macro-A (proto+state+is_sm) changed: {changed_a.sum().item()}")
    print(f"    Samples where Macro-B (service+is_ftp) changed: {changed_b.sum().item()}")
    print(f"    Samples where both changed: {both_changed.sum().item()} (should be 0 for K=macro)")
    print(f"    Samples where neither changed (only cont PGD): {(~changed_either).sum().item()}")

    # Save results
    os.makedirs('verification/staged', exist_ok=True)
    save = {
        'dataset': 'unsw_nb15',
        'method': 'rsc',
        'seed': 42,
        'n_samples': n,
        'checkpoint_sha256': hashlib.sha256(open(ckpt, 'rb').read()).hexdigest(),
        'epsilon': EPS,
        'alpha': ALPHA,
        'pgd_steps': STEPS,
        'macro_groups': {
            'A': {'groups': macro_a_groups, 'cols': {'proto': proto_cols, 'state': state_cols, 'is_sm': is_sm_cols}, 'candidates': 11*13*2},
            'B': {'groups': [1, 4], 'cols': {'service': svc_cols, 'is_ftp': is_ftp_cols}, 'candidates': 11*4},
        },
        'state_counts': {'K0': 1, 'K1': 37, 'K_macro': 329},
        'results': {
            'K0': {'count': r_k0, 'pct': r_k0/n*100},
            'K1_standard': {'count': r_k1, 'pct': r_k1/n*100, 'time': dt_k1},
            'K_macro': {'count': r_macro, 'pct': r_macro/n*100, 'time': dt_macro},
        },
        'additional_collapse_K1_to_macro': (r_k1 - r_macro)/n*100,
        'total_collapse_K0_to_macro': (r_k0 - r_macro)/n*100,
        'constraints': {'max_L_inf': max_pert, 'one_hot_valid': all_valid},
    }
    with open('verification/staged/unsw_nb15_rsc_seed42_macro.json', 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: verification/staged/unsw_nb15_rsc_seed42_macro.json")

    return 0

from app.ml.utils.checkpoint import load_model_checkpoint

if __name__ == "__main__":
    sys.exit(main())
