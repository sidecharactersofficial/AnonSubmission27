import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from app.ml.attacks.unified_pgd import unified_pgd_attack
from app.ml.attacks.eval_protocol import (
    EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
)
from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_test_loader, get_config
from app.ml.utils.checkpoint import load_model_checkpoint, compute_sha256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ['nsl-kdd', 'cicids2017', 'unsw_nb15', 'nf_ton_iot', 'nf_ton_iot_oh']
METHODS = ['hardened', 'curriculum', 'rsc', 'lite']
KNOWN_TOTALS = {"cicids2017": 623869, "unsw_nb15": 508010, "nsl_kdd": 22543,
                "nf_ton_iot": 200000, "nf_ton_iot_oh": 200000}

# Output layout mirrors results/unified/ but is a separate dir so we never
# overwrite the archival anchor JSONs.
OUT_DIR = "results/replication"


# =============================================================================
# STEP 1 — Constraint Mapping (extracted from REAL loaded data, not assumed)
# =============================================================================
def map_constraints(config, loader, max_batches=1):
    """Returns (continuous_cols, categorical_groups, valid_states_per_group).

    valid_states_per_group[g] is a (V_g, |G_g|) float tensor containing ONLY the
    one-hot states actually observed in the loaded test data. For groups where
    data is strictly one-hot (NSL-KDD, CICIDS2017, most UNSW-NB15 groups), this
    reduces to the full simplex-vertex basis eye(|G|).
    """
    x, _ = next(iter(loader))
    valid_states = []
    for group in config.CATEGORICAL_GROUPS:
        sub = x[:, group]
        uniq, _ = torch.unique(sub, dim=0, return_counts=True)
        valid_states.append(uniq.float().to(DEVICE))
    return config.CONTINUOUS_COLS, config.CATEGORICAL_GROUPS, valid_states


def check_onehot_and_bounds(config, loader, max_batches=1):
    """Report one-hot fraction + continuous bounds per group (verification only)."""
    x, _ = next(iter(loader))
    report = {'n_rows': x.size(0), 'groups': []}
    for group in config.CATEGORICAL_GROUPS:
        sub = x[:, group]
        is_onehot = ((sub.sum(dim=1) == 1.0) & ((sub == 1.0) | (sub == 0.0)).all(dim=1))
        frac = is_onehot.float().mean().item()
        report['groups'].append({
            'group': group, 'onehot_frac': frac,
            'valid_states_observed': int(torch.unique(sub, dim=0).size(0))
        })
    cont = x[:, config.CONTINUOUS_COLS]
    report['continuous'] = {
        'min': cont.min().item(), 'max': cont.max().item(),
        'within_[0,1]_frac': ((cont >= 0) & (cont <= 1)).float().mean().item()
    }
    return report


# =============================================================================
# STEP 2 — Defense layers (TabNet-emb, Gumbel-Softmax, Feature Squeezing)
# =============================================================================
class TabularMLPWithEmbedding(nn.Module):
    """TabNet-style: pushes each categorical group through nn.Embedding into a
    continuous latent space before the main MLP (Kireev et al. 2023, arXiv:2306.04064).
    Wraps a base TabularMLP checkpoint. Dimension-preserving form: each group's
    |G| slots are replaced by the rows of the embedding matrix Q_G (V_G x |G|),
    so the wrapped model consumes exactly the same input width as the base —
    checkpoint loads with strict=True. Q is initialized to the identity so the
    wrapper starts EXACTLY equivalent to the base model (no fabricated gains);
    attack pressure then moves the effective rows the way the paper describes."""

    def __init__(self, base, categorical_groups):
        super().__init__()
        self.base = base
        self.categorical_groups = categorical_groups
        self.embeddings = nn.ModuleList()
        for g in categorical_groups:
            e = nn.Embedding(len(g), len(g))
            e.weight.data.copy_(torch.eye(len(g)))   # identity init
            self.embeddings.append(e)
        self.register_buffer('_cat_idx', torch.tensor([i for group in categorical_groups for i in group]))

    def forward(self, x):
        xs = x.clone()
        for e, group in zip(self.embeddings, self.categorical_groups):
            idx = x[:, group].argmax(dim=1)   # one-hot -> token
            xs[:, group] = e(idx)             # replace slots with embedding rows
        return self.base(xs)


class GumbelMLP(nn.Module):
    """Replaces the hard one-hot with F.gumbel_softmax in the forward pass —
    produces a smooth relaxation of the categorical subspace that the L∞
    gradient actually sees. Dimension-preserving: the relaxed sample is
    spirited through the same |G| slots, so the frozen base weights still see
    their expected layout. The wrapper is stochastic (Gumbel noise), so each
    forward pass is a fresh relaxation.

    NOTE: because inputs arrive as hard one-hot vectors at eval time (real data
    is one-hot by constraint map), the relaxed vector is gumbel_softmax(onehot)
    which differs from the one-hot the base MLP was trained on — that mismatch
    is the 'fake smooth landscape' effect; we measure it honestly rather than
    hide it."""

    def __init__(self, base, categorical_groups, tau=1.0):
        super().__init__()
        self.base = base
        self.categorical_groups = categorical_groups
        self.tau = tau

    def forward(self, x):
        xs = x.clone()
        for group in self.categorical_groups:
            logits = x[:, group]
            xs[:, group] = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        return self.base(xs)


class FeatureSqueezing(nn.Module):
    """Feature Squeezing (Xu et al. 2018, NDSS): median filter + rounding wrapper.
    Rounds continuous features to k decimal_places and applies a 1D median filter
    of width=3 right before the model input."""

    def __init__(self, base, continuous_cols, bits=4, window=3):
        super().__init__()
        self.base = base
        self.continuous_cols = continuous_cols
        self.levels = 2 ** bits
        self.window = window

    def forward(self, x):
        xs = x.clone()
        cont = x[:, self.continuous_cols]
        cont_q = torch.round(cont * (self.levels - 1)) / (self.levels - 1)
        if self.window >= 3 and cont_q.size(1) >= self.window:
            pad = torch.nn.functional.pad(cont_q, (1, 1), mode='replicate')
            cont_med = torch.median(
                torch.stack([pad[:, :-2], pad[:, 1:-1], pad[:, 2:]], dim=-1),
                dim=-1
            ).values
            xs[:, self.continuous_cols] = cont_med
        else:
            xs[:, self.continuous_cols] = cont_q
        return self.base(xs)


# =============================================================================
# STEP 3 — Their NATIVE training loop / protocol (only used for --train flag)
# =============================================================================
def native_training_pass(model, loader, config, epochs, method='hardened'):
    """Their exact loss function + standard L∞-only PGD adversarial training
    (alpha_cat effectively 0 — no categorical perturbation in the attack),
    matching train_unified.py hyperparameters: Adam lr=1e-3, eps=0.15,
    alpha_cont=0.01, pgd_steps=10 (training-time), 50/50 clean/adv loss."""
    from torch.optim import Adam
    opt = Adam(model.parameters(), lr=1e-3)
    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        ce, n, acc = 0.0, 0, 0
        for data, target in loader:
            data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)
            out_clean = model(data)
            loss_clean = F.cross_entropy(out_clean, target)

            model.eval()
            adv = unified_pgd_attack(model, data, target, epsilon=EVAL_EPSILON,
                                     alpha=EVAL_ALPHA_CONT, alpha_cat=0.0,
                                     steps=10,
                                     continuous_cols=config.CONTINUOUS_COLS,
                                     categorical_groups=[])
            model.train()
            out_adv = model(adv)
            loss_adv = F.cross_entropy(out_adv, target)

            loss = 0.5 * loss_clean + 0.5 * loss_adv
            opt.zero_grad(); loss.backward(); opt.step()

            ce += loss.item() * data.size(0); n += data.size(0)
            acc += (out_clean.argmax(1) == target).sum().item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={ce/n:.4f} clean train acc={acc/n:.4f}")


# =============================================================================
# STEP 4 (Illusion) — L∞-ONLY PGD eval, their standard protocol
# =============================================================================
def linf_only_pgd_eval(model, loader, config, epsilon, alpha, steps):
    """Their standard evaluation: PGD on continuous cols only, no categorical flips.
    This is the 'false high' robust number when categorical AEs are ignored."""
    model.eval()
    clean_correct = 0
    robust = 0
    total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            clean_correct += (pred == target).sum().item()
            mask = (pred == target)
        data_sub, target_sub = data[mask], target[mask]
        if data_sub.size(0) == 0:
            total += data.size(0)
            continue
        total += data.size(0)
        adv = unified_pgd_attack(model, data_sub, target_sub,
                                 epsilon=epsilon, alpha=alpha, alpha_cat=0.0,
                                 steps=steps,
                                 continuous_cols=config.CONTINUOUS_COLS,
                                 categorical_groups=[])
        with torch.no_grad():
            out = model(adv)
            robust += (out.argmax(dim=1) == target_sub).sum().item()
    return clean_correct, robust, total


# =============================================================================
# STEP 5 (Takedown) — Exhaustive mixed-norm K=1 evaluator
# =============================================================================
def exhaustive_k1_eval(model, loader, config, valid_states,
                       epsilon, alpha, steps, batch_inner=512, device=DEVICE):
    """For each clean-correct sample:
       1. K=0: continuous PGD only (survivor gate).
       2. Every categorical group G, every valid one-hot state V of G: set the
          group's columns to that state, run continuous PGD-40, keep worst loss.
       A sample survives K=1 only if it survives K=0 AND all groups × all states.
    """
    model.eval()
    clean_correct = 0
    total = 0
    survive_k0 = 0
    survive_k1 = 0
    for data, target in loader:
        data, target = data.to(device), target.long().view(-1).to(device)
        with torch.no_grad():
            pred = model(data).argmax(dim=1)
            mask = (pred == target)
            clean_correct += mask.sum().item()
            total += data.size(0)
        data, target = data[mask], target[mask]
        if data.size(0) == 0:
            continue

        # --- K=0 base case
        adv = unified_pgd_attack(model, data, target, epsilon=epsilon, alpha=alpha,
                                 alpha_cat=0.0, steps=steps,
                                 continuous_cols=config.CONTINUOUS_COLS,
                                 categorical_groups=[])
        with torch.no_grad():
            surviving = (model(adv).argmax(dim=1) == target)
        survive_k0 += surviving.sum().item()

        data, target = data[surviving], target[surviving]
        if data.size(0) == 0:
            continue

        still_alive = torch.ones(data.size(0), dtype=torch.bool, device=device)
        for group, states in zip(config.CATEGORICAL_GROUPS, valid_states):
            n_states = states.size(0)
            k = data.size(0)
            # Broadcast each sample against every valid state of this group.
            # x_rep: (k * n_states, D)
            data_rep = data.repeat_interleave(n_states, dim=0)
            target_rep = target.repeat_interleave(n_states, dim=0)
            state_rep = states.repeat(k, 1)                  # (k*n_states, |G|)
            data_rep[:, group] = state_rep                    # hard-coded one-hot state

            adv = unified_pgd_attack(model, data_rep, target_rep,
                                     epsilon=epsilon, alpha=alpha, alpha_cat=0.0,
                                     steps=steps,
                                     continuous_cols=config.CONTINUOUS_COLS,
                                     categorical_groups=[])
            with torch.no_grad():
                pred_rep = model(adv).argmax(dim=1)
            broken = (pred_rep != target_rep).view(k, n_states)  # (k, n_states)
            any_success = broken.any(dim=1)  # attacker wins if ANY state breaks it
            still_alive &= ~any_success

            data = data[still_alive]   # prune: continue only on robust samples
            target = target[still_alive]
            if data.size(0) == 0:
                break
            still_alive2 = torch.ones(data.size(0), dtype=torch.bool, device=device)
            still_alive = still_alive2  # placeholder; real logic below
        # Recompute a clean still-alive set for this batch
        # (re-run cheap logic — see note; final count handled in aggregate below)
        survive_k1 += data.size(0)
    return clean_correct, survive_k0, survive_k1, total


# NOTE: the `still_alive` bookkeeping above is subtle; replace with an exact
# bitmask implementation below. We keep exactness by re-checking per-batch with
# a boolean tensor that never gets masked away by the pruning path.

def exhaustive_k1_eval_exact(model, loader, config, valid_states,
                             epsilon, alpha, steps, device=DEVICE):
    model.eval()
    clean_correct = 0
    total = 0
    survive_k0 = 0
    survive_k1 = 0

    for data, target in loader:
        data, target = data.to(device), target.long().view(-1).to(device)
        with torch.no_grad():
            pred = model(data).argmax(dim=1)
            mask = (pred == target)
        clean_correct += mask.sum().item()
        total += data.size(0)

        data, target = data[mask], target[mask]
        if data.size(0) == 0:
            continue

        # K=0
        adv = unified_pgd_attack(model, data, target, epsilon=epsilon, alpha=alpha,
                                 alpha_cat=0.0, steps=steps,
                                 continuous_cols=config.CONTINUOUS_COLS,
                                 categorical_groups=[])
        with torch.no_grad():
            alive = (model(adv).argmax(dim=1) == target)  # (k,)
        survive_k0 += alive.sum().item()

        data, target = data[alive], target[alive]
        if data.size(0) == 0:
            continue

        # Exhaustively test every (group, state) pair — exact,
        # because we never prune inside the (group,state) loop; we only mutate
        # the categorical subset. Any single success kills the sample.
        k = data.size(0)
        alive = torch.ones(k, dtype=torch.bool, device=device)
        alive_target = target.clone()
        alive_data = data.clone()
        for group, states in zip(config.CATEGORICAL_GROUPS, valid_states):
            if alive.sum() == 0:
                break
            idx_alive = alive.nonzero(as_tuple=True)[0]
            sub_data = alive_data[idx_alive]
            sub_target = alive_target[idx_alive]
            n_states = states.size(0)
            data_rep = sub_data.repeat_interleave(n_states, dim=0)
            target_rep = sub_target.repeat_interleave(n_states, dim=0)
            data_rep[:, group] = states.repeat(idx_alive.size(0), 1)
            adv = unified_pgd_attack(model, data_rep, target_rep,
                                     epsilon=epsilon, alpha=alpha, alpha_cat=0.0,
                                     steps=steps,
                                     continuous_cols=config.CONTINUOUS_COLS,
                                     categorical_groups=[])
            with torch.no_grad():
                pred_rep = model(adv).argmax(dim=1)
            success = (pred_rep != target_rep).view(-1, n_states).any(dim=1)  # attack succeeds when prediction is WRONG
            # alive[idx_alive[i]] = not success[i]
            flip = idx_alive[success]
            alive[flip] = False
        survive_k1 += alive.sum().item()
    return clean_correct, survive_k0, survive_k1, total


# =============================================================================
# Provenance trail
# =============================================================================
def sha256_of(path):
    return compute_sha256(path)


DEFENSE_WRAPPERS = {
    'tabnet_embedding': lambda base, config: TabularMLPWithEmbedding(
        base, config.CATEGORICAL_GROUPS),
    'gumbel_softmax': lambda base, config: GumbelMLP(
        base, config.CATEGORICAL_GROUPS, tau=1.0),
    'feature_squeezing': lambda base, config: FeatureSqueezing(
        base, config.CONTINUOUS_COLS, bits=4, window=3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--samples', type=int, default=None,
                    help='limit test samples for a quick verification pass')
    ap.add_argument('--batch-size', type=int, default=2000)
    ap.add_argument('--steps', type=int, default=EVAL_PGD_STEPS)
    ap.add_argument('--epsilon', type=float, default=EVAL_EPSILON)
    ap.add_argument('--alpha', type=float, default=EVAL_ALPHA_CONT)
    ap.add_argument('--with-defense-layers', action='store_true',
                    help='also eval wrappers: TabNet-embedding / Gumbel / Squeeze')
    ap.add_argument('--skip-takedown', action='store_true')
    ap.add_argument('--skip-illusion', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing eval JSON at the target path')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)

    constraint_report_path = os.path.join(OUT_DIR, "constraint_map.json")
    if not os.path.exists(constraint_report_path):
        constraint_report = {}
    else:
        with open(constraint_report_path) as f:
            constraint_report = json.load(f)

    for dataset in args.datasets:
        if dataset not in DATASETS:
            raise ValueError(dataset)
        config = get_config(dataset)
        safe_name = dataset.replace('-', '_')
        expected_n = KNOWN_TOTALS[safe_name]

        loader = get_test_loader(dataset, batch_size=args.batch_size)
        if args.samples is not None:
            x_all, y_all = [], []
            for xb, yb in loader:
                x_all.append(xb); y_all.append(yb)
                if sum(t.size(0) for t in x_all) >= args.samples:
                    break
            x = torch.cat(x_all)[:args.samples]
            y = torch.cat(y_all)[:args.samples]
            loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=False)

        c_cols, c_groups, valid_states_list = map_constraints(config, loader)
        crep = check_onehot_and_bounds(config, loader)
        constraint_report[dataset] = crep
        with open(constraint_report_path, 'w') as f:
            json.dump(constraint_report, f, indent=1)
        print(f"[{dataset}] constraint map: "
              f"{[g['onehot_frac'] for g in crep['groups']]} one-hot fracs, "
              f"cont bounds {crep['continuous']}")

        eval_kwargs = dict(epsilon=args.epsilon, alpha=args.alpha, steps=args.steps)

        for method in args.methods:
            seeds = args.seeds if args.seeds is not None else [None]
            for seed in seeds:
                suffix = f"_seed{seed}" if seed is not None else ""
                ckpt_path = f"models/unified/model_adv_{method}_{safe_name}{suffix}.pth"
                if not os.path.exists(ckpt_path):
                    print(f"[skip] {ckpt_path} not found")
                    continue
                sha = sha256_of(ckpt_path)
                model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state)
                model.eval()

                res_path = os.path.join(OUT_DIR, f"eval_{method}_{safe_name}{suffix}.json")
                base_result_exists = os.path.exists(res_path)
                if base_result_exists and not args.force \
                        and not (args.with_defense_layers and args.skip_illusion
                                 and args.skip_takedown):
                    print(f"[skip] results already exist: {res_path}")
                    continue
                if args.skip_illusion and args.skip_takedown and base_result_exists:
                    with open(res_path) as f:
                        result = json.load(f)
                else:
                    result = {}

                t0 = time.time()
                result = {
                    'checkpoint': ckpt_path,
                    'checkpoint_sha256': sha,
                    'epsilon': args.epsilon,
                    'alpha_cont': args.alpha,
                    'pgd_steps': args.steps,
                    'protocol': 'linf-only-pgd + exhaustive-k1 (ADR-001)',
                    'dataset': dataset
                }

                if not args.skip_illusion:
                    cc, rb, tot = linf_only_pgd_eval(model, loader, config, **eval_kwargs)
                    result['linf_only_pgd'] = {
                        'clean_correct': cc, 'robust_correct': rb, 'total': tot,
                        'clean_acc': cc / tot * 100.0,
                        'robust_acc': rb / tot * 100.0,
                    }
                    print(f"[{dataset}/{method}{suffix}] ILLUSION  L∞-only-PGD robust={rb/tot*100:.2f}% (n={tot})")

                if not args.skip_takedown:
                    cc, sk0, sk1, tot = exhaustive_k1_eval_exact(
                        model, loader, config, valid_states_list, **eval_kwargs)
                    result['exhaustive_k1'] = {
                        'clean_correct': cc, 'survive_k0': sk0, 'survive_k1': sk1, 'total': tot,
                        'clean_acc_pct_of_full': cc / tot * 100.0,
                        'k0_robust_acc_pct_of_clean_correct': sk0 / max(cc, 1) * 100.0,
                        'k1_robust_acc_pct_of_clean_correct': sk1 / max(cc, 1) * 100.0,
                        'k1_robust_acc_pct_of_full': sk1 / tot * 100.0,
                    }
                    print(f"[{dataset}/{method}{suffix}] TAKEDOWN  K1 exhaustive={sk1}/{cc} "
                          f"(clean-correct-based {sk1/max(cc,1)*100:.2f}%) in {time.time()-t0:.1f}s")

                if not (args.skip_illusion and args.skip_takedown):
                    with open(res_path, 'w') as f:
                        json.dump(result, f, indent=1)
                    print(f"[saved] {res_path}")

                # --- Defense-layer wrappers: same exhaustive K=1 takedown ---
                if args.with_defense_layers:
                    for wrap_name, build in DEFENSE_WRAPPERS.items():
                        wrap_res = os.path.join(
                            OUT_DIR, f"eval_{method}_{safe_name}{suffix}_{wrap_name}.json")
                        if os.path.exists(wrap_res):
                            continue
                        torch.manual_seed(42)   # reset RNG for reproducible Gumbel runs
                        wrapped = build(model, config).to(DEVICE)
                        wrapped.eval()
                        if wrap_name == 'feature_squeezing':
                            # Squeezing must be inside the attack's forward pass to
                            # preserve gradient semantics; wrapped model already does
                            # that, so unified_pgd_attack sees the wrapper correctly.
                            pass
                        w_cc, w_sk0, w_sk1, w_tot = exhaustive_k1_eval_exact(
                            wrapped, loader, config, valid_states_list, **eval_kwargs)
                        w_result = {
                            'checkpoint': ckpt_path,
                            'checkpoint_sha256': sha,
                            'wrapper': wrap_name,
                            'epsilon': args.epsilon, 'alpha_cont': args.alpha,
                            'pgd_steps': args.steps,
                            'protocol': 'exhaustive-k1 (ADR-001) through defense wrapper',
                            'dataset': dataset,
                            'exhaustive_k1': {
                                'clean_correct': w_cc, 'survive_k0': w_sk0,
                                'survive_k1': w_sk1, 'total': w_tot,
                                'clean_acc_pct_of_full': w_cc / w_tot * 100.0,
                                'k0_robust_acc_pct_of_clean_correct': w_sk0 / max(w_cc, 1) * 100.0,
                                'k1_robust_acc_pct_of_clean_correct': w_sk1 / max(w_cc, 1) * 100.0,
                                'k1_robust_acc_pct_of_full': w_sk1 / w_tot * 100.0,
                            },
                        }
                        with open(wrap_res, 'w') as f:
                            json.dump(w_result, f, indent=1)
                        print(f"[{dataset}/{method}{suffix}][{wrap_name}] TAKEDOWN K1="
                              f"{w_sk1}/{w_cc} ({w_sk1/max(w_cc,1)*100:.2f}%)")


if __name__ == '__main__':
    main()
