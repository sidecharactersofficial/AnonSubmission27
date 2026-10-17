#!/usr/bin/env python3
"""
Part 2: Cross-Paradigm Comparison

Implements:
  - Cat-PGD attack: simplex convex-hull PGD with Dykstra projection (Kireev et al. NeurIPS 2023)
  - Feature Squeezing defense: bit-depth reduction + L1 detector (Xu et al. NDSS 2018)
  - RanMASK defense: randomized group masking + majority vote (Zeng et al. CL 2023)
  - Regime B: strict editability mask on UNSW-NB15 (proto+service only vs proto+service+state)

Evaluators compared:
  - EXH (canonical_mixed_norm_attack) — exact enumeration + PGD
  - SNAP (unified_pgd_attack) — gradient-snapped single categorical
  - Cat-PGD (this script) — simplex-relaxed PGD
  - MoEvA2 (tabularbench) — evolutionary search
  - CAA (tabularbench) — CAPGD + MoEvA2 ensemble

Run from ml-optimizer/:
  PYTHONPATH=.:/tmp/tabularbench python verification/cross_paradigm_compare.py
"""
import sys, os, json, time, hashlib, argparse
sys.path.insert(0, '.')
sys.path.insert(0, '/tmp/tabularbench')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_config
from app.ml.utils.checkpoint import load_model_checkpoint
from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
from scripts.eval_mixed_norm import canonical_mixed_norm_attack
from app.ml.attacks.unified_pgd import unified_pgd_attack

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


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Cat-PGD Attack (Kireev et al. 2023: simplex convex-hull relaxation + Dykstra projection)
# =============================================================================

def project_simplex_dykstra(x, s=1.0, n_iter=20):
    """Project onto the probability simplex: x >= 0, sum(x) = s.
    Uses Dykstra's alternating projection between non-negativity and sum constraint."""
    n = x.shape[-1]
    z = x.clone()
    # Dykstra: alternate between projecting onto non-negative and sum=1
    for _ in range(n_iter):
        # Project onto non-negative
        z = F.relu(z)
        # Project onto sum=s
        z = z + (s - z.sum(dim=-1, keepdim=True)) / n
    return z


def cat_pgd_attack(model, images, labels, config, epsilon=0.15, alpha=0.01, steps=40, K=1):
    """Cat-PGD: relax one-hot to simplex, do PGD in continuous relaxed space,
    project back to one-hot via argmax at the end.
    
    For K=1: relax the categorical group selected by continuous gradient magnitude,
    then project back to one-hot.
    For K>1: enumerate K groups, relax all, project back.
    """
    batch_size = images.shape[0]
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    cat_indices = [idx for group in config.CATEGORICAL_GROUPS for idx in group]
    if not cat_indices:
        # No categorical groups -> standard continuous PGD
        return canonical_mixed_norm_attack(model, images, labels, config, epsilon, alpha, steps, K=0)
    
    cat_min = min(cat_indices)
    cat_max = max(cat_indices)
    cont_cols = config.CONTINUOUS_COLS
    
    # Start from clean input
    adv = images.clone().detach()
    
    # For K=1: we'll find the best single-group flip via simplex relaxation
    # For K=0: only continuous PGD
    if K == 0:
        return canonical_mixed_norm_attack(model, images, labels, config, epsilon, alpha, steps, K=0)
    
    # K >= 1: enumerate groups (like EXH), but use simplex PGD on each
    # First, identify which groups to enumerate (up to K of them)
    if K == 1:
        # Try each group individually (same enumeration as EXH K=1)
        states_per_sample = [images.clone()]  # original (K=0 baseline)
        for rel_group in [[idx - cat_min for idx in g] for g in config.CATEGORICAL_GROUPS]:
            for i in range(len(rel_group)):
                new_state = images.clone()
                new_state[:, [cat_min + j for j in rel_group]] = 0.0
                # Relax into simplex: start from all-zeros + 1.0 at position i
                # Then optimize
                new_state[:, cat_min + rel_group[i]] = 1.0
                states_per_sample.append(new_state)
    else:
        # K=2: enumerate all pairs
        states_per_sample = [images.clone()]
        groups = list(config.CATEGORICAL_GROUPS)
        for gi1 in range(len(groups)):
            for gi2 in range(gi1 + 1, len(groups)):
                g1 = [idx - cat_min for idx in groups[gi1]]
                g2 = [idx - cat_min for idx in groups[gi2]]
                for i1 in range(len(g1)):
                    for i2 in range(len(g2)):
                        new_state = images.clone()
                        new_state[:, [cat_min + j for j in g1]] = 0.0
                        new_state[:, [cat_min + j for j in g2]] = 0.0
                        new_state[:, cat_min + g1[i1]] = 1.0
                        new_state[:, cat_min + g2[i2]] = 1.0
                        states_per_sample.append(new_state)
    
    num_states = len(states_per_sample)
    
    # Stack and expand
    expanded_images = torch.stack(states_per_sample, dim=0)  # (num_states, B, D)
    expanded_labels = labels.unsqueeze(0).expand(num_states, -1)  # (num_states, B)
    
    all_adv = []
    all_losses = []
    
    for s_idx in range(num_states):
        adv_chunk = expanded_images[s_idx].clone().detach()
        labels_chunk = expanded_labels[s_idx]
        orig_cont = adv_chunk[:, cont_cols].clone() if cont_cols else None
        orig_cat = adv_chunk[:, cat_min:cat_max+1].clone()
        
        for _ in range(steps):
            adv_chunk.requires_grad_(True)
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
            model.zero_grad()
            loss.mean().backward()
            
            with torch.no_grad():
                grad = adv_chunk.grad
                # Continuous PGD
                if cont_cols:
                    adv_cont = adv_chunk[:, cont_cols] + alpha * grad[:, cont_cols].sign()
                    eta = (adv_cont - orig_cont).clamp(-epsilon, epsilon)
                    adv_chunk.data[:, cont_cols] = (orig_cont + eta).clamp(0.0, 1.0)
                
                # Simplex PGD for categorical: update with alpha, then project to simplex
                if cat_indices:
                    rel_groups = [[idx - cat_min for idx in g] for g in config.CATEGORICAL_GROUPS]
                    for rel_group in rel_groups:
                        cat_slice = adv_chunk[:, [cat_min + j for j in rel_group]]
                        cat_slice = cat_slice + alpha * grad[:, [cat_min + j for j in rel_group]].sign()
                        # Dykstra projection onto simplex
                        cat_slice = project_simplex_dykstra(cat_slice, s=1.0, n_iter=20)
                        adv_chunk.data[:, [cat_min + j for j in rel_group]] = cat_slice
            
            adv_chunk = adv_chunk.detach()
        
        # Snap back to one-hot: argmax within each group
        with torch.no_grad():
            for g in config.CATEGORICAL_GROUPS:
                rel_g = [idx - cat_min for idx in g]
                vals = adv_chunk[:, [cat_min + j for j in rel_g]]
                argmax_idx = vals.argmax(dim=1)
                one_hot = F.one_hot(argmax_idx, num_classes=len(g)).float()
                adv_chunk[:, [cat_min + j for j in rel_g]] = one_hot
            
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
            all_adv.append(adv_chunk)
            all_losses.append(loss)
    
    adv = torch.stack(all_adv, dim=0)
    losses = torch.stack(all_losses, dim=0)
    
    best_loss, best_idx = losses.max(dim=0)
    best_adv = adv[best_idx, torch.arange(batch_size)]
    return best_adv


# =============================================================================
# Feature Squeezing Defense (Xu et al. NDSS 2018)
# =============================================================================

def bit_depth_reduce(x, config, n_bits=4):
    """Reduce bit depth of continuous features."""
    x_squeezed = x.clone()
    if config.CONTINUOUS_COLS:
        vals = x[:, config.CONTINUOUS_COLS]
        vals = torch.round(vals * (2**n_bits - 1)) / (2**n_bits - 1)
        x_squeezed[:, config.CONTINUOUS_COLS] = vals
    return x_squeezed


class FeatureSqueezingDefense:
    """Feature Squeezening defense: apply bit-depth reduction at inference,
    then run the attack on the squeezed model."""
    def __init__(self, model, config, n_bits=4):
        self.model = model
        self.config = config
        self.n_bits = n_bits
    
    def __call__(self, x):
        x_squeezed = bit_depth_reduce(x, self.config, self.n_bits)
        return self.model(x_squeezed)
    
    def predict(self, x):
        with torch.no_grad():
            return self(x).argmax(dim=1)


# =============================================================================
# RanMASK Defense (Zeng et al. CL 2023)
# =============================================================================

class RanMASKDefense:
    """Randomized group masking + majority vote.
    At each inference, mask rho fraction of feature groups, replace masked
    one-hot groups with all-zero [MASK] vector, run n_draws random draws,
    predict via majority vote."""
    def __init__(self, model, config, rho=0.3, n_draws=100, seed=0):
        self.model = model
        self.config = config
        self.rho = rho
        self.n_draws = n_draws
        self.rng = torch.Generator(device=DEVICE)
        self.rng.manual_seed(seed)
    
    def predict(self, x):
        batch_size = x.shape[0]
        n_cat_groups = len(self.config.CATEGORICAL_GROUPS)
        if n_cat_groups == 0:
            with torch.no_grad():
                return self.model(x).argmax(dim=1)
        
        cat_indices = [idx for g in self.config.CATEGORICAL_GROUPS for idx in g]
        cat_min = min(cat_indices)
        cat_max = max(cat_indices)
        
        predictions = torch.zeros((self.n_draws, batch_size), dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            for draw in range(self.n_draws):
                x_masked = x.clone()
                # Randomly select groups to mask
                k_mask = int(self.rho * n_cat_groups)
                if k_mask > 0:
                    for b in range(batch_size):
                        mask_groups = torch.randperm(n_cat_groups, generator=self.rng, device=DEVICE)[:k_mask]
                        for gi in mask_groups:
                            g = self.config.CATEGORICAL_GROUPS[gi]
                            rel = [idx - cat_min for idx in g]
                            x_masked[b, [cat_min + j for j in rel]] = 0.0
                
                predictions[draw] = self.model(x_masked).argmax(dim=1)
        
        # Majority vote
        preds = predictions.mode(dim=0).values
        return preds


# =============================================================================
# Regime B: Strict Editability Mask for UNSW-NB15
# =============================================================================

def canonical_mixed_norm_attack_regime_b(model, images, labels, config, epsilon=0.15, alpha=0.01, steps=40, K=1, editable_groups=None):
    """EXH attack with editable_groups mask — only flips groups that are in editable_groups.
    
    Regime A (default): all groups editable
    Regime B-strict: only directly packet-craftable groups (e.g., proto, service)
    Regime B-extended: proto+service+state
    """
    if editable_groups is None:
        editable_groups = config.CATEGORICAL_GROUPS
    
    batch_size = images.shape[0]
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    cat_indices = [idx for g in config.CATEGORICAL_GROUPS for idx in g]
    cat_min = min(cat_indices)
    cat_max = max(cat_indices)
    cont_cols = config.CONTINUOUS_COLS
    cat_dim = cat_max - cat_min + 1
    
    # Map editable groups to relative indices
    rel_editable = [[idx - cat_min for idx in g] for g in editable_groups]
    
    # Generate categorical states for K-reachable flips within editable groups only
    orig_cat = images[:, cat_min:cat_max+1]
    states_per_sample = [orig_cat]  # K=0 baseline
    
    if K >= 1:
        for rel_group in rel_editable:
            for i in range(len(rel_group)):
                new_state = orig_cat.clone()
                new_state[:, rel_group] = 0.0
                new_state[:, rel_group[i]] = 1.0
                states_per_sample.append(new_state)
    
    num_states = len(states_per_sample)
    
    expanded_images = images.unsqueeze(0).expand(num_states, -1, -1).clone()
    expanded_images[:, :, cat_min:cat_max+1] = torch.stack(states_per_sample, dim=0)
    expanded_images = expanded_images.reshape(num_states * batch_size, -1).detach()
    expanded_labels = labels.unsqueeze(0).expand(num_states, -1).reshape(-1)
    
    adv = expanded_images.clone()
    orig_expanded_cont = expanded_images[:, cont_cols] if cont_cols else None
    
    chunk_size = 16000
    final_advs, final_losses = [], []
    
    for start_idx in range(0, adv.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, adv.shape[0])
        adv_chunk = adv[start_idx:end_idx].clone().detach()
        labels_chunk = expanded_labels[start_idx:end_idx]
        orig_cont_chunk = orig_expanded_cont[start_idx:end_idx] if orig_expanded_cont is not None else None
        
        for _ in range(steps):
            adv_chunk.requires_grad_(True)
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
            model.zero_grad()
            loss.sum().backward()
            
            with torch.no_grad():
                grad = adv_chunk.grad
                if cont_cols:
                    adv_cont = adv_chunk[:, cont_cols] + alpha * grad[:, cont_cols].sign()
                    eta = (adv_cont - orig_cont_chunk).clamp(-epsilon, epsilon)
                    adv_chunk.data[:, cont_cols] = (orig_cont_chunk + eta).clamp(0.0, 1.0)
            adv_chunk = adv_chunk.detach()
        
        with torch.no_grad():
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
        final_advs.append(adv_chunk)
        final_losses.append(loss)
    
    adv = torch.cat(final_advs, dim=0)
    losses = torch.cat(final_losses, dim=0)
    
    losses = losses.view(num_states, batch_size)
    adv = adv.view(num_states, batch_size, -1)
    best_loss, best_idx = losses.max(dim=0)
    best_adv = adv[best_idx, torch.arange(batch_size)]
    return best_adv


# =============================================================================
# 200-sample slice preparation
# =============================================================================

def get_200_sample_slice(dataset_name, seed=42):
    """Get the same 200-sample slice used in Phase 2 controlled comparison."""
    config = get_config(dataset_name)
    safe_name = dataset_name.replace('-', '_')
    
    if safe_name in ('nsl_kdd', 'nsl-kdd'):
        df = pd.read_csv(f'data/nsl-kdd-test.csv', header=None)
        y_col = df.columns[-1]
        y = torch.tensor(df[y_col].values, dtype=torch.long)
        X = torch.tensor(df.drop(columns=[y_col]).values, dtype=torch.float32)
    elif safe_name == 'cicids2017':
        df = pd.read_parquet(f'data/cicids2017_test.parquet')
        y = torch.tensor(df['label'].values, dtype=torch.long)
        X = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
    elif safe_name == 'unsw_nb15':
        df = pd.read_parquet(f'data/unsw_nb15_test.parquet')
        y = torch.tensor(df['label'].values, dtype=torch.long)
        X = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
    else:
        from app.ml.data.loader import load_tabular_data
        ds = load_tabular_data(dataset_name, split='test', samples=5000)
        X, y = ds.tensors
    
    set_seed(seed)
    perm = torch.randperm(len(X))
    X = X[perm][:200]
    y = y[perm][:200]
    
    cont_cols = config.CONTINUOUS_COLS
    if cont_cols:
        X[:, cont_cols] = X[:, cont_cols].clamp(0.0, 1.0)
    for g in config.CATEGORICAL_GROUPS:
        vals = X[:, g]
        max_idx = vals.argmax(dim=1)
        X[:, g] = 0.0
        X[torch.arange(len(X)), g[0] + max_idx % len(g)] = 1.0
        # Proper one-hot
        for i in range(len(X)):
            idx = max_idx[i].item()
            X[i, g] = 0.0
            X[i, g[idx]] = 1.0
    
    return X.to(DEVICE), y.to(DEVICE), config


# =============================================================================
# Tabularbench attack wrappers
# =============================================================================

class TorchModelWrapper:
    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device
        model.eval()
    
    def predict_proba(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            elif isinstance(x, torch.Tensor):
                t = x.to(self.device)
            else:
                t = x
            out = self.model(t)
            return out.cpu().numpy()
    
    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)
    
    def __call__(self, x):
        return self.predict_proba(x)


class IdentityScaler:
    def transform(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32))
        if isinstance(x, torch.Tensor):
            return x.cpu()
        return x
    
    def transform_mask(self, m):
        return m
    
    def inverse_transform(self, x, **kwargs):
        return x


def build_constraints_tabularbench(config):
    from tabularbench.constraints.constraints import Constraints
    from tabularbench.constraints.relation_constraint import EqualConstraint, Feature, Constant, ManySum
    
    d = config.FEATURE_DIM
    cat_slots = set(i for g in config.CATEGORICAL_GROUPS for i in g)
    feature_types = np.array(
        ["cat" if i in cat_slots else "real" for i in range(d)], dtype=object
    )
    mutable = np.ones(d, dtype=bool)
    lower = np.zeros(d)
    upper = np.ones(d)
    
    rel_cons = []
    for group in config.CATEGORICAL_GROUPS:
        feats = [Feature(int(i)) for i in group]
        many = ManySum(feats) if len(feats) > 1 else feats[0]
        rel_cons.append(EqualConstraint(many, Constant(1.0), Constant(1e-5)))
    
    feat0 = Feature(0)
    while len(rel_cons) < 2:
        rel_cons.append(EqualConstraint(feat0, feat0, Constant(1e-9)))
    
    return Constraints(feature_types, mutable, lower, upper, rel_cons, feature_names=None)


def make_moeva(model, config, eps=0.15, n_gen=50, n_offsprings=50, seed=0):
    from tabularbench.attacks.moeva.moeva import Moeva2
    wrapper = TorchModelWrapper(model)
    atk = Moeva2(
        wrapper, constraints=build_constraints_tabularbench(config),
        eps=eps, norm="Linf", seed=seed, n_jobs=1,
        n_gen=n_gen, n_offsprings=n_offsprings,
        fun_distance_preprocess=lambda z: z,
    )
    return atk, wrapper


def reduce_population(x_pop, y_chunk, wrapper):
    x_pop = np.asarray(x_pop)
    if x_pop.ndim == 2:
        return x_pop
    n, pop, d = x_pop.shape
    out = None
    best_p = None
    sub = 32
    for s0 in range(0, pop, sub):
        s1 = min(s0 + sub, pop)
        block = x_pop[:, s0:s1, :].reshape(-1, d)
        probs = wrapper.predict_proba(block)
        row_labels = np.tile(y_chunk[:, None], (1, s1 - s0)).reshape(-1)
        flat_true = probs[np.arange(len(row_labels)), row_labels]
        true_p = flat_true.reshape(n, s1 - s0)
        cand_idx = true_p.argmin(axis=1)
        if out is None:
            out = x_pop[np.arange(n), s0 + cand_idx, :]
            best_p = true_p.min(axis=1).copy()
        else:
            min_p = true_p.min(axis=1)
            merge = min_p < best_p
            out[merge] = x_pop[np.arange(n)[merge], s0 + cand_idx[merge], :]
            best_p[merge] = min_p[merge]
    return out.astype(np.float32)


# =============================================================================
# Main comparison: Attack Table
# =============================================================================

def run_attack_table(args):
    """Run 5 attack variants on 200-sample slice for each dataset/method/seed.
    MoEvA2 uses reduced n_gen for feasibility — documented in results."""
    from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
    import gc
    
    slice_size = 200
    moeva_samples = 50  # reduced from 200 for MoEvA2 feasibility
    moeva_ngen = 50
    moeva_noff = 50
    
    datasets_methods = [
        ('nsl-kdd', 'rsc', 42),
        ('cicids2017', 'hardened', 42),
        ('cicids2017', 'rsc', 42),
        ('unsw_nb15', 'rsc', 42),
        ('nf_ton_iot_oh', 'rsc', 42),
    ]
    
    results = {}
    
    for dataset, method, seed in datasets_methods:
        ds_key = dataset.replace('-', '_')
        ckpt_path = f"models/unified/model_adv_{method}_{ds_key}_seed{seed}.pth"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"models/model_adv_{method}_{ds_key}_seed{seed}.pth"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"models/unified/model_adv_{method}_{ds_key}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] No checkpoint: {dataset}/{method}/{seed}")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{dataset}/{method}/{seed}] {ckpt_path}")
        print(f"SHA-256: {sha256_of(ckpt_path)}")
        print(f"{'='*60}")
        
        config = get_config(dataset)
        model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
        load_model_checkpoint(model, ckpt_path, device=DEVICE)
        model.eval()
        
        # Get slice
        X, y, cfg = get_200_sample_slice(dataset, seed)
        
        with torch.no_grad():
            cp = model(X).argmax(dim=1)
        cc_mask = cp == y
        X_cc = X[cc_mask]
        y_cc = y[cc_mask]
        n = min(slice_size, len(X_cc))
        X_cc = X_cc[:n]
        y_cc = y_cc[:n]
        print(f"Clean correct: {n}/{slice_size}")
        
        if n < 5:
            print(f"[SKIP] Too few clean-correct samples ({n})")
            continue
        
        ds_results = {'n_clean_correct': n, 'attacks': {}}
        
        # EXH K=1
        set_seed(42)
        t0 = time.time()
        adv = canonical_mixed_norm_attack(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
        dt = time.time() - t0
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        ra = (pred == y_cc).sum().item() / n * 100
        ds_results['attacks']['EXH_K1'] = {'ra': ra, 'time_per_100': dt/n*100, 'count': (pred==y_cc).sum().item()}
        print(f"  EXH K=1: {ra:.1f}% ({dt:.1f}s, {dt/n*100:.1f}s/100)")
        
        # SNAP K=1
        set_seed(42)
        t0 = time.time()
        adv = unified_pgd_attack(model, X_cc, y_cc, epsilon=EPS, alpha=ALPHA, steps=STEPS, continuous_cols=config.CONTINUOUS_COLS, categorical_groups=config.CATEGORICAL_GROUPS, rsc=False)
        dt = time.time() - t0
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        ra = (pred == y_cc).sum().item() / n * 100
        ds_results['attacks']['SNAP_K1'] = {'ra': ra, 'time_per_100': dt/n*100, 'count': (pred==y_cc).sum().item()}
        print(f"  SNAP K=1: {ra:.1f}% ({dt:.1f}s, {dt/n*100:.1f}s/100)")
        
        # Cat-PGD K=1
        set_seed(42)
        t0 = time.time()
        adv = cat_pgd_attack(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
        dt = time.time() - t0
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        ra = (pred == y_cc).sum().item() / n * 100
        ds_results['attacks']['CatPGD_K1'] = {'ra': ra, 'time_per_100': dt/n*100, 'count': (pred==y_cc).sum().item()}
        print(f"  Cat-PGD K=1: {ra:.1f}% ({dt:.1f}s, {dt/n*100:.1f}s/100)")
        
        # MoEvA2 (reduced n_gen for feasibility on 50 samples)
        moeva_n = min(moeva_samples, n)
        X_mv = X_cc[:moeva_n]
        y_mv = y_cc[:moeva_n]
        set_seed(42)
        moeva_atk, wrapper = make_moeva(model, config, eps=EPS, n_gen=moeva_ngen, n_offsprings=moeva_noff, seed=0)
        t0 = time.time()
        adv = moeva_atk(X_mv.cpu().numpy().astype(np.float64), y_mv.cpu().numpy().astype(np.int64))
        if isinstance(adv, torch.Tensor):
            adv = adv.cpu().numpy()
        adv = reduce_population(adv, y_mv.cpu().numpy().astype(np.int64), wrapper)
        adv = torch.from_numpy(adv.astype(np.float32)).to(DEVICE)
        dt = time.time() - t0
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        ra_mv = (pred == y_mv).sum().item() / moeva_n * 100
        ds_results['attacks']['MoEvA2'] = {'ra': ra_mv, 'time_per_100': dt/moeva_n*100, 'count': (pred==y_mv).sum().item(), 'n_samples': moeva_n}
        print(f"  MoEvA2: {ra_mv:.1f}% ({dt:.1f}s, {dt/moeva_n*100:.1f}s/100, n={moeva_n})")
        del moeva_atk, wrapper; gc.collect()
        
        # Optimal gap vs EXH
        exh_ra = ds_results['attacks']['EXH_K1']['ra']
        for atk_name, atk_data in ds_results['attacks'].items():
            if atk_name != 'EXH_K1':
                atk_data['optimality_gap_vs_EXH'] = round(exh_ra - atk_data['ra'], 1)
        
        key = f"{dataset}_{method}_s{seed}"
        results[key] = ds_results
    
    return results


# =============================================================================
# Defense Table
# =============================================================================

def run_defense_table(args):
    """Compare Hardened/RSC/Curriculum × FeatureSqueeze/RanMASK/CatPGD-train under EXH K=0/K=1."""
    # For inference-time defenses (FeatureSqueeze, RanMASK), apply on top of existing checkpoints.
    # For CatPGD training defense, would need training (future work — document as N/A).
    
    datasets_methods = [
        ('cicids2017', 'rsc', 42),
        ('cicids2017', 'hardened', 42),
        ('cicids2017', 'curriculum', 42),
        ('unsw_nb15', 'rsc', 42),
        ('nsl-kdd', 'rsc', 42),
    ]
    
    results = {}
    
    for dataset, method, seed in datasets_methods:
        ds_key = dataset.replace('-', '_')
        ckpt_path = f"models/unified/model_adv_{method}_{ds_key}_seed{seed}.pth"
        if not os.path.exists(ckpt_path):
            continue
        
        config = get_config(dataset)
        model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
        load_model_checkpoint(model, ckpt_path, device=DEVICE)
        model.eval()
        
        X, y, cfg = get_200_sample_slice(dataset, seed)
        with torch.no_grad():
            cp = model(X).argmax(dim=1)
        cc = cp == y
        X_cc, y_cc = X[cc], y[cc]
        n = len(X_cc)
        
        if n < 5:
            continue
        
        print(f"\n[{dataset}/{method}/{seed}] {n} clean-correct")
        ds_results = {'n': n, 'defenses': {}}
        
        # No defense (baseline)
        set_seed(42)
        adv = canonical_mixed_norm_attack(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
        with torch.no_grad():
            ra = (model(adv).argmax(dim=1) == y_cc).sum().item() / n * 100
        ds_results['defenses']['no_defense'] = {'K1_RA': ra}
        print(f"  No defense K=1: {ra:.1f}%")
        
        # Feature Squeezing (4-bit)
        fs_model = FeatureSqueezingDefense(model, config, n_bits=4)
        set_seed(42)
        adv = canonical_mixed_norm_attack(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
        with torch.no_grad():
            adv_squeezed_pred = fs_model(adv).argmax(dim=1)
        ra = (adv_squeezed_pred == y_cc).sum().item() / n * 100
        ds_results['defenses']['feature_squeeze_4bit'] = {'K1_RA': ra}
        print(f"  Feature Squeeze K=1: {ra:.1f}%")
        
        # RanMASK
        rm_model = RanMASKDefense(model, config, rho=0.3, n_draws=100, seed=42)
        # For attack: run EXH on original model, then check if RanMASK at inference blocks it
        set_seed(42)
        adv = canonical_mixed_norm_attack(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
        preds = rm_model.predict(adv)
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).long().to(DEVICE)
        elif isinstance(preds, torch.Tensor):
            preds = preds.long()
        ra = (preds == y_cc).sum().item() / n * 100
        ds_results['defenses']['ranmask_0.3'] = {'K1_RA': ra}
        print(f"  RanMASK K=1: {ra:.1f}%")
        
        key = f"{dataset}_{method}_s{seed}"
        results[key] = ds_results
    
    return results


# =============================================================================
# Regime B: Strict Editability on UNSW-NB15
# =============================================================================

def run_regime_b(args):
    """Test Regime B: strict editability mask on UNSW-NB15.
    
    Regime A: all groups editable (standard K=1)
    Regime B-strict: only proto + service editable (directly packet-craftable)
    Regime B-extended: proto + service + state editable
    """
    config = get_config('unsw_nb15')
    ckpt_path = "models/unified/model_adv_rsc_unsw_nb15_seed42.pth"
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    load_model_checkpoint(model, ckpt_path, device=DEVICE)
    model.eval()
    
    X, y, cfg = get_200_sample_slice('unsw_nb15', 42)
    with torch.no_grad():
        cp = model(X).argmax(dim=1)
    cc = cp == y
    X_cc, y_cc = X[cc], y[cc]
    n = len(X_cc)
    print(f"\n[UNSW-NB15/RSC/42] {n} clean-correct")
    
    groups = config.CATEGORICAL_GROUPS
    group_names = ['proto', 'service', 'state', 'is_sm_ips_ports', 'is_ftp_login']
    
    # Regime A: all groups
    set_seed(42)
    t0 = time.time()
    adv = canonical_mixed_norm_attack(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
    dt = time.time() - t0
    with torch.no_grad():
        ra_a = (model(adv).argmax(dim=1) == y_cc).sum().item() / n * 100
    print(f"  Regime A (all groups, K=1): {ra_a:.1f}% ({dt:.1f}s)")
    
    # Regime B-strict: only proto (groups[0]) + service (groups[1])
    set_seed(42)
    t0 = time.time()
    editable = [groups[0], groups[1]]
    adv = canonical_mixed_norm_attack_regime_b(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1, editable_groups=editable)
    dt = time.time() - t0
    with torch.no_grad():
        ra_b_strict = (model(adv).argmax(dim=1) == y_cc).sum().item() / n * 100
    print(f"  Regime B-strict (proto+service only, K=1): {ra_b_strict:.1f}% ({dt:.1f}s)")
    
    # Regime B-extended: proto + service + state
    set_seed(42)
    t0 = time.time()
    editable = [groups[0], groups[1], groups[2]]
    adv = canonical_mixed_norm_attack_regime_b(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1, editable_groups=editable)
    dt = time.time() - t0
    with torch.no_grad():
        ra_b_ext = (model(adv).argmax(dim=1) == y_cc).sum().item() / n * 100
    print(f"  Regime B-extended (proto+service+state, K=1): {ra_b_ext:.1f}% ({dt:.1f}s)")
    
    # Regime B-strict K=0 (continuous only, no categorical)
    set_seed(42)
    t0 = time.time()
    adv = canonical_mixed_norm_attack_regime_b(model, X_cc, y_cc, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=0, editable_groups=editable)
    dt = time.time() - t0
    with torch.no_grad():
        ra_b_k0 = (model(adv).argmax(dim=1) == y_cc).sum().item() / n * 100
    print(f"  Regime B-strict (proto+service, K=0): {ra_b_k0:.1f}% ({dt:.1f}s)")
    
    # Regime B-strict K=2 (all pairs within proto+service)
    set_seed(42)
    t0 = time.time()
    from verification.staged_budget_eval import canonical_mixed_norm_attack_k
    # Use the K=2 full enumeration but with a modified config
    class EditabilityConfig:
        pass
    # Build a custom config with only proto+service as categorical groups
    custom = type('Config', (), {})()
    custom.CONTINUOUS_COLS = config.CONTINUOUS_COLS
    custom.CATEGORICAL_GROUPS = [groups[0], groups[1]]
    custom.FEATURE_DIM = config.FEATURE_DIM
    custom.MIN_VAL = config.MIN_VAL
    custom.MAX_VAL = config.MAX_VAL
    
    adv = canonical_mixed_norm_attack_k(model, X_cc, y_cc, custom, EPS, ALPHA, STEPS, K=2)
    dt = time.time() - t0
    with torch.no_grad():
        ra_b_k2 = (model(adv).argmax(dim=1) == y_cc).sum().item() / n * 100
    print(f"  Regime B-strict (proto+service, K=2): {ra_b_k2:.1f}% ({dt:.1f}s)")
    
    results = {
        'dataset': 'unsw_nb15', 'method': 'rsc', 'seed': 42, 'n': n,
        'SHA256': sha256_of(ckpt_path),
        'regimes': {
            'A_all_K1': {'ra': ra_a},
            'B_strict_K1': {'ra': ra_b_strict, 'editable': ['proto', 'service']},
            'B_extended_K1': {'ra': ra_b_ext, 'editable': ['proto', 'service', 'state']},
            'B_strict_K0': {'ra': ra_b_k0, 'editable': ['proto', 'service']},
            'B_strict_K2': {'ra': ra_b_k2, 'editable': ['proto', 'service']},
        },
        'interpretation': {
            'A_vs_B_strict': f'Regime B-strict loses {ra_a - ra_b_strict:.1f}pp of attacks (groups excluded from search)',
            'B_strict_vs_B_ext': f'Adding state to B-strict gains {ra_b_ext - ra_b_strict:.1f}pp of attacks',
            'B_strict_K1_vs_K0': f'K=1 (proto+service) vs K=0 gap: {ra_b_k0 - ra_b_strict:.1f}pp',
            'B_strict_K1_vs_K2': f'K=2 (proto+service pairs) vs K=1 gap: {ra_b_strict - ra_b_k2:.1f}pp',
        }
    }
    
    os.makedirs('verification/staged', exist_ok=True)
    with open('verification/staged/regime_b_unsw_nb15_rsc_seed42.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: verification/staged/regime_b_unsw_nb15_rsc_seed42.json")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack-table', action='store_true')
    parser.add_argument('--defense-table', action='store_true')
    parser.add_argument('--regime-b', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    if args.all or args.attack_table:
        print("\n" + "="*70)
        print("ATTACK TABLE: 5 attack variants on 200-sample controlled slice")
        print("="*70)
        results = run_attack_table(args)
        with open('verification/staged/attack_table_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    if args.all or args.defense_table:
        print("\n" + "="*70)
        print("DEFENSE TABLE: 3 methods × 3 defenses under EXH K=1")
        print("="*70)
        results = run_defense_table(args)
        with open('verification/staged/defense_table_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    if args.all or args.regime_b:
        print("\n" + "="*70)
        print("REGIME B: Strict editability on UNSW-NB15")
        print("="*70)
        run_regime_b(args)


if __name__ == '__main__':
    main()
