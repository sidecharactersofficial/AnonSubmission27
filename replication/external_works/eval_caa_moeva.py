"""
SOTA cross-check (Steps 1-3): pull the OFFICIAL implementations
(serval-uni-lu/tabularbench), map OUR exact mixed-norm constraints into their
API, and run CAA (cAPGD + MoEvA2 ensemble) and MoEvA2 standalone against our
existing hardened checkpoints.

Memory-streamed driver: NEVER materializes full-set tensors or full attack
populations. Instead:
  - dataset is consumed loader-batch by loader-batch
  - clean-correct samples are filtered per batch (nothing big is concatenated)
  - MoEvA2 population output of each small chunk is reduced to a scalar count
    immediately (best-of-population + success counting), then freed
  - only running tallies survive across chunks

Official code is unmodified EXCEPT two memory/import-compat patches, with
attack math untouched:
  1. numpy-2.x dtype-compat aliases (required just to import).
  2. objective_calculator.py: Parallel(n_jobs=-1) -> n_jobs=1. The joblib
     default spawns one LokyProcess worker PER CPU core (20 here), and every
     worker re-imports torch (~375 MB) -> 7.5 GB of workers alone = the
     recurring OOM killer. Sequential evaluation is mathematically identical.

Run from ml-optimizer/:
    PYTHONPATH=.:/tmp/tabularbench python scripts/eval_caa_moeva.py \
        --datasets nsl-kdd --methods hardened --samples 100 --attacks moeva
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch


def trim_malloc():
    """Return freed memory to the OS. pymoo/torch churn leaves arenas with
    many small allocations whose pages glibc never releases by default; RSS
    then creeps up across hundreds of chunks even though live data is small
    (~70 MB/chunk observed). malloc_trim makes the creep plateau."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def rss_mb():
    """Current resident set size in MB (Linux)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0

sys.path.insert(0, "/tmp/tabularbench")

from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_test_loader, get_config

DEVICE = torch.device(os.environ.get("BLACKICE_DEVICE", "cpu"))
DATASETS = ['nsl-kdd', 'cicids2017', 'unsw_nb15', 'nf_ton_iot', 'nf_ton_iot_oh']
METHODS = ['hardened', 'curriculum', 'rsc', 'lite']
KNOWN_TOTALS = {"cicids2017": 623869, "unsw_nb15": 508010, "nsl_kdd": 22543,
                "nf_ton_iot": 200000, "nf_ton_iot_oh": 200000}

OUT_DIR = "results/replication/sota"

# Chunk size fed to the attack per call. Small enough that the population
# tensor from MoEvA2 (chunk x ~230 pop x D float64) stays well under ~2 GB
# even for the widest dataset (D=80), independent of total test-set size.
# Override with --chunk; the attacks process one sample at a time internally,
# so small chunks cost no extra compute.
ATTACK_CHUNK = 25
TORCH_THREADS = 2


# =============================================================================
# Constraint mapping (Step 2)
# =============================================================================

def build_constraints_tabularbench(config):
    """Translate our strict one-hot groups into tabularbench's grammar.

    - feature_types: 'real' for continuous cols, 'cat' for one-hot slots
      (MoEvA2 uses this mask to pick discrete-aware crossover operators).
    - mutable: all True (attacker may touch every feature within budget).
    - lower/upper bounds: [0,1] everywhere (matches our MIN_VAL/MAX_VAL).
    - relation constraints: for each group G, EqualConstraint(sum(G) == 1.0)
      with a numerical tolerance — exactly the strict one-hot rule.
    """
    from tabularbench.constraints.constraints import Constraints
    from tabularbench.constraints.relation_constraint import (
        EqualConstraint, Feature, Constant, ManySum,
    )

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
    # CAA wraps all relation constraints in AndConstraint, which requires
    # >=2 operands. Datasets with zero/one one-hot group (NF-ToN-IoT has
    # none; CICIDS2017 has one) would crash there, so pad with neutral
    # feature-referencing tautologies (Feature(0) == Feature(0)). A pure
    # constant tautology (0 == 0) has no Feature operands and pymoo's
    # executor then returns a scalar, breaking MoEvA2's objective stacking
    # ("array at index 2 has size 1") — the feature reference keeps every
    # objective per-row. Vacuously true; changes no semantics.
    feat0 = Feature(0)
    while len(rel_cons) < 2:
        rel_cons.append(EqualConstraint(feat0, feat0, Constant(1e-9)))
    feat0 = Feature(0)
    while len(rel_cons) < 2:
        rel_cons.append(EqualConstraint(feat0, feat0, Constant(1e-9)))
    return Constraints(feature_types, mutable, lower, upper, rel_cons,
                       feature_names=None)


# =============================================================================
# Model adapter
# =============================================================================

class TorchModelWrapper:
    """Expose our TabularMLP through the numpy predict_proba interface the
    attacks and objective calculator use. Weights unchanged. Also accepts
    torch tensors (CPU or otherwise) and routes them to the model's device —
    required because CAA's internal paths mix numpy, CPU tensors and the
    model's CUDA residency."""

    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = next(model.parameters()).device
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
        """ObjectiveCalculator (CAA eval path) invokes the classifier directly;
        route those calls through predict_proba as tabularbench's own Model
        subclasses do (the calculator feeds raw logits, not probabilities)."""
        return self.predict_proba(x)


class IdentityScaler:
    """Our data is already in [0,1] / one-hot — the attack operates on it
    directly (this is the honest no-scaler configuration). All returns are
    plain torch Tensors on CPU (numpy converts implicitly via __array__;
    tocuh tensors pass through); CAPGD's internal device-mixing paths handle
    CPU tensors correctly."""

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


# =============================================================================
# Attack runners (official implementations, called on small chunks only)
# =============================================================================

def make_caa(model, config, eps=0.15, steps=10, n_gen=100, n_offsprings=100, seed=0):
    from tabularbench.attacks.caa.caa import ConstrainedAutoAttack
    constraints = build_constraints_tabularbench(config)
    wrapper = TorchModelWrapper(model)
    # ConstrainedMultiAttack.check_validity() expects every attack to reference
    # the same `wrapper_model`, but MoEvA2 gets the bare torch model — our
    # wrapper object is a plain adapter without that attribute. Provide a
    # benign attribute so the identity check passes without changing attack math.
    wrapper.wrapper_model = model
    return ConstrainedAutoAttack(
        constraints=constraints,
        constraints_eval=build_constraints_tabularbench(config),
        scaler=IdentityScaler(),
        model=model, model_objective=wrapper,
        n_jobs=1,
        eps=eps, norm="Linf",
        version="standard",
        n_classes=2, seed=seed,
        steps=steps, n_gen=n_gen, n_offsprings=n_offsprings,
    )


def make_moeva(model, config, eps=0.15, n_gen=100, n_offsprings=100, seed=0):
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
    """Reduce (n, pop, D) population output to per-sample best-of-population
    adversarial candidates. Processes the pop axis in sub-chunks of 32 to
    bound peak memory; returns (n, D) array."""
    x_pop = np.asarray(x_pop)
    if x_pop.ndim == 2:
        return x_pop
    n, pop, d = x_pop.shape
    y_rep_chunk = np.repeat(y_chunk, 1)  # one label per sample
    out = None
    best_p = None
    sub = 32
    for s0 in range(0, pop, sub):
        s1 = min(s0 + sub, pop)
        block = x_pop[:, s0:s1, :].reshape(-1, d)     # (n*blk, D)
        probs = wrapper.predict_proba(block)           # (n*blk, C)
        # label for each flattened row: sample i owns rows [i*blk, (i+1)*blk)
        row_labels = np.tile(y_chunk[:, None], (1, s1 - s0)).reshape(-1)
        flat_true = probs[np.arange(len(row_labels)), row_labels]
        true_p = flat_true.reshape(n, s1 - s0)         # (n, blk)
        cand = x_pop[:, s0:s1, :][np.arange(n)[:, None],
                                  true_p.argmin(axis=1)[:, None],
                                  np.arange(d)[None, :]]  # hmm keep simple
        cand_idx = true_p.argmin(axis=1)
        cand = x_pop[np.arange(n)[:, None],
                     (np.arange(s0, s1)[None, :] * np.ones((n, 1), dtype=int)),
                     :][np.arange(n), cand_idx]
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
# Evaluation driver — streamed by construction
# =============================================================================

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Tally:
    """Streaming counters; only these survive across loader batches."""
    def __init__(self):
        self.total = 0
        self.clean_correct = 0
        self.n_attacked = 0
        self.robust = 0
        self.pop_max = 0


def evaluate_dataset_attacks(dataset, method, atk, args, config):
    """Stream the loader: per batch -> filter clean-correct -> attack in
    ATTACK_CHUNK-sized pieces -> count + free. Returns Tally + wall time."""
    safe_name = dataset.replace('-', '_')
    ckpt_path = f"models/unified/model_adv_{method}_{safe_name}.pth"
    if not os.path.exists(ckpt_path):
        return None
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    wrapper = TorchModelWrapper(model)

    attack_obj = None
    moerva_wrapper = None
    if atk == 'caa':
        attack_obj = make_caa(model, config, eps=args.eps, steps=args.steps,
                              n_gen=args.n_gen)
    else:
        attack_obj, moerva_wrapper = make_moeva(model, config, eps=args.eps,
                                                n_gen=args.n_gen)
    probe = TorchModelWrapper(model)

    tally = Tally()
    t_start = time.time()

    loader = get_test_loader(dataset,
                             batch_size=500 if args.samples is None else min(500, args.samples))
    done = 0
    for xb, yb in loader:
        y_np = yb.long().view(-1).numpy().astype(np.int64)
        x_np = xb.numpy().astype(np.float32)
        pred = probe.predict(x_np)
        mask = pred == y_np
        tally.total += len(y_np)
        tally.clean_correct += int(mask.sum())
        if not mask.any():
            done += len(y_np)
            del x_np, xb, pred, mask, y_np
            gc.collect()
            continue
        keep = np.nonzero(mask)[0]
        x_sub, y_sub = x_np[keep], y_np[keep]
        del x_np, xb, pred, mask, keep
        gc.collect()

        for c0 in range(0, len(y_sub), args.chunk):
            c1 = min(c0 + args.chunk, len(y_sub))
            xc, yc = x_sub[c0:c1].copy(), y_sub[c0:c1]
            adv = attack_obj(xc, yc)  # (n, D) or (n, pop, D) float64/float32
            if isinstance(adv, torch.Tensor):
                adv = adv.cpu().numpy()
            tally.pop_max = max(tally.pop_max,
                                adv.shape[1] if adv.ndim == 3 else 0)
            adv = reduce_population(adv, yc, probe)
            adv_pred = probe.predict(adv)
            tally.n_attacked += c1 - c0
            tally.robust += int((adv_pred == yc).sum())
            del adv, adv_pred, xc, yc
            gc.collect()
        del x_sub, y_np
        trim_malloc()
        done += len(yb)
        if args.progress_every:
            print(f"    ... {done} attacked_so_far={tally.n_attacked} "
                  f"robust={tally.robust} rss={rss_mb():.0f}MB", flush=True)
        if args.samples is not None and done >= args.samples:
            break
    tally.wall = time.time() - t_start
    return ckpt_path, sha256_of(ckpt_path), tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['nsl-kdd'])
    ap.add_argument('--methods', nargs='+', default=['hardened'])
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--steps', type=int, default=10, help='CAPGD steps')
    ap.add_argument('--n-gen', type=int, default=100, help='MoEvA2 generations')
    ap.add_argument('--eps', type=float, default=EVAL_EPSILON)
    ap.add_argument('--attacks', nargs='+', default=['caa', 'moeva'])
    ap.add_argument('--chunk', type=int, default=ATTACK_CHUNK,
                    help='samples per attack call (memory knob)')
    ap.add_argument('--progress-every', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(TORCH_THREADS)

    for dataset in args.datasets:
        if dataset not in DATASETS:
            raise ValueError(dataset)
        config = get_config(dataset)
        safe_name = dataset.replace('-', '_')
        N = KNOWN_TOTALS[safe_name]

        for method in args.methods:
            for atk in args.attacks:
                tag = (f"_n{args.samples}" if args.samples is not None else "")
                res_path = os.path.join(
                    OUT_DIR, f"sota_{atk}_{method}_{safe_name}{tag}.json")
                if os.path.exists(res_path):
                    print(f"[skip] {res_path} exists")
                    continue
                out = evaluate_dataset_attacks(dataset, method, atk, args, config)
                if out is None:
                    print(f"[skip] checkpoint missing for {method}/{safe_name}")
                    continue
                ckpt_path, sha, tally = out
                result = {
                    'attack': atk,
                    'official_impl': 'serval-uni-lu/tabularbench @ HEAD',
                    'checkpoint': ckpt_path,
                    'checkpoint_sha256': sha,
                    'dataset': dataset,
                    'epsilon': args.eps,
                    'norm': 'Linf',
                    'capgd_steps': args.steps,
                    'n_gen': args.n_gen,
                    'attacked_total': tally.n_attacked,
                    'population_max': tally.pop_max,
                    'clean_correct': tally.clean_correct,
                    'total_evaluated': tally.total,
                    'known_total': N,
                    'limited_to_samples': args.samples,
                    'robust_under_attack': tally.robust,
                    'robust_acc_pct_of_clean_correct':
                        tally.robust / max(tally.n_attacked, 1) * 100.0,
                    'robust_acc_pct_of_full': tally.robust / max(tally.total, 1) * 100.0,
                    'wall_seconds': tally.wall,
                }
                with open(res_path, 'w') as f:
                    json.dump(result, f, indent=1)
                print(f"[{dataset}/{method}][{atk}] "
                      f"robust={tally.robust}/{tally.n_attacked} "
                      f"({result['robust_acc_pct_of_clean_correct']:.2f}% of clean-correct)"
                      f" wall={tally.wall:.1f}s -> {res_path}")


if __name__ == '__main__':
    main()
