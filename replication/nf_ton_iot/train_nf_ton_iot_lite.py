"""
Quick 'lite' checkpoint for ToN-IoT-v2 sample runs (NF or one-hot re-encoded).

This is NOT one of the three canonical checkpoints (hardened/curriculum/rsc
on NSL-KDD/UNSW/CICIDS): it is a small, explicitly-labeled training run so
the official tabularbench attacks (CAA / MoEvA2) can be sample-tested on
NF-ToN-IoT-v2 without a full canonical training schedule.

Schedule (echoed to stdout and stored in the JSON that eval_caa_moeva writes,
so the light provenance is always visible):
  - 3 supervised epochs (Adam 1e-3, CrossEntropy, batch 4096)
  - 1 adversarial epoch with unified_pgd_attack on continuous cols,
    training budget: steps=10, alpha=EVAL_ALPHA_CONT, eps=EVAL_EPSILONOutput: models/unified/model_adv_lite_<dataset>.pth
(also prints clean accuracy on the test parquet).

Run from ml-optimizer/:  python3 scripts/train_nf_ton_iot_lite.py \
    [--dataset nf_ton_iot | nf_ton_iot_oh]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, ".")
from app.ml.models.architecture import TabularMLP
from app.ml.attacks.unified_pgd import unified_pgd_attack
from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT
from app.ml.data.loader import get_config

SEED = 42
BATCH = 4096
EPOCHS_SUP = 3
EPOCHS_ADV = 1
STEPS_TRAIN = 10  # train-time budget, cf. eval budget = EVAL_PGD_STEPS (40)
OUT = "models/unified/model_adv_lite_nf_ton_iot.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='nf_ton_iot',
                    choices=['nf_ton_iot', 'nf_ton_iot_oh'])
    ds_name = ap.parse_args().dataset
    cfg = get_config(ds_name)
    tr_path = f"data/{ds_name}_train.parquet"
    te_path = f"data/{ds_name}_test.parquet"
    OUT = f"models/unified/model_adv_lite_{ds_name}.pth"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    tr = pd.read_parquet(tr_path)
    te = pd.read_parquet(te_path)
    y_tr = np.asarray(tr.pop("label"), dtype=np.int64)
    y_te = np.asarray(te.pop("label"), dtype=np.int64)
    Xtr = torch.tensor(tr.to_numpy())
    Xte = torch.tensor(te.to_numpy())
    Ytr = torch.tensor(y_tr)
    Yte = torch.tensor(y_te)
    print(f"train {Xtr.shape} labels {np.bincount(y_tr)} | "
          f"test {Xte.shape} labels {np.bincount(y_te)}", flush=True)
    del tr, te

    assert Xtr.shape[1] == cfg.FEATURE_DIM, (Xtr.shape, cfg.FEATURE_DIM)
    model = TabularMLP(input_dim=cfg.FEATURE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    for ep in range(EPOCHS_SUP):
        model.train()
        tot, corr = 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = ce(out, yb)
            loss.backward()
            opt.step()
            corr += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        print(f"sup epoch {ep+1}: train_acc={corr/tot:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    for ep in range(EPOCHS_ADV):
        model.train()
        tot, corr = 0, 0
        for xb, yb in loader:
            xb_adv = unified_pgd_attack(model, xb.clone(), yb, epsilon=EVAL_EPSILON,
                                        alpha=EVAL_ALPHA_CONT, steps=STEPS_TRAIN,
                                        continuous_cols=list(cfg.CONTINUOUS_COLS),
                                        categorical_groups=list(cfg.CATEGORICAL_GROUPS))
            opt.zero_grad()
            out = model(xb_adv)
            loss = ce(out, yb)
            loss.backward()
            opt.step()
            corr += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        print(f"adv epoch {ep+1} (PGD-{STEPS_TRAIN}): adv_train_acc="
              f"{corr/tot:.4f} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(model.state_dict(), OUT)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
        clean_acc = (pred == Yte).float().mean().item()
        print(f"clean test acc: {clean_acc:.4f}", flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
