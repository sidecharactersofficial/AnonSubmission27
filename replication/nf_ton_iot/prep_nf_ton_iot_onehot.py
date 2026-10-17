"""
One-hot re-encoding of NF-ToN-IoT-v2 for the "encoding choice vs dataset
incompetence" experiment: the NF (NetFlow) encoding flattens PROTOCOL and
L7_PROTO into single continuous columns, which removes the discrete surface
our exhaustive one-hot takedown needs. This script rebuilds the same 500k
subsample with those two fields properly one-hot encoded:

  - PROTOCOL  -> block of 9 slots  (top-8 observed values + 'other')
  - L7_PROTO  -> block of 13 slots (top-12 observed values + 'other')
  - the 39 remaining numeric flow features stay continuous in [0,1]

Deterministic layout (feature columns):
  [0..38]  continuous (quantile-clipped + min-max, same as the NF prep)
  [39..47] one-hot PROTOCOL block   (group A)
  [48..60] one-hot L7_PROTO block   (group B)

Output: data/nf_ton_iot_oh_{train,test}.parquet (+ N_TRAIN=300k, N_TEST=200k)

Run from ml-optimizer/:  python3 scripts/prep_nf_ton_iot_onehot.py
"""
import os
import time

import numpy as np
import pandas as pd

SRC = ("/root/workspace/workspace/03-Code/Projects/Legacy/Project Archimedes/"
       "data/raw/ton_iot/NF-ToN-IoT-v2-test.csv")
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"]
ONEHOT_COLS = ["PROTOCOL", "L7_PROTO"]
K = {"PROTOCOL": 8, "L7_PROTO": 12}
N_TRAIN = 300_000
N_TEST = 200_000
CHUNK = 500_000
SEED = 42

OUT_DIR = "data"
HEADER = pd.read_csv(SRC, nrows=1)
FEATURE_COLS = [c for c in HEADER.columns if c not in DROP_COLS + ["Label"]]
CONT_COLS = [c for c in FEATURE_COLS if c not in ONEHOT_COLS]
assert len(CONT_COLS) == 39, len(CONT_COLS)


def pass1_topk():
    """Stream once: row/label counts + frequency of PROTOCOL/L7_PROTO values."""
    n_total, n_atk = 0, 0
    counts = {"PROTOCOL": pd.Series(dtype=int), "L7_PROTO": pd.Series(dtype=int)}
    for chunk in pd.read_csv(SRC, usecols=["Label"] + ONEHOT_COLS,
                             chunksize=CHUNK, low_memory=False):
        n_total += len(chunk)
        n_atk += int((chunk["Label"] > 0).sum())
        for c in ONEHOT_COLS:
            counts[c] = counts[c].add(chunk[c].value_counts(), fill_value=0)
    top = {c: counts[c].nlargest(K[c]).index.tolist() for c in ONEHOT_COLS}
    return n_total, n_atk, top


def onehot(vals: pd.Series, topk):
    """Return (n, len(topk)+1) float32 array: 1 in the observed-slot column,
    last column = 'other' for values outside the top-k (incl. NaN)."""
    out = np.zeros((len(vals), len(topk) + 1), dtype=np.float32)
    idx = {v: j for j, v in enumerate(topk)}
    codes = vals.map(idx).astype("float64")
    known = codes.notna().to_numpy()
    out[np.arange(len(vals))[known], codes[known].astype(int)] = 1.0
    out[~known, -1] = 1.0
    return out


def main():
    t0 = time.time()
    n_total, n_atk, top = pass1_topk()
    print(f"rows={n_total:,} attacks={n_atk/n_total:.1%} top={top}",
          flush=True)

    rng = np.random.default_rng(SEED)
    keep = np.zeros(n_total, dtype=bool)
    keep[rng.choice(n_total, size=N_TRAIN + N_TEST, replace=False)] = True

    cont_pool, proto_pool, l7_pool, lab_pool = [], [], [], []
    for i, chunk in enumerate(pd.read_csv(
            SRC, usecols=FEATURE_COLS + ["Label"], chunksize=CHUNK,
            low_memory=False)):
        base = i * CHUNK
        loc = keep[base:base + len(chunk)]
        if not loc.any():
            continue
        sub = chunk[loc]
        cont_pool.append(sub[CONT_COLS].to_numpy(dtype=np.float64))
        proto_pool.append(onehot(sub["PROTOCOL"], top["PROTOCOL"]))
        l7_pool.append(onehot(sub["L7_PROTO"], top["L7_PROTO"]))
        lab_pool.append((sub["Label"] > 0).to_numpy(dtype=np.int64))

    vals = np.concatenate(cont_pool)
    proto = np.concatenate(proto_pool)
    l7 = np.concatenate(l7_pool)
    labels = np.concatenate(lab_pool)
    del cont_pool, proto_pool, l7_pool, lab_pool
    print(f"subsampled={len(labels):,} {time.time()-t0:.0f}s", flush=True)

    vals[vals > 1e300] = 1e300
    vals[~np.isfinite(vals)] = 0.0
    lo = np.quantile(vals, 0.001, axis=0)
    hi = np.quantile(vals, 0.999, axis=0)
    scale = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    vals = np.clip((vals - lo) / scale, 0.0, 1.0).astype(np.float32)
    print("normalized; label dist:", np.bincount(labels), flush=True)

    X = np.concatenate([vals, proto, l7], axis=1)
    idx = rng.permutation(len(labels))
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_TEST]

    cols = ([f"c{j}" for j in range(39)]
            + [f"proto{i}" for i in range(len(top["PROTOCOL"]) + 1)]
            + [f"l7{i}" for i in range(len(top["L7_PROTO"]) + 1)])
    tr_path = os.path.join(OUT_DIR, "nf_ton_iot_oh_train.parquet")
    te_path = os.path.join(OUT_DIR, "nf_ton_iot_oh_test.parquet")
    pd.DataFrame(X[tr], columns=cols).assign(
        label=labels[tr]).to_parquet(tr_path, index=False)
    pd.DataFrame(X[te], columns=cols).assign(
        label=labels[te]).to_parquet(te_path, index=False)
    print(f"wrote {tr_path} ({len(tr):,}) {te_path} ({len(te):,}) "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
