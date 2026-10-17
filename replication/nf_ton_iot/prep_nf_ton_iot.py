"""
Prep NF-ToN-IoT-v2 (NetFlow variant of UNSW ToN-IoT v2) for the tabular
adversarial pipeline.

Source: /root/workspace/workspace/03-Code/Projects/Legacy/Project Archimedes/
        data/raw/ton_iot/NF-ToN-IoT-v2-test.csv
        (3,388,100 flows x 45 columns; 42 numeric flow features, binary Label,
         multiclass Attack).

Memory-safe by construction: the 531 MB CSV is parsed in 500k-row pandas
chunks with numeric usecols only; we hold one chunk at a time and only
accumulate a bounded subsample (not the full matrix), so peak RAM stays
well under 1 GB.

Transforms:
- drop IPV4_SRC_ADDR / IPV4_DST_ADDR / Attack (keep binary Label only,
  matching the binary-attack protocol of the other three datasets)
- NaN -> 0, +/-inf -> clip
- robust scaling to [0,1]: per-feature [0.1%, 99.9%] quantile clip
  (computed on the pooled subsample, honest to the subset we actually
  train/eval on), then min-max; label binaries kept as-is (Label is 0/1)

Outputs (consumed by get_train_loader / get_test_loader, which prefer
parquet and stream row-batches from it):
    data/nf_ton_iot_train.parquet  (N_TRAIN)
    data/nf_ton_iot_test.parquet   (N_TEST)

Run from ml-optimizer/:  python3 scripts/prep_nf_ton_iot.py
"""
import os
import time

import numpy as np
import pandas as pd

SRC = ("/root/workspace/workspace/03-Code/Projects/Legacy/Project Archimedes/"
       "data/raw/ton_iot/NF-ToN-IoT-v2-test.csv")
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"]
N_TRAIN = 300_000
N_TEST = 200_000
SUBSAMPLE_TOTAL = N_TRAIN + N_TEST
CHUNK = 500_000
SEED = 42

OUT_DIR = "data"


def pass1_counts():
    """First streaming pass: count rows/labels per chunk to plan stratified
    subsample rates, and peek feature column list."""
    label_cols = "Label"
    n_total = 0
    n_atk = 0
    feature_cols = None
    for chunk in pd.read_csv(SRC, usecols=[label_cols], chunksize=CHUNK):
        n_total += len(chunk)
        n_atk += int((chunk["Label"] > 0).sum())
    # determine feature columns from header
    header = pd.read_csv(SRC, nrows=1)
    feature_cols = [c for c in header.columns if c not in DROP_COLS + ["Label"]]
    return n_total, n_atk, feature_cols


def main():
    t0 = time.time()
    n_total, n_atk, feature_cols = pass1_counts()
    class_rate = min(1.0, SUBSAMPLE_TOTAL / max(n_total, 1))
    print(f"rows={n_total:,} attacks={n_atk:,} ({n_atk/n_total:.1%}) "
          f"features={len(feature_cols)} target_subsample={SUBSAMPLE_TOTAL:,}",
          flush=True)

    rng = np.random.default_rng(SEED)
    pooled = []
    # Second pass: random per-row keeps (uniform subsample keeps ratio fixed).
    # Numpy Generator with per-chunk skip ahead for determinism.
    skips = rng.integers(0, n_total, size=SUBSAMPLE_TOTAL)
    skips = np.sort(skips)
    skips_set = set(skips.tolist())

    for i, chunk in enumerate(pd.read_csv(SRC, usecols=feature_cols + ["Label"],
                                          chunksize=CHUNK, low_memory=False)):
        base = i * CHUNK
        rel = [(r - base) for r in skips if base <= r < base + CHUNK]
        if rel:
            pooled.append(chunk.iloc[rel])
    df = pd.concat(pooled, ignore_index=True).astype(np.float64)
    del pooled
    print(f"subsampled={len(df):,} peak_pass=ok {time.time()-t0:.0f}s",
          flush=True)

    labels = (df.pop("Label") > 0).astype(np.int64).to_numpy()
    vals = df.to_numpy(dtype=np.float64)  # float64: some cols exceed float32
    vals[vals > 1e300] = 1e300  # pre-clip the extreme tails before scaling
    vals[~np.isfinite(vals)] = 0.0

    lo = np.quantile(vals, 0.001, axis=0)
    hi = np.quantile(vals, 0.999, axis=0)
    scale = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    vals = np.clip((vals - lo) / scale, 0.0, 1.0).astype(np.float32)
    print("normalized to [0,1]; label dist:",
          np.bincount(labels), flush=True)

    # stratified split: shuffle, take first N_TRAIN as train rest as test,
    # re-shuffling with proportions preserved by the uniform subsample.
    idx = rng.permutation(len(labels))
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_TEST]

    tr_path = os.path.join(OUT_DIR, "nf_ton_iot_train.parquet")
    te_path = os.path.join(OUT_DIR, "nf_ton_iot_test.parquet")
    cols = list(feature_cols)
    pd.DataFrame(vals[tr], columns=cols, dtype=np.float32).assign(
        label=labels[tr]).to_parquet(tr_path, index=False)
    pd.DataFrame(vals[te], columns=cols, dtype=np.float32).assign(
        label=labels[te]).to_parquet(te_path, index=False)
    print(f"wrote {tr_path} ({len(tr):,}) and {te_path} ({len(te):,}) "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
