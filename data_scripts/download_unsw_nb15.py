#!/usr/bin/env python3
"""Download and preprocess UNSW-NB15 dataset.

This script downloads UNSW_NB15_4.csv, applies the canonical preprocessing
pipeline matching app/ml/data/unsw_nb15_config.py, and writes Parquet files
compatible with app/ml/data/loader.py.

The canonical feature layout is 79 dimensions:
  - 38 continuous features (indices 0-37)
  - 11 proto one-hot features (indices 38-48)
  - 11 state one-hot features (indices 49-59)
  - 13 service one-hot features (indices 60-72)
  - 2 is_sm_ips_ports one-hot features (indices 73-74)
  - 4 is_ftp_login one-hot features (indices 75-78)

High-cardinality categorical columns (proto, state, service) are capped at
the top-10 most frequent values in the training set, with all remaining
values mapped to an "OTHER" category. This ensures deterministic output
dimensions matching the config.

NOTE: The shipped, hash-verified data/unsw_nb15_*.parquet files in this
artifact are the authoritative source used by all make reproduce-* targets.
This script provides a best-effort from-scratch regeneration path. Output
hashes will differ from the shipped manifests if public mirrors have changed
or if category frequencies differ slightly. For exact reproduction, use the
shipped parquet files.

Dataset source: UNSW Canberra Cyber. Public sources:
  - Official: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/research/unsw-nb15-dataset/
  - UCI ML Repo: https://archive.ics.uci.edu/ml/datasets/UNSW-NB15

If automatic download fails, place UNSW_NB15_4.csv in ./data/ and re-run.
"""
import os
import urllib.request
import pandas as pd
import numpy as np
import hashlib

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

UNSW_NB15_CSV = "UNSW_NB15_4.csv"
UNSW_NB15_URLS = [
    "https://raw.githubusercontent.com/defcom17/UNSW_NB15/master/UNSW_NB15_4.csv",
    "https://raw.githubusercontent.com/ronak99/UNSW-NB15/main/UNSW_NB15_4.csv",
    "https://raw.githubusercontent.com/liufrei01/UNSW_NB15_Data_Analysis/main/UNSW_NB15_4.csv",
]

# Raw column names from UNSW-NB15 feature definitions (1-indexed in docs,
# 0-indexed here after dropping identity columns).
# The canonical config expects these 5 categorical groups in this order.
CATEGORICAL_GROUPS_RAW = [
    "proto",      # 11 dims (top-10 + other)
    "state",      # 11 dims (top-10 + other)
    "service",    # 13 dims (top-12 + other)
    "is_sm_ips_ports",  # 2 dims (binary)
    "is_ftp_login",     # 4 dims (all unique values)
]

# Identity columns to drop (row-identifiers, not features)
DROP_COLS = ["srcip", "dstip", "srcport", "dstport", "attack_cat"]

# Top-k capping for high-cardinality categorical columns.
# These produce the exact group dimensions in unsw_nb15_config.py.
TOP_K_CAP = {
    "proto": 10,        # 10 + OTHER = 11 dims
    "state": 10,        # 10 + OTHER = 11 dims
    "service": 12,      # 12 + OTHER = 13 dims
}

def download_file(urls, dest):
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return True
    for url in urls:
        print(f"Trying {url} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Saved -> {dest}")
            return True
        except Exception as e:
            print(f"  FAILED: {e}")
    return False

def _fit_categorical_vocabs(df_train, cat_cols):
    """Compute top-k vocabularies for high-cardinality categorical columns."""
    vocabs = {}
    for col in cat_cols:
        if col in TOP_K_CAP:
            k = TOP_K_CAP[col]
            top = df_train[col].value_counts().head(k).index.tolist()
            vocabs[col] = top + ["OTHER"]
        else:
            vocabs[col] = sorted(df_train[col].dropna().unique().tolist())
    return vocabs

def _apply_one_hot(df, vocabs):
    """Apply deterministic one-hot encoding with fixed vocabularies."""
    encoded_parts = []
    for col in CATEGORICAL_GROUPS_RAW:
        vocab = vocabs[col]
        # Map values to vocab indices, unknown -> OTHER
        mapped = df[col].apply(lambda x: x if x in vocab else "OTHER")
        one_hot = pd.get_dummies(mapped, prefix=col, dummy_na=False)
        # Reorder columns to match vocab order (deterministic)
        one_hot = one_hot.reindex(columns=[f"{col}_{v}" for v in vocab], fill_value=0)
        encoded_parts.append(one_hot)
    return pd.concat(encoded_parts, axis=1)

def load_and_preprocess():
    csv_path = os.path.join(DATA_DIR, UNSW_NB15_CSV)
    ok = download_file(UNSW_NB15_URLS, csv_path)
    if not ok:
        print("\nERROR: Could not download UNSW_NB15_4.csv")
        print("Please download it manually from:")
        print("  https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/research/unsw-nb15-dataset/")
        print(f"Place it in {DATA_DIR}/ and re-run this script.")
        return

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop identity columns and attack category
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Re-code label to binary
    if "label" not in df.columns:
        raise ValueError("Could not find 'label' column in UNSW-NB15 data")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.int64)

    # Separate label
    y = df["label"].values.astype(np.int64)
    x = df.drop(columns=["label"])

    # Identify continuous columns: everything except the 5 categorical groups
    cont_cols = [c for c in x.columns if c not in CATEGORICAL_GROUPS_RAW]
    cat_cols = [c for c in CATEGORICAL_GROUPS_RAW if c in x.columns]

    if set(cat_cols) != set(CATEGORICAL_GROUPS_RAW):
        missing = set(CATEGORICAL_GROUPS_RAW) - set(cat_cols)
        print(f"WARNING: Missing expected categorical columns: {missing}")
        print(f"Available columns: {x.columns.tolist()}")

    # Train/test split 70/30, stratified, with fixed seed
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )

    # Standardize continuous features (fit on train only)
    cont_means = x_train[cont_cols].mean(axis=0)
    cont_stds = x_train[cont_cols].std(axis=0)
    cont_stds[cont_stds == 0] = 1.0
    x_train[cont_cols] = (x_train[cont_cols] - cont_means) / cont_stds
    x_test[cont_cols] = (x_test[cont_cols] - cont_means) / cont_stds

    # Fit categorical vocabs on training data only
    vocabs = _fit_categorical_vocabs(x_train, cat_cols)

    # Apply one-hot encoding with fixed vocabs
    x_train_cat = _apply_one_hot(x_train, vocabs)
    x_test_cat = _apply_one_hot(x_test, vocabs)

    # Concatenate continuous + categorical in canonical order
    x_train_final = pd.concat([x_train[cont_cols].reset_index(drop=True),
                               x_train_cat.reset_index(drop=True)], axis=1)
    x_test_final = pd.concat([x_test[cont_cols].reset_index(drop=True),
                              x_test_cat.reset_index(drop=True)], axis=1)

    # Ensure label is last column
    train_df = x_train_final.copy()
    train_df["label"] = y_train
    test_df = x_test_final.copy()
    test_df["label"] = y_test

    cols = [c for c in train_df.columns if c != "label"] + ["label"]
    train_df = train_df[cols]
    test_df = test_df[cols]

    # Verify dimensions match config
    expected_dim = 79
    actual_dim = x_train_final.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}. "
            f"Check categorical group definitions and TOP_K_CAP settings."
        )

    train_path = os.path.join(DATA_DIR, "unsw_nb15_train.parquet")
    test_path = os.path.join(DATA_DIR, "unsw_nb15_test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Saved train -> {train_path} ({train_df.shape})")
    print(f"Saved test  -> {test_path} ({test_df.shape})")

    # Print SHA-256 for manifest verification
    for p in [train_path, test_path]:
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"SHA-256 {os.path.basename(p)}: {h}")

    print("\nNOTE: The shipped data/unsw_nb15_*.parquet files are the authoritative")
    print("source for exact reproduction. This script provides a best-effort")
    print("from-scratch regeneration. Hashes will likely differ from the shipped")
    print("manifest. For exact reproduction, use the shipped parquet files.")

if __name__ == "__main__":
    load_and_preprocess()
