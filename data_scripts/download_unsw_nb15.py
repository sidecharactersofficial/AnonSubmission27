#!/usr/bin/env python3
"""Download and preprocess UNSW-NB15 dataset.

This script downloads the UNSW-NB15 CSV and feature files, applies the
documented preprocessing pipeline, and writes Parquet files compatible with
app/ml/data/loader.py.

Dataset source: UNSW Canberra Cyber. Public sources:
  - Official: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/research/unsw-nb15-dataset/
  - UCI ML Repo: https://archive.ics.uci.edu/ml/datasets/UNSW-NB15

If automatic download fails, place UNSW_NB15_4.csv and NUSW-NB15_features.csv
in ./data/ and re-run this script.
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

FEATURES_URLS = [
    "https://raw.githubusercontent.com/defcom17/UNSW_NB15/master/NUSW-NB15_features.csv",
    "https://raw.githubusercontent.com/ronak99/UNSW-NB15/main/NUSW-NB15_features.csv",
    "https://raw.githubusercontent.com/liufrei01/UNSW_NB15_Data_Analysis/main/NUSW-NB15_features.csv",
]

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

    # Load feature definitions if available
    feat_path = os.path.join(DATA_DIR, "NUSW-NB15_features.csv")
    download_file(FEATURES_URLS, feat_path)

    # Drop row-identity columns as documented
    drop_cols = ['srcip', 'dstip', 'srcport', 'dstport']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Drop attack_cat if present (not needed for binary robustness evaluation)
    if 'attack_cat' in df.columns:
        df = df.drop(columns=['attack_cat'])

    # Re-code label to binary
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(np.int64)
    else:
        raise ValueError("Could not find 'label' column in UNSW-NB15 data")

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'label' in cat_cols:
        cat_cols.remove('label')
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
        print(f"One-hot encoded {len(cat_cols)} categorical columns -> {df.shape[1]} features")

    # Standardize continuous features (fit on train split only)
    from sklearn.model_selection import train_test_split
    y = df['label'].values.astype(np.int64)
    x = df.drop(columns=['label'])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )

    cont_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    means = x_train[cont_cols].mean(axis=0)
    stds = x_train[cont_cols].std(axis=0)
    stds[stds == 0] = 1.0
    x_train[cont_cols] = (x_train[cont_cols] - means) / stds
    x_test[cont_cols] = (x_test[cont_cols] - means) / stds

    train_df = x_train.copy()
    train_df['label'] = y_train
    test_df = x_test.copy()
    test_df['label'] = y_test

    cols = [c for c in train_df.columns if c != 'label'] + ['label']
    train_df = train_df[cols]
    test_df = test_df[cols]

    train_path = os.path.join(DATA_DIR, "unsw_nb15_train.parquet")
    test_path = os.path.join(DATA_DIR, "unsw_nb15_test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Saved train -> {train_path} ({train_df.shape})")
    print(f"Saved test  -> {test_path} ({test_df.shape})")

    for p in [train_path, test_path]:
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"SHA-256 {os.path.basename(p)}: {h}")

if __name__ == "__main__":
    load_and_preprocess()
