#!/usr/bin/env python3
"""Download and preprocess CICIDS2017 dataset.

This script downloads the CICIDS2017 day CSV files, applies the documented
preprocessing pipeline, and writes Parquet files compatible with
app/ml/data/loader.py.

Dataset source: Canadian Institute for Cybersecurity (CIC) CICIDS2017.
Public sources:
  - Official: https://www.unb.ca/cic/datasets/ids-2017.html
  - UCI ML Repo: https://archive.ics.uci.edu/ml/datasets/CICIDS2017

If automatic download fails, place the six day CSVs in ./data/ and re-run.
"""
import os
import urllib.request
import pandas as pd
import numpy as np
import hashlib

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# The six CICIDS2017 day files used in the artifact, per dataset_links.md.
# URLs point to common public mirrors; they may change over time.
CICIDS2017_FILES = {
    "Wednesday-WorkingHours.pcap_ISCX.csv": (
        "https://raw.githubusercontent.com/defcom17/CICIDS2017/master/Wednesday-WorkingHours.pcap_ISCX.csv"
    ),
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": (
        "https://raw.githubusercontent.com/defcom17/CICIDS2017/master/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    ),
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": (
        "https://raw.githubusercontent.com/defcom17/CICIDS2017/master/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    ),
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": (
        "https://raw.githubusercontent.com/defcom17/CICIDS2017/master/Friday-WorkingHours-Morning.pcap_ISCX.csv"
    ),
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": (
        "https://raw.githubusercontent.com/defcom17/CICIDS2017/master/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    ),
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": (
        "https://raw.githubusercontent.com/defcom17/CICIDS2017/master/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ),
}

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return True
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Saved -> {dest}")
        return True
    except Exception as e:
        print(f"FAILED to download {url}: {e}")
        return False

def load_and_preprocess():
    frames = []
    missing = []
    for fname, url in CICIDS2017_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        ok = download_file(url, path)
        if not ok:
            missing.append(fname)
            continue
        print(f"Loading {fname} ...")
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)

    if missing:
        print("\nERROR: Could not download the following files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease download them manually from:")
        print("  https://www.unb.ca/cic/datasets/ids-2017.html")
        print(f"Place them in {DATA_DIR}/ and re-run this script.")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined shape: {df.shape}")

    # Drop duplicate columns documented in dataset_links.md
    # 'Label' duplicate and 'Fwd Header Length' duplicate
    drops = [c for c in df.columns if c.lower() in ('label', 'fwd header length')]
    drops = [c for c in drops if c in df.columns]
    df = df.drop(columns=drops)
    print(f"After dropping duplicates {drops}: {df.shape}")

    # Drop all-zero column
    zero_cols = [c for c in df.columns if df[c].abs().sum() == 0]
    if zero_cols:
        df = df.drop(columns=zero_cols)
        print(f"Dropped all-zero columns: {zero_cols}")

    # Binary labels: benign/normal -> 0, any attack -> 1
    label_col = None
    for candidate in ['Label', 'label', 'CLASS', 'class']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Could not find label column in CICIDS2017 data")
    df[label_col] = df[label_col].apply(
        lambda x: 0 if str(x).lower() in ('benign', 'normal') else 1
    )
    df = df.rename(columns={label_col: 'label'})

    # Separate features and label
    y = df['label'].values.astype(np.int64)
    x = df.drop(columns=['label'])

    # One-hot encode categorical columns
    cat_cols = x.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        x = pd.get_dummies(x, columns=cat_cols, dummy_na=False)
        print(f"One-hot encoded {len(cat_cols)} categorical columns -> {x.shape[1]} features")

    # Train/test split 70/30, stratified
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )

    # Standardize continuous features (fit on train only)
    cont_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    means = x_train[cont_cols].mean(axis=0)
    stds = x_train[cont_cols].std(axis=0)
    stds[stds == 0] = 1.0
    x_train[cont_cols] = (x_train[cont_cols] - means) / stds
    x_test[cont_cols] = (x_test[cont_cols] - means) / stds

    # Ensure label is last column
    train_df = x_train.copy()
    train_df['label'] = y_train
    test_df = x_test.copy()
    test_df['label'] = y_test

    cols = [c for c in train_df.columns if c != 'label'] + ['label']
    train_df = train_df[cols]
    test_df = test_df[cols]

    train_path = os.path.join(DATA_DIR, "cicids2017_train.parquet")
    test_path = os.path.join(DATA_DIR, "cicids2017_test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Saved train -> {train_path} ({train_df.shape})")
    print(f"Saved test  -> {test_path} ({test_df.shape})")

    # Print SHA-256 for manifest verification
    for p in [train_path, test_path]:
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"SHA-256 {os.path.basename(p)}: {h}")

if __name__ == "__main__":
    load_and_preprocess()
