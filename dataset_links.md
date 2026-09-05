# Dataset Links

All dataset download and preprocessing logic lives in `data_scripts/`.

## Quick start

```bash
python3 data_scripts/download_data.py           # NSL-KDD
python3 data_scripts/download_cicids2017.py     # CICIDS2017
python3 data_scripts/download_unsw_nb15.py      # UNSW-NB15
```

Each script writes preprocessed files to `./data/` and prints SHA-256 hashes.

## Integrity anchors (SHA-256)

These hashes are over the exact tensors consumed by the shipped evaluation and
training scripts. If your preprocessed files match, your numbers will match.

| File | Rows | Cols* | SHA-256 |
|------|------|------|---------|
| `data/cicids2017_test.parquet`  | 623,869 | 81 | `4dff5049…1665` |
| `data/cicids2017_train.parquet` | 2,495,476 | 81 | `8e76734f…95e3` |
| `data/unsw_nb15_test.parquet`   | 508,010 | 80 | `fd542077…853e1` |
| `data/unsw_nb15_train.parquet`  | 2,032,037 | 80 | `6d5cf5c7…0544` |
| `data/nsl-kdd-test.csv`         | 22,543 | 19 | `e0e48f5b…9fcf3` |
| `data/nsl-kdd-train.csv`        | 125,973 | 19 | `56aeb5a7…b74af` |

\* Column counts are the effective feature width after the loader drops the
`__index_level_0__` index column carried by the Parquet files (81 = features +
`label`; the `__index_level_0__` column is dropped before training/eval).

> Full 64-char hashes are stored in `manifest/dataset_sha256.txt`
> (`sha256sum -c` friendly). Verify with:
> `cp manifest/dataset_sha256.txt . && sha256sum -c dataset_sha256.txt`.
> All reported evaluations use the **full** test sets: UNSW-NB15 = 508,010 rows,
> CICIDS2017 = 623,869 rows, NSL-KDD = 22,543 rows. (One superseded legacy file,
> `results/results_unsw_nb15_Hardened.json`, was produced by a partial run with
> `total = 348,000`; it is not a source of any paper table and is superseded by
> the full-test-set foolbox sweeps and by `results_unsw_nb15_Hardened_fgsm.json`.)

## Dataset sources and preprocessing

### NSL-KDD

**Script:** `data_scripts/download_data.py`

Downloads raw CSVs from `defcom17/NSL_KDD`, keeps 4 continuous + 2 categorical
groups + `label`, one-hots categoricals, min-max scales continuous columns on
train statistics only. Output: `data/nsl-kdd-train.csv`, `data/nsl-kdd-test.csv`.

### CICIDS2017

**Script:** `data_scripts/download_cicids2017.py`

Downloads six selected day CSVs from public mirrors, drops duplicate `Label` +
`Fwd Header Length` columns, drops all-zero `Fwd Packet Length Max`, re-codes
multi-class labels to binary, one-hot encodes categoricals, standardizes on
train only. 70/30 train-test split. Output: `data/cicids2017_train.parquet`,
`data/cicids2017_test.parquet`.

### UNSW-NB15

**Script:** `data_scripts/download_unsw_nb15.py`

Downloads `UNSW_NB15_4.csv` and feature definitions, drops row-identity columns
(`srcip/dstip/srcport/dstport`) and `attack_cat`, re-codes `label` to binary,
then constructs the canonical 79-dimensional feature layout defined in
`app/ml/data/unsw_nb15_config.py`:

- 38 continuous features (indices 0–37)
- 11 proto one-hot features (indices 38–48), top-10 + OTHER
- 11 state one-hot features (indices 49–59), top-10 + OTHER
- 13 service one-hot features (indices 60–72), top-12 + OTHER
- 2 is_sm_ips_ports one-hot features (indices 73–74)
- 4 is_ftp_login one-hot features (indices 75–78)

High-cardinality categorical columns are capped at top-k frequencies in the
training set to ensure deterministic dimensions. Continuous features are
standardized on train statistics only. 70/30 train-test split.

Output: `data/unsw_nb15_train.parquet`, `data/unsw_nb15_test.parquet`.

> **Note:** The shipped `data/unsw_nb15_*.parquet` files are the authoritative
> source for exact reproduction. This script provides a best-effort from-scratch
> regeneration path; output hashes will likely differ from the shipped manifest.
> For exact reproduction, use the shipped parquet files.

## Manual download fallback

If automatic download URLs are unavailable (GitHub mirrors are occasionally
taken down), manually download the raw datasets and place them in `./data/`:

- **CICIDS2017**: `https://www.unb.ca/cic/datasets/ids-2017.html`
- **UNSW-NB15**: `https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/research/unsw-nb15-dataset/`

Then re-run the corresponding script; it will detect existing files and skip
download.
