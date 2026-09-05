# Datasets

The repository does **not** bundle the raw datasets. Three options exist:

1. **NSL-KDD** — regenerated automatically by the in-repo `download_data.py`
   (pull from the public `defcom17/NSL_KDD` mirror, preprocess deterministically
   to `data/nsl-kdd-train.csv` / `data/nsl-kdd-test.csv`).
2. **CICIDS2017 / UNSW-NB15** — preprocessed to Parquet **offline** (not
   reproducible by in-repo scripts). Download the raw originals and reproduce
   the preprocessing steps documented below, or use the integrity anchors to
   verify already-preprocessed copies.

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

## CICIDS2017 / UNSW-NB15 preprocessing (documented, not scripted)

- **CICIDS2017.** Use the `Intrusion_Detection_Evaluation_Dataset` (CICIDS2017),
  `pcap`-derived CSV. Steps: select the final two days
  (`Wednesday-WorkingHours.pcap_ISCX.csv`, `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`,
  `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`,
  `Friday-WorkingHours-Morning.pcap_ISCX.csv`,
  `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`,
  `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`), drop the duplicate
  `Label` + the `Fwd Header Length` duplicate, re-code the multi-class labels to
  binary (any attack → `1`), drop the all-zero `Fwd Packet Length Max` column,
  and leak-safe one-hot + standardize (fit on train only). Final 81-column
  schema (`label` last).
- **UNSW-NB15.** Use `UNSW_NB15_4.csv` augmented with the five
  `*_labels.parquet`/feature files for the `attack_cat` split, drop
  `attack_cat`, re-code `label` to binary, drop the `srcip/dstip/srcport/dstport`
  row-identity columns, then the same one-hot + standardize pipeline. Final
  80-column schema (`label` last).

Leak-safe preprocessing (fit standardization on **train** only) is essential to
reproduce reported numbers; the exact train-holdout split is `train: 70%`,
`test: 30%`.

## NSL-KDD

```
python3 download_data.py          # writes data/nsl-kdd-train.csv + nsl-kdd-test.csv
```

Downloads the raw ARFF/TXT from the public mirror `defcom17/NSL_KDD` on GitHub,
keeps the 22 continuous + 3 categorical + `label` fields, one-hots the
categoricals, and standardizes continuous columns on train statistics only.