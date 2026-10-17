# Threat Model Clarification (CICIDS2017)

## Categorical group mapping
- Dataset: CICIDS2017 (preprocessed parquet files in ml-optimizer/data/)
- Categorical group defined: CATEGORICAL_GROUPS = [[77, 78, 79]] (config: app/ml/data/cicids2017_config.py)
- Feature columns (by index in the model input): 77→Protocol_0.0, 78→Protocol_0.3529411764705882, 79→Protocol_1.0 (from parquet column order; positions match config)
- Encoding: one-hot over protocol; for each flow exactly one of these is 1.0.
- Observed distribution (train): Protocol_0.352941… dominates (1694466), Protocol_1.0 next (799654), Protocol_0.0 rare (1356). These are the three protocol states present in the encoded dataset.

## K=1 threat model (as implemented)
- K=1 means: change the value of exactly one categorical feature group to another valid category (enumerate all other one-hot assignments within that group), then run continuous PGD under L∞ bound on continuous features [0..76] for each resulting state; worst case taken by loss. 
- For CICIDS2017 this is a single group of size 3 → 2 alternative protocol states per sample (in addition to original). NSL-KDD has 2 groups (sizes 3,11) → more alternatives total.

## Implications for cross-dataset comparison
- The categorical attack surface differs (coverage and cardinality). Therefore “same nominal attack configuration” is true for eps/alpha/steps and K semantics, but the concrete categorical perturbation space differs by dataset by design. Any claim about “dataset-dependent” behavior must be explicit about this difference in threat model scope.
