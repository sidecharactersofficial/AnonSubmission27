CONTINUOUS_COLS = list(range(39))
CATEGORICAL_GROUPS = [
    [39, 40, 41, 42, 43, 44, 45],                              # PROTOCOL block
    [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],      # L7_PROTO block
]
FEATURE_DIM = 59
MIN_VAL = 0.0
MAX_VAL = 1.0
# NF-ToN-IoT-v2 RE-ENCODED (scripts/prep_nf_ton_iot_onehot.py): the NF
# encoding's flattened PROTOCOL / L7_PROTO numeric columns are now proper
# one-hot simplices (PROTOCOL: top-6 observed values + 'other' -> 7 slots,
# of which 2 are dead zero-occupancy in the subsample; L7_PROTO: top-12 +
# 'other' -> 13 slots). 'other' catches NaNs and rare values, and is a
# reachable-but-not-enumerated state — the takedown enumerates only states
# observed in the data, exactly like the other three datasets (ADR-001,
# data-driven state extraction, no fabricated states).
