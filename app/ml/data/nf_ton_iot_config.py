CONTINUOUS_COLS = list(range(41))
CATEGORICAL_GROUPS = []
FEATURE_DIM = 41
MIN_VAL = 0.0
MAX_VAL = 1.0
# NF-ToN-IoT-v2 is pure NetFlow: every feature is a continuous flow statistic
# already normalized to [0,1] by scripts/prep_nf_ton_iot.py (quantile-clip +
# min-max). No one-hot groups — the constraint builder pads the empty set
# with a neutral tautology so the official CAA init does not crash.
