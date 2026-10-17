# Audits 1-4

## Audit 1: Feature coverage
- NSL-KDD: CATEGORICAL_GROUPS = [[4,5,6], [7..17]] from config (protocol, service). Total categorical features encoded as one-hot groups: 2 groups covering cols 4-17. These correspond to the protocol and service fields in NSL-KDD (after preprocessing). The evaluated threat model only perturbs these categorical groups (K<=1 changes one group assignment). No other categorical groups defined.
- CICIDS2017: CATEGORICAL_GROUPS = [[77,78,79]] only (protocol one-hot). Continuous = range(77). Total 80-dim. The threat model as implemented is restricted to protocol flips; other categorical-like aspects (if any) are not enumerated as groups.

## Audit 2: Category coverage
- CICIDS2017: Protocol group has 3 one-hot slots representing 3 states present (with 1 extremely rare). The implementation enumerates all 3 states as alternatives under K>=1 - full coverage of encoded states. No states dropped by config; all one-hot slots are part of the group. Justification: evaluates over encoded feature space as defined.
- NSL-KDD: Protocol has 3 states (all present in data). Service has 11 states (all present; some rare). All slots covered; enumerates all alternatives in groups.
- Limitation: “dropped categories, rare-category buckets, unused one-hot slots” - if preprocessing created unused slots, config would still enumerate them; we see all encoded states that appear in data are covered. The threat model is defined in the encoded space.

## Audit 3: Semantic validity
- Enumerated alternatives are syntactically valid one-hot vectors within each group (sum=1, exactly one active) - verified. Semantically, flipping protocol/service means choosing a different category value; for the encoded datasets, these correspond to different categorical assignments in the feature representation. The evaluation tests adversarial modifications in that encoded space under mixed-norm constraints. Whether all enumerated alternatives are “plausible” in the original network semantics depends on preprocessing (e.g. which protocol labels map to which encoded columns) - but from the repo artifacts, the protocol columns in CICIDS2017 are the three protocol states present after encoding. The implementation does not invent impossible values; it permutes to other existing category slots. So alternatives are valid within the defined feature space.
