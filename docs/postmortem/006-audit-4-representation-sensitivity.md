# Audit 4: Representation sensitivity

## Current state
The evaluation as implemented is defined entirely in the encoded (one-hot) space with fixed categorical groupings. There is no systematic comparison in this codebase under alternative encodings (e.g., ordinal vs one-hot, different binning, or treating protocol as a single continuous/ordinal variable) or under alternative threat-model definitions (e.g., allowing different sets of allowed categorical transitions, or restricting to observed transitions). The configs and attack code assume the given one-hot group structure.

## What would be needed to satisfy audit 4
To properly assess representation sensitivity, one would need:
1. Alternative encodings of the same underlying categorical features (e.g., protocol encoded as integer index / ordinal, or target-encoded, or frequency-based) and corresponding model checkpoints trained under those encodings.
2. Alternative threat-model definitions (e.g., only allow flipping to protocol states that actually appear in similar traffic contexts, or constrain K>=1 to a subset of semantically plausible replacements) and corresponding evaluation logic.
3. Side-by-side results under (1) and (2) to quantify how much K1 robust accuracy changes.

## Observation
Given only the existing checkpoints and data in this repo, we cannot directly recompute results under materially different encodings without retraining models - that is outside the scope of a pure audit of the existing evaluation. However, the existing design (one-hot groups + exhaustive enumeration over all encoded categories) is the stated threat model; the key issue is whether the paper treats this as an encoding-dependent measurement or as a general property. The codebase itself does not contain such sensitivity analyses.

## Conclusion (Audit 4)
Not satisfied by the existing artifacts. The repo does not provide evidence that results are robust to defensible alternative encodings or threat-model definitions; no comparative experiments under different representations are present.
