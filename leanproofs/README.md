# BlackIce: Formal Verification & Mathematical Audit in Lean 4

This directory contains the machine-checked formal verification in Lean 4 and the step-by-step mathematical soundness verification for all theoretical foundations, bounds, and complexity counts established in the **BlackIce** paper (*IEEE S&P 2026*).

## Directory Structure

- `MATH_AUDIT_REPORT.md`: Comprehensive 30KB line-by-line mathematical audit report dissecting every proposition, theorem, and equation in `paper/main.tex` and Appendix A.
- `verify_math.py`: Standalone executable Python mathematical verification test suite checking algebraic identities, Taylor remainders, interval arithmetic, and combinatorial complexity.
- `BlackiceProofs/Basic.lean`: Lean 4 formal proofs for all 6 core mathematical theorems and propositions.
- `BlackiceProofs.lean`: Root library entry point for Lake.
- `Main.lean`: Executable driver reporting verification status.
- `lakefile.toml` / `lean-toolchain`: Lake project configuration (Lean v4.35.0).

---

## Verified Theorems and Propositions

| Paper Section | Label | Description | Lean Formalization | Status |
|:---|:---|:---|:---|:---:|
| **Section 4.1 & App. A.1** | `prop:linf` | **Proposition 1**: Structural $L_\infty$ Insufficiency & Metric Separation | `prop1_coordinate_separation`<br>`prop1_euclidean_distance_algebra`<br>`prop1_canonical_separation`<br>`prop1_step_threshold` | **PROVED** |
| **Section 4.3 & App. A.2** | `prop:exh_mono` | **Proposition 2**: Monotonicity & Dominance of Fiber Enumeration ($EXH$) | `prop2_supremum_monotone`<br>`prop2_exh_dominates_heuristic`<br>`prop2_robust_accuracy_inequality` | **PROVED** |
| **Section 6.1 & App. A.3** | `thm:masking` | **Theorem 1**: First-Order Taylor Misalignment of $SNAP$ & Rank Inversion | `thm1_taylor_decomposition`<br>`thm1_snap_masking`<br>`thm1_rank_inversion` | **PROVED** |
| **Section 6.2 & App. A.4** | `prop:margin` | **Proposition 3**: Margin-Bounded Survival & Multi-Group Scaling | `prop3_margin_survival`<br>`prop3_margin_failure` | **PROVED** |
| **Section 7.7 & App. A.5** | `prop:cert` | **Proposition 4**: Soundness of Mixed-Norm Certification & Low-$m$ RanMASK Vacuity | `prop4_ranmask_overlap_algebra`<br>`prop4_ranmask_delta_ge_half`<br>`prop4_ranmask_certification_vacuity`<br>`prop4_ranmask_impossible_strict`<br>`prop4_sandwich_relation` | **PROVED** |
| **Section 7.6.1 & App. A.6** | `eq:complexity` | **Equation (6)**: Exhaustive Candidate State Complexity Counting | `distinct_k1`<br>`unpruned_k1`<br>`unsw_nb15_k1_exact`<br>`unsw_nb15_k2_unpruned_exact`<br>`unsw_nb15_k2_distinct_exact` | **PROVED** |

---

## How to Run the Verification

### 1. Build and Run the Lean 4 Formal Proofs
```bash
export PATH="$HOME/.elan/bin:$PATH"
cd leanproofs
lake build
./.lake/build/bin/blackice_proofs
```

### 2. Run Individual File Verification in Lean 4
```bash
lean BlackiceProofs/Basic.lean
```

### 3. Run the Python Computational Verification Suite
```bash
python3 verify_math.py
```
