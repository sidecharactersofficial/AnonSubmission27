# Reviewer Verification Checklist

Use this checklist to verify the artifact systematically. Each section maps
to a claim or reproducibility requirement.

## Environment Setup
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Checkpoint manifest passes (`make verify`)
- [ ] Smoke tests pass (`make smoke`)

## Claim C1 — Legacy evaluation unreliability
- [ ] `canonical/section3_faithful_diagnostic.py` executes without error
- [ ] Baseline robust accuracy ≈ 16.06% at ε=0.15
- [ ] Hardened robust accuracy ≈ 40.36% at ε=0.15
- [ ] Retracted 29.10% figure is identified as unreliable

## Claim C2 — Exhaustive enumeration validity
- [ ] `canonical/run_paper_mixednorm.py --check` passes
- [ ] No invalid categorical states in mixed-norm K=0/1/2 counts
- [ ] Archived counts match paper tables

## Claim C3 — Canonical exhaustive K=1 accuracy
- [ ] Fresh evaluation generates `results/foolbox/exh_k1_*.json`
- [ ] `n_total` matches archive exactly
- [ ] `clean_correct` matches archive exactly
- [ ] `k0_survivors`, `k1_survivors` within ±0.75% relative
- [ ] `k1_pct_of_attacked` within ±0.5 pp absolute

## Claim C4 — Greedy projection artifacts
- [ ] `canonical/eval_jsma_vs_exhaustive.py` executes
- [ ] JSMA misses ~12.0% of true adversarial examples
- [ ] JSMA runtime is ~10x slower than PGD

## Claim C5 — Statistical validation
- [ ] `canonical/consolidated_canonical_table.py` runs successfully
- [ ] Aggregated mean±std tables are produced
- [ ] Multi-seed coverage matches manifest (NSL-KDD: 3 seeds, CICIDS2017: 13 seeds, UNSW-NB15: 9 seeds)

## Claim C6 — Independent attack validation
- [ ] `canonical/run_crown_vs_ibp.py` executes without soundness assertion failures
- [ ] Certified bounds align with empirical PGD-40 exhaustive masks
- [ ] C&W survivor records present in `results/cw/`
- [ ] DeepFool EXH K=1 records present in `results/foolbox/`

## Claim C7 — Logit reversal
- [ ] `canonical/trace_logit_reversal_hardened.py` executes
- [ ] Output is byte-identical to `results/logit_trace_model_adv_hardened.json`
- [ ] Per-sample logit margins flip under exhaustive K=1 categorical state changes

## Claim C8 — Scalability
- [ ] `canonical/eval_scalability.py` executes
- [ ] K=1 timing ≈ 0.0155 s/state on reference GPU
- [ ] K=2 timing ≈ 0.118 s/state on reference GPU

## Claim C9 — Mathematical Foundations & Machine-Checked Lean 4 Proofs
- [ ] `make verify-math` (or `python3 leanproofs/verify_math.py`) passes all 6 mathematical suites
- [ ] `make verify-lean` (or `cd leanproofs && lake build`) compiles cleanly with 0 errors and 0 warnings
- [ ] Proposition 1 proved: Coordinate separation $x_{c^*} - x_c \ge 1-2\epsilon > 0$, Euclidean distance difference $2(x_{c^*} - x_c)$, and unprojected sign-PGD step threshold
- [ ] Proposition 2 proved: Monotonicity under candidate set inclusion and guaranteed dominance of EXH over heuristic discrete attacks
- [ ] Theorem 1 proved: Discrete loss decomposition $\Delta J = \Delta^{(1)} + R$, SNAP blindness when local gradient is non-positive, and rank inversion condition
- [ ] Proposition 3 proved: Margin-bounded survival when clean margin exceeds single-group bound + continuous budget
- [ ] Proposition 4 proved: RanMASK hypergeometric overlap $\Delta = k_x/m \ge 0.5$ for $m \le 2$, proving certification is vacuous ($p \le 0.5$)
- [ ] Equation (6) verified: Exact combinatorial candidate counting ($K=1$: 37 states, 42 evaluations; $K=2$: 508 states, 667 evaluations on UNSW-NB15)
- [ ] Step-by-step mathematical audit report verified (`leanproofs/MATH_AUDIT_REPORT.md`)

## Provenance
- [ ] `PROVENANCE.md` documents methodological evolution
- [ ] `diagnostics/` contains superseded implementations
- [ ] `docs/adr/` contains ADR-001 through ADR-004
- [ ] Canonical vs superseded evaluators are clearly distinguished

## Evaluation Diagnostics (docs/postmortem)
- [ ] `docs/postmortem/001` diagnostic summary reviewed
- [ ] Evaluator-correctness diagnostic (002) confirms exhaustive enumeration
      and K=0/K=1 AND-semantics
- [ ] Reconciliation (003) explains the 11.69% ± 24.9% vs ~40% K=1 gap on
      CICIDS2017 RSC (seed variance + population semantics, not evaluator bugs)
- [ ] Threat-model clarification (004) matches the CICIDS2017 categorical
      mapping described in `docs/threat_model.md`
- [ ] Coverage audits (005) and representation-sensitivity finding (006,
      also in `LIMITATIONS.md`) acknowledged
- [ ] EXH vs SNAP canonical comparison (007): 66/75 positive K=1 gaps,
      K=0 exact match under controlled init; K-budget staging on UNSW-NB15
      notes K=1 threat-model incompleteness
- [ ] OPTIONAL — replication addendum (008): confirm it is self-contained
      under `replication/`, none of its artifacts are in
      `manifest/checkpoint_sha256.json`, and the added checkpoint/model files
      do not affect canonical claims; note the NF-ToN-IoT native-vs-onehot
      encoding result as additional support for the representation-sensitivity
      limitation

## Limitations
- [ ] Dataset reconstruction limitation understood (CICIDS2017/UNSW-NB15)
- [ ] Continuous optimization limitation understood (no global certificate)
- [ ] Random-start variance limitation understood (tolerance-based comparison)
- [ ] Legacy model provenance understood (some checkpoints from superseded pipelines)

## Artifact Integrity
- [ ] All 104 checkpoints verify against `manifest/checkpoint_sha256.json`
- [ ] Canonical results are present in `results/`
- [ ] No author-identifying information in committed files
- [ ] Remote repository is accessible via the anonymized 4open.science proxy link provided in the submission
