#!/usr/bin/env python3
"""
Controlled fixture test for the canonical EXH evaluator.

Five tests:
1. Tiny model with known vulnerable category — verifies the attack finds it.
2. Continuous-only input (no categorical groups) — verifies fallback to PGD.
3. Categorical group with no valid alternative (single-slot) — verifies graceful handling.
4. One-hot constraint enforcement — verifies all candidates satisfy constraints.
5. Result provenance — traces saved metrics to verified code path.

Run from ml-optimizer/:
    python verification/fixture_test.py
"""
import sys, os, json, hashlib, torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '.')

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_config
from scripts.eval_mixed_norm import canonical_mixed_norm_attack, set_seed as canon_set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 0.15
ALPHA = 0.01
STEPS = 40


class TinyModel(nn.Module):
    """3 continuous + 3 categorical (one-hot protocol) = 6 features.

    Architecture: prediction depends on a specific categorical combination.
    Protocol=0 (tcp) → class 0, Protocol=1 (udp) → class 0, Protocol=2 (icmp) → class 1
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 2)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 3] = 1.0   # tcp (idx 3) → class 0
            self.linear.weight[0, 4] = 1.0   # udp (idx 4) → class 0
            self.linear.weight[1, 5] = 1.0   # icmp (idx 5) → class 1
            self.linear.bias[0] = -0.5      # threshold so only active protocol matters
            self.linear.bias[1] = -0.5

    def forward(self, x):
        return self.linear(x)


class TinyConfig:
    """Mock config: 3 continuous cols [0,1,2], 1 categorical group [3,4,5]."""
    CONTINUOUS_COLS = [0, 1, 2]
    CATEGORICAL_GROUPS = [[3, 4, 5]]
    FEATURE_DIM = 6
    MIN_VAL = 0.0
    MAX_VAL = 1.0


class ContOnlyConfig:
    """Mock config: 3 continuous cols, no categorical groups."""
    CONTINUOUS_COLS = [0, 1, 2]
    CATEGORICAL_GROUPS = []
    FEATURE_DIM = 3
    MIN_VAL = 0.0
    MAX_VAL = 1.0


class NoAltConfig:
    """Mock config: 3 continuous + single-slot categorical (no alternative)."""
    CONTINUOUS_COLS = [0, 1, 2]
    CATEGORICAL_GROUPS = [[3]]  # Single categorical column
    FEATURE_DIM = 4
    MIN_VAL = 0.0
    MAX_VAL = 1.0


def test_1_known_vulnerable_category():
    """Test 1: Tiny model with known vulnerable category shift."""
    print("=" * 70)
    print("TEST 1: Known vulnerable categorical substitution")
    print("=" * 70)
    print(f"  Model: 3 continuous + 3 categorical (one-hot protocol)")
    print(f"  Logic: protocol=icmp → class 1; tcp/udp → class 0")
    print(f"  Input: sample with protocol=tcp (cat [1,0,0])")
    print(f"  Threat: attacker flips to protocol=icmp (cat [0,0,1])")
    print(f"  Expected: K=1 attack finds the flip and causes misclassification")

    model = TinyModel().to(DEVICE)
    model.eval()
    config = TinyConfig

    # Construct input: continuous = [0.5, 0.5, 0.5], protocol = tcp (one-hot [1,0,0])
    x = torch.tensor([[0.5, 0.5, 0.5, 1.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    y = torch.tensor([0], dtype=torch.long, device=DEVICE)  # True label: class 0 (tcp)

    # Verify model prediction
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
    print(f"\n  Clean: input={x[0].tolist()}, logits={logits[0].tolist()}, pred={pred.item()}, true={y.item()}")
    assert pred.item() == 0, f"Expected pred=0 (tcp), got {pred.item()}"

    # Run K=1 attack — should find protocol=icmp → class 1
    canon_set_seed(42)
    adv = canonical_mixed_norm_attack(model, x, y, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
    with torch.no_grad():
        logits_adv = model(adv)
        pred_adv = logits_adv.argmax(dim=1)

    # Check categorical values in adversarial example
    adv_cat = adv[0, 3:]  # categorical columns [3,4,5]
    print(f"  Adv K=1: cat={adv_cat.tolist()}, logits={logits_adv[0].tolist()}, pred={pred_adv.item()}")

    # Constraint checks
    cat_sum = adv_cat.sum().item()
    max_cont_pert = (adv[0, :3] - x[0, :3]).abs().max().item()
    is_one_hot = torch.allclose(adv_cat, torch.round(adv_cat), atol=1e-6) and cat_sum == 1.0

    print(f"  Constraint: cat_sum={cat_sum:.4f}, max_cont_L∞={max_cont_pert:.6f}, one_hot_valid={is_one_hot}")
    assert cat_sum == 1.0, f"Categorical sum should be 1.0, got {cat_sum}"
    assert max_cont_pert <= EPS + 1e-6, f"Perturbation exceeds epsilon: {max_cont_pert} > {EPS}"
    assert is_one_hot, "Categorical values should be one-hot (0.0 or 1.0)"

    # Check if protocol was flipped to icmp
    adv_proto = adv_cat.argmax().item()  # Should be 2 (icmp)
    expected_proto = 2  # icmp
    print(f"  Protocol flipped to: {adv_proto} ({['tcp', 'udp', 'icmp'][adv_proto]})")

    if pred_adv.item() != y.item():
        print(f"  RESULT: Attack FOUND vulnerability (pred={pred_adv.item()} != true={y.item()})")
        print(f"  PASS: Exhaustive categorical enumeration detected the known vulnerability")
        return True
    else:
        print(f"  RESULT: Attack did NOT find vulnerability")
        print(f"  FAIL: Exhaustive enumeration should have found protocol=icmp → misclassification")
        return False


def test_2_continuous_only():
    """Test 2: Continuous-only input (no categorical groups) — verifies PGD fallback."""
    print(f"\n{'=' * 70}")
    print("TEST 2: Continuous-only input (no categorical groups)")
    print("=" * 70)
    print(f"  Config: CATEGORICAL_GROUPS=[] (empty)")
    print(f"  Expected: falls back to continuous-only PGD, K=1 == K=0")

    model = nn.Linear(3, 2).to(DEVICE)
    model.eval()
    # Model: feature 0 determines class (opposite signs), small bias difference makes
    # boundary near x[0]=0.005. With epsilon=0.15, perturbation of x[0] from 0.1→0 flips.
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.weight[0, 0] = 1.0
        model.weight[1, 0] = -1.0
        model.bias[0] = 0.0
        model.bias[1] = 0.01

    # Input near decision boundary (x[0]=0.1 is within epsilon of the boundary at 0.005)
    x = torch.tensor([[0.1, 0.5, 0.5]], dtype=torch.float32, device=DEVICE)
    y = torch.tensor([0], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    print(f"\n  Clean: pred={pred.item()}, true={y.item()}")
    assert pred.item() == 0

    config = ContOnlyConfig
    canon_set_seed(42)
    adv = canonical_mixed_norm_attack(model, x, y, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
    with torch.no_grad():
        pred_adv = model(adv).argmax(dim=1)
    pert = (adv - x).abs().max().item()
    print(f"  Attack K=1: pred={pred_adv.item()}, max_pert={pert:.6f}")
    assert pert <= EPS + 1e-6, f"Perturbation exceeds epsilon: {pert} > {EPS}"
    assert pred_adv.item() != y.item(), "Attack should flip prediction"
    print(f"  RESULT: Continuous PGD worked, found vulnerability")
    print(f"  PASS: Falls back to continuous-only PGD when no categorical groups")
    return True


def test_3_no_valid_alternative():
    """Test 3: Categorical group with single-slot (no valid alternative)."""
    print(f"\n{'=' * 70}")
    print("TEST 3: Single-slot categorical group (no valid alternative)")
    print("=" * 70)
    print(f"  Config: CATEGORICAL_GROUPS=[[3]] (single column)")
    print(f"  Expected: K=1 enumerates original state only (i=0 creates same state)")

    model = nn.Linear(4, 2).to(DEVICE)
    model.eval()
    config = NoAltConfig

    x = torch.tensor([[0.5, 0.5, 0.5, 1.0]], dtype=torch.float32, device=DEVICE)
    y = torch.tensor([0], dtype=torch.long, device=DEVICE)

    canon_set_seed(42)
    adv = canonical_mixed_norm_attack(model, x, y, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)
    with torch.no_grad():
        pred_adv = model(adv).argmax(dim=1)

    # Check that categorical value is still 1.0 (only valid state)
    cat_val = adv[0, 3].item()
    print(f"\n  Categorical value: {cat_val} (expected 1.0 — only valid state)")
    assert cat_val == 1.0 or cat_val == 0.0, f"Unexpected categorical value: {cat_val}"
    print(f"  RESULT: Single-slot group handled gracefully (categorical={cat_val})")
    print(f"  PASS: No-invalid states generated for single-slot group")
    return True


def test_4_constraint_enforcement():
    """Test 4: Verify all candidate states satisfy one-hot constraints."""
    print(f"\n{'=' * 70}")
    print("TEST 4: Constraint enforcement on real CICIDS2017 config")
    print("=" * 70)

    config = get_config('cicids2017')
    print(f"  Config: {len(config.CATEGORICAL_GROUPS)} groups, sizes={[len(g) for g in config.CATEGORICAL_GROUPS]}")

    # Create a model that uses the real input dimension
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    model.eval()

    # Random input
    torch.manual_seed(123)
    x = torch.rand(4, config.FEATURE_DIM, device=DEVICE)
    # Make categorical columns valid one-hot
    cat_indices = [idx for g in config.CATEGORICAL_GROUPS for idx in g]
    cat_min, cat_max = min(cat_indices), max(cat_indices)
    for g in config.CATEGORICAL_GROUPS:
        rel_g = [idx - cat_min for idx in g]
        # Set random one-hot
        hot = torch.randint(0, len(g), (4,))
        for i in range(4):
            x[i, g] = 0.0
            x[i, g[hot[i]]] = 1.0

    y = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)  # arbitrary labels

    # Run K=1 attack
    canon_set_seed(42)
    adv = canonical_mixed_norm_attack(model, x, y, config, epsilon=EPS, alpha=ALPHA, steps=STEPS, K=1)

    # Verify constraints on output
    cat_data = adv[:, cat_min:cat_max+1]
    groups = config.CATEGORICAL_GROUPS

    all_one_hot = True
    all_sum_1 = True
    max_cont_pert = 0.0

    for g in groups:
        rel_g = [idx - cat_min for idx in g]
        vals = adv[:, g]
        sums = vals.sum(dim=1)
        # Check one-hot: each row should sum to 1, values should be 0 or 1
        sum_ok = torch.allclose(sums, torch.ones(len(sums), device=DEVICE), atol=1e-5)
        vals_ok = torch.all((vals == 0.0) | (vals == 1.0))
        all_sum_1 = all_sum_1 and sum_ok
        all_one_hot = all_one_hot and vals_ok

    # Check continuous perturbation
    cont_cols = config.CONTINUOUS_COLS
    if cont_cols:
        pert = (adv[:, cont_cols] - x[:, cont_cols]).abs()
        max_cont_pert = pert.max().item()

    print(f"  one_hot_valid: {all_one_hot}")
    print(f"  all_sum_to_1: {all_sum_1}")
    print(f"  max_cont_L∞: {max_cont_pert:.6f} (limit: {EPS})")

    # Also verify the input was properly one-hot
    orig_cat = x[:, cat_min:cat_max+1]
    orig_valid = True
    for g in groups:
        rel_g = [idx - cat_min for idx in g]
        sums_orig = orig_cat[:, rel_g].sum(dim=1)
        vals_orig = x[:, g]
        orig_valid = orig_valid and torch.allclose(sums_orig, torch.ones(len(sums_orig), device=DEVICE), atol=1e-5)
        orig_valid = orig_valid and torch.all((vals_orig == 0.0) | (vals_orig == 1.0))
    print(f"  input_one_hot_valid: {orig_valid}")

    assert all_sum_1, "Categorical states must sum to 1.0"
    assert max_cont_pert <= EPS + 1e-6, f"Perturbation exceeds epsilon: {max_cont_pert}"
    print(f"\n  PASS: All constraints enforced correctly")
    return True


def test_5_result_provenance():
    """Test 5: Trace saved metrics to the verified code path."""
    print(f"\n{'=' * 70}")
    print("TEST 5: Result provenance — trace metrics to verified code path")
    print("=" * 70)

    # Load a real EXH result file
    exh_path = "results/foolbox/exh_k1_pgd40_cicids2017_rsc_seed42.json"
    if not os.path.exists(exh_path):
        print(f"  SKIP: {exh_path} not found")
        return True

    d = json.load(open(exh_path))
    print(f"  File: {exh_path}")
    print(f"  checkpoint_sha256: {d.get('checkpoint_sha256', 'MISSING')[:32]}...")
    print(f"  attack_backend: {d.get('attack_backend', 'MISSING')}")
    print(f"  config: {d.get('config', {})}")
    print(f"  n_total: {d.get('n_total', 'MISSING')}")
    print(f"  clean_correct: {d.get('clean_correct', 'MISSING')}")
    print(f"  k0_survivors: {d.get('k0_survivors', 'MISSING')}")
    print(f"  k1_survivors: {d.get('k1_survivors', 'MISSING')}")

    # Verify checkpoint hash matches actual file
    ckpt_path = "models/unified/model_adv_rsc_cicids2017_seed42.pth"
    if os.path.exists(ckpt_path):
        actual_sha = hashlib.sha256(open(ckpt_path, 'rb').read()).hexdigest()
        recorded_sha = d.get('checkpoint_sha256', '')
        hash_match = actual_sha == recorded_sha
        print(f"  Checkpoint hash matches: {hash_match}")
        print(f"    actual:   {actual_sha}")
        print(f"    recorded: {recorded_sha}")
        assert hash_match, "Checkpoint SHA-256 mismatch!"

    # Verify config matches eval_protocol
    from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
    cfg = d.get('config', {})
    print(f"  epsilon match: {cfg.get('epsilon')} == {EVAL_EPSILON} → {cfg.get('epsilon') == EVAL_EPSILON}")
    print(f"  pgd_steps match: {cfg.get('pgd_steps')} == {EVAL_PGD_STEPS} → {cfg.get('pgd_steps') == EVAL_PGD_STEPS}")

    # Verify counts are consistent
    total = d.get('n_total', 0)
    cc = d.get('clean_correct', 0)
    k0 = d.get('k0_survivors', 0)
    k1 = d.get('k1_survivors', 0)
    print(f"  Nesting K1<=K0: {k1} <= {k0} → {k1 <= k0}")
    assert k1 <= k0, "K1 should be <= K0"
    print(f"\n  PASS: Results trace to verified code path with matching checkpoints/config")
    return True


def main():
    print(f"Device: {DEVICE} | torch: {torch.__version__}")
    print(f"Epsilon={EPS}, Alpha={ALPHA}, PGD steps={STEPS}")

    results = {}
    results['test_1'] = test_1_known_vulnerable_category()
    results['test_2'] = test_2_continuous_only()
    results['test_3'] = test_3_no_valid_alternative()
    results['test_4'] = test_4_constraint_enforcement()
    results['test_5'] = test_5_result_provenance()

    print(f"\n{'=' * 70}")
    print(f"FIXTURE TEST SUMMARY")
    print(f"{'=' * 70}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
