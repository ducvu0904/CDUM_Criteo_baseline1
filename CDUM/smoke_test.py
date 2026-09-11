#!/usr/bin/env python3
"""Smoke test for the CPM (Coarse-grained Preference Modeling) module.

Run from the project root:
    python -m CDUM.smoke_test
  or:
    python CDUM/smoke_test.py
"""

import sys
import os
import traceback

# Ensure project root is on the path so `CDUM` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_imports():
    """Test 1: All modules import without error."""
    header("Test 1: Imports")
    from CDUM.encoder import FeatureEncoder
    from CDUM.treatment_refine import TreatmentRefine
    from CDUM.experts import Expert, UserExpert, GuidanceGate
    from CDUM.cpm import TreatmentTower, CPM
    from CDUM.trainer import CPMTrainer
    print("  ✅ All imports successful")


def test_encoder(B=8, num_features=12, num_bins=100, emb_dim=32):
    """Test 2: FeatureEncoder produces correct shapes."""
    header("Test 2: FeatureEncoder")
    from CDUM.encoder import FeatureEncoder

    enc = FeatureEncoder(num_features=num_features, num_bins=num_bins, embedding_dim=emb_dim)
    x_ids = torch.randint(0, num_bins, (B, num_features))
    t = torch.randint(0, 2, (B,))

    x_emb = enc.encode_features(x_ids)
    assert x_emb.shape == (B, num_features, emb_dim), \
        f"Expected ({B}, {num_features}, {emb_dim}), got {x_emb.shape}"
    print(f"  ✅ encode_features: {x_emb.shape}")

    t_emb = enc.encode_treatment(t)
    assert t_emb.shape == (B, emb_dim), \
        f"Expected ({B}, {emb_dim}), got {t_emb.shape}"
    print(f"  ✅ encode_treatment: {t_emb.shape}")


def test_treatment_refine(B=8, emb_dim=32, hidden=64, out_dim=16):
    """Test 3: TreatmentRefine produces guidance & indicator with correct shapes."""
    header("Test 3: TreatmentRefine")
    from CDUM.treatment_refine import TreatmentRefine

    refine = TreatmentRefine(treatment_dim=emb_dim, hidden_dim=hidden, output_dim=out_dim)
    t_emb = torch.randn(B, emb_dim)
    e_gui, e_ind = refine(t_emb)

    assert e_gui.shape == (B, out_dim), f"Guidance: expected ({B}, {out_dim}), got {e_gui.shape}"
    assert e_ind.shape == (B, out_dim), f"Indicator: expected ({B}, {out_dim}), got {e_ind.shape}"
    # Both should be in [0, 1] because of sigmoid
    assert e_gui.min() >= 0 and e_gui.max() <= 1, "Guidance not in [0,1] — missing sigmoid?"
    assert e_ind.min() >= 0 and e_ind.max() <= 1, "Indicator not in [0,1] — missing sigmoid?"
    print(f"  ✅ e_guidance: {e_gui.shape}, range [{e_gui.min():.3f}, {e_gui.max():.3f}]")
    print(f"  ✅ e_indicator: {e_ind.shape}, range [{e_ind.min():.3f}, {e_ind.max():.3f}]")


def test_experts(B=8, input_dim=384, hidden_dim=64, expert_dim=32, num_experts=3):
    """Test 4: UserExpert & GuidanceGate produce correct shapes."""
    header("Test 4: UserExpert & GuidanceGate")
    from CDUM.experts import UserExpert, GuidanceGate

    experts = UserExpert(num_experts=num_experts, input_dim=input_dim,
                         hidden_dim=hidden_dim, expert_dim=expert_dim)
    x = torch.randn(B, input_dim)
    out = experts(x)
    assert out.shape == (B, num_experts, expert_dim), \
        f"UserExpert: expected ({B}, {num_experts}, {expert_dim}), got {out.shape}"
    print(f"  ✅ UserExpert output: {out.shape}")

    gate = GuidanceGate(guidance_dim=16, num_experts=num_experts)
    g_emb = torch.randn(B, 16)
    weights = gate(g_emb)
    assert weights.shape == (B, num_experts), \
        f"GuidanceGate: expected ({B}, {num_experts}), got {weights.shape}"
    # Softmax: each row should sum to 1
    row_sums = weights.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), \
        f"Gate weights don't sum to 1: {row_sums}"
    print(f"  ✅ GuidanceGate output: {weights.shape}, row sums ≈ 1.0")


def test_treatment_tower(B=8, expert_dim=32, hidden_dim=64):
    """Test 5: TreatmentTower produces scalar output."""
    header("Test 5: TreatmentTower")
    from CDUM.cpm import TreatmentTower

    tower = TreatmentTower(input_dim=expert_dim, hidden_dim=hidden_dim)
    mixed = torch.randn(B, expert_dim)
    e_ind = torch.sigmoid(torch.randn(B, hidden_dim))  # indicator in [0,1]
    y = tower(mixed, e_ind)
    assert y.shape == (B, 1), f"Expected ({B}, 1), got {y.shape}"
    print(f"  ✅ TreatmentTower output: {y.shape}")


def test_cpm_forward(B=8, num_features=12, num_bins=100, emb_dim=32,
                     refine_hidden=64, refine_dim=64,
                     num_experts=3, expert_hidden=64, expert_dim=32):
    """Test 6: Full CPM forward pass — shapes, keys, and dimension compatibility."""
    header("Test 6: CPM Forward Pass")
    from CDUM.cpm import CPM

    # ── Dimension check ──────────────────────────────────────────────
    # UserExpert input_dim must match flattened feature embedding:
    #   x_emb.flatten(start_dim=1) → (B, num_features * emb_dim)
    # But in CPM.__init__, UserExpert gets input_dim=embedding_dim.
    # This is a bug if num_features > 1.
    actual_flat_dim = num_features * emb_dim  # 12 * 32 = 384

    print(f"  Flattened feature dim: {num_features} x {emb_dim} = {actual_flat_dim}")
    print(f"  UserExpert input_dim in CPM.__init__: embedding_dim = {emb_dim}")

    if actual_flat_dim != emb_dim:
        print(f"  ⚠️  DIMENSION MISMATCH: UserExpert expects {emb_dim} but will receive {actual_flat_dim}")
        print(f"     This will cause a RuntimeError in the forward pass.")
        print(f"     Fix: change CPM.__init__ line 44 to:")
        print(f"       input_dim=num_features * embedding_dim")

    # Try forward — expect crash if dimensions are wrong
    model = CPM(
        num_features=num_features,
        num_bins=num_bins,
        embedding_dim=emb_dim,
        refine_hidden_dim=refine_hidden,
        refine_dim=refine_dim,
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden,
        expert_dim=expert_dim,
    )

    x_ids = torch.randint(0, num_bins, (B, num_features))
    t = torch.randint(0, 2, (B,))

    try:
        outputs = model(x_ids, t)
    except RuntimeError as e:
        if "mat1 and mat2" in str(e) or "size mismatch" in str(e).lower():
            print(f"\n  ❌ FORWARD CRASHED with dimension error (as predicted):")
            print(f"     {e}")
            print(f"\n  Fix in cpm.py line 44:")
            print(f"     self.user_experts = UserExpert(... input_dim=num_features * embedding_dim ...)")
            return False
        raise

    # If we got here, verify output dict
    expected_keys = {"y_factual", "y0_hat", "y1_hat", "uplift"}
    assert set(outputs.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {set(outputs.keys())}"
    print(f"  ✅ Output keys: {sorted(outputs.keys())}")

    for key in ["y_factual", "y0_hat", "y1_hat", "uplift"]:
        assert outputs[key].shape == (B, 1), \
            f"  {key}: expected ({B}, 1), got {outputs[key].shape}"
    print(f"  ✅ All output shapes: ({B}, 1)")

    # Factual masking: for t=0 samples, y_factual should equal y0_hat
    mask_0 = (t == 0)
    if mask_0.any():
        y_f = outputs["y_factual"][mask_0]
        y_0 = outputs["y0_hat"][mask_0]
        assert torch.allclose(y_f, y_0, atol=1e-6), "y_factual != y0_hat for control samples"
    mask_1 = (t == 1)
    if mask_1.any():
        y_f = outputs["y_factual"][mask_1]
        y_1 = outputs["y1_hat"][mask_1]
        assert torch.allclose(y_f, y_1, atol=1e-6), "y_factual != y1_hat for treatment samples"
    print("  ✅ Factual masking correct (y_factual = y0 when t=0, y1 when t=1)")

    return True


def test_gradient_flow(B=16, num_features=12, num_bins=100, emb_dim=32,
                       refine_hidden=64, refine_dim=64,
                       num_experts=3, expert_hidden=64, expert_dim=32):
    """Test 7: Gradients flow to all parameters."""
    header("Test 7: Gradient Flow")
    from CDUM.cpm import CPM

    model = CPM(
        num_features=num_features, num_bins=num_bins, embedding_dim=emb_dim,
        refine_hidden_dim=refine_hidden, refine_dim=refine_dim,
        num_experts=num_experts, expert_hidden_dim=expert_hidden, expert_dim=expert_dim,
    )

    x_ids = torch.randint(0, num_bins, (B, num_features))
    t = torch.randint(0, 2, (B,))
    y = torch.randn(B, 1)

    outputs = model(x_ids, t)
    loss = nn.HuberLoss()(outputs["y_factual"], y)
    loss.backward()

    no_grad_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad_params.append(name)

    if no_grad_params:
        print(f"  ⚠️  Parameters with no gradient ({len(no_grad_params)}):")
        for n in no_grad_params:
            print(f"      - {n}")
    else:
        total = sum(1 for _ in model.parameters())
        print(f"  ✅ All {total} parameters received gradients")


def test_trainer(B=32, num_features=12, num_bins=100, emb_dim=32,
                 refine_hidden=64, refine_dim=64,
                 num_experts=3, expert_hidden=64, expert_dim=32):
    """Test 8: CPMTrainer can train for 2 epochs and evaluate."""
    header("Test 8: CPMTrainer Integration")
    from CDUM.cpm import CPM
    from CDUM.trainer import CPMTrainer

    model = CPM(
        num_features=num_features, num_bins=num_bins, embedding_dim=emb_dim,
        refine_hidden_dim=refine_hidden, refine_dim=refine_dim,
        num_experts=num_experts, expert_hidden_dim=expert_hidden, expert_dim=expert_dim,
    )

    # Fake dataset
    N = 128
    x_ids = torch.randint(0, num_bins, (N, num_features))
    t = torch.randint(0, 2, (N,))
    y = torch.randn(N)
    dataset = TensorDataset(x_ids, t, y)
    train_loader = DataLoader(dataset, batch_size=B, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=B)

    trainer = CPMTrainer(model=model, lr=1e-3, device="cpu")

    # Train 2 epochs
    history = trainer.fit(train_loader, val_loader, epochs=2,
                          early_stopping_patience=5, verbose=0)
    assert len(history["train_loss"]) == 2, f"Expected 2 train losses, got {len(history['train_loss'])}"
    assert len(history["val_loss"]) == 2, f"Expected 2 val losses, got {len(history['val_loss'])}"
    print(f"  ✅ Training: epoch 1 loss={history['train_loss'][0]:.4f}, epoch 2 loss={history['train_loss'][1]:.4f}")

    # Evaluate
    results = trainer.evaluate(val_loader)
    assert "loss" in results, "Evaluate should return 'loss'"
    print(f"  ✅ Evaluation: loss={results['loss']:.4f}")

    # Loss should be finite
    assert all(not (x != x) for x in history["train_loss"]), "NaN in train loss!"
    print("  ✅ No NaN in losses")


def test_save_load():
    """Test 9: Save and load model weights."""
    header("Test 9: Save / Load")
    import tempfile
    from CDUM.cpm import CPM
    from CDUM.trainer import CPMTrainer

    model = CPM(
        num_features=12, num_bins=100, embedding_dim=32,
        refine_hidden_dim=64, refine_dim=64,
        num_experts=3, expert_hidden_dim=64, expert_dim=32,
    )
    trainer = CPMTrainer(model=model, device="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cpm_test.pth")
        trainer.save(path)
        assert os.path.exists(path), "Checkpoint file not created"
        print(f"  ✅ Saved to {path} ({os.path.getsize(path)} bytes)")

        trainer.load(path)
        print("  ✅ Loaded successfully")


# ──────────────────────────────────────────────────────────────────────
def main():
    print("+" + "="*58 + "+")
    print("|          CPM Smoke Test Suite                            |")
    print("+" + "="*58 + "+")

    passed = 0
    failed = 0
    results = []

    tests = [
        ("Imports", test_imports),
        ("FeatureEncoder", test_encoder),
        ("TreatmentRefine", test_treatment_refine),
        ("UserExpert & GuidanceGate", test_experts),
        ("TreatmentTower", test_treatment_tower),
        ("CPM Forward Pass", test_cpm_forward),
        ("Gradient Flow", test_gradient_flow),
        ("CPMTrainer Integration", test_trainer),
        ("Save / Load", test_save_load),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            # test_cpm_forward returns False if dimension mismatch detected
            if result is False:
                failed += 1
                results.append((name, "FAIL"))
            else:
                passed += 1
                results.append((name, "PASS"))
        except Exception:
            failed += 1
            results.append((name, "FAIL"))
            traceback.print_exc()

    # Summary
    header("Summary")
    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")

    total = passed + failed
    print(f"\n  Result: {passed}/{total} passed", end="")
    if failed:
        print(f" ({failed} FAILED)")
        sys.exit(1)
    else:
        print(" -- All clear!")
        sys.exit(0)


if __name__ == "__main__":
    main()
