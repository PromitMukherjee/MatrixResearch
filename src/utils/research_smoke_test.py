"""
Research Smoke Test System
==========================
Automated pre-training validation to prevent wasting compute on broken experiments.
Runs a battery of sanity checks before launching expensive training runs.
"""

import os
import sys
import json
import math
import shutil
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.matrix_dataset import DynamicMatrixDataset
from src.models.autoencoder import MatrixAutoencoder


def run_research_smoke_test(size=16, latent_dim=128, matrix_type="low_rank", rank=4):
    """
    Runs a complete battery of research smoke tests before full training.
    
    Returns True if all tests pass, False otherwise.
    """
    SEP = "=" * 41
    results = {}
    all_passed = True

    print(f"\n{SEP}")
    print("  Running Research Smoke Tests...")
    print(f"{SEP}\n")

    # ------------------------------------------------------------------
    # Test 1: Dataset Validation
    # ------------------------------------------------------------------
    test_name = "Dataset Validation"
    try:
        dataset = DynamicMatrixDataset(
            num_samples=100,
            size=size,
            matrix_type=matrix_type,
            rank=rank
        )

        assert len(dataset) > 0, "Dataset length is 0"

        sample = dataset[0]
        A = sample['A']

        assert not torch.isnan(A).any(), "NaN detected in sample matrix A"
        assert not torch.isinf(A).any(), "Inf detected in sample matrix A"
        assert A.shape == (size, size), f"Expected shape ({size},{size}), got {A.shape}"

        results[test_name] = "PASS"
        print(f"  [PASS] {test_name}")
    except Exception as e:
        results[test_name] = "FAIL"
        all_passed = False
        print(f"  [FAIL] {test_name}")
        print(f"         Reason: {e}")
        _print_summary(results, all_passed)
        return False

    # ------------------------------------------------------------------
    # Test 2: Rank Validation
    # ------------------------------------------------------------------
    test_name = "Rank Validation"
    try:
        computed_rank = torch.linalg.matrix_rank(A).item()
        assert computed_rank <= rank, \
            f"Computed rank {computed_rank} exceeds expected max rank {rank}"

        results[test_name] = "PASS"
        print(f"  [PASS] {test_name} (rank={computed_rank}, expected<={rank})")
    except Exception as e:
        results[test_name] = "FAIL"
        all_passed = False
        print(f"  [FAIL] {test_name}")
        print(f"         Reason: {e}")
        _print_summary(results, all_passed)
        return False

    # ------------------------------------------------------------------
    # Test 3: Normalization Validation
    # ------------------------------------------------------------------
    test_name = "Normalization"
    try:
        std_val = A.std().item()
        assert 0.8 <= std_val <= 1.2, \
            f"Std {std_val:.4f} outside expected range [0.8, 1.2]"

        results[test_name] = "PASS"
        print(f"  [PASS] {test_name} (std={std_val:.4f})")
    except Exception as e:
        results[test_name] = "FAIL"
        all_passed = False
        print(f"  [FAIL] {test_name}")
        print(f"         Reason: {e}")
        _print_summary(results, all_passed)
        return False

    # ------------------------------------------------------------------
    # Test 3.5: CUDA Check (non-blocking)
    # ------------------------------------------------------------------
    test_name = "CUDA"
    if torch.cuda.is_available():
        results[test_name] = "PASS"
        print(f"  [PASS] {test_name} ({torch.cuda.get_device_name(0)})")
    else:
        results[test_name] = "PASS"
        print(f"  [PASS] {test_name} (CPU mode — no GPU required for smoke test)")

    # ------------------------------------------------------------------
    # Test 4: Autoencoder Forward Pass
    # ------------------------------------------------------------------
    test_name = "Forward Pass"
    try:
        device = torch.device("cpu")
        model = MatrixAutoencoder(size=size, latent_dim=latent_dim).to(device)
        A_batch = A.unsqueeze(0).to(device)
        A_rec = model(A_batch)

        assert A_rec.shape == A_batch.shape, \
            f"Output shape {A_rec.shape} != input shape {A_batch.shape}"
        assert not torch.isnan(A_rec).any(), "NaN in forward pass output"
        assert not torch.isinf(A_rec).any(), "Inf in forward pass output"

        results[test_name] = "PASS"
        print(f"  [PASS] {test_name}")
    except Exception as e:
        results[test_name] = "FAIL"
        all_passed = False
        print(f"  [FAIL] {test_name}")
        print(f"         Reason: {e}")
        _print_summary(results, all_passed)
        return False

    # ------------------------------------------------------------------
    # Test 5: Short Training Validation (3 epochs)
    # ------------------------------------------------------------------
    test_name = "Training"
    try:
        device = torch.device("cpu")
        train_dataset = DynamicMatrixDataset(1000, size=size, matrix_type=matrix_type, rank=rank)
        val_dataset = DynamicMatrixDataset(100, size=size, matrix_type=matrix_type, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = MatrixAutoencoder(size=size, latent_dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        epoch_losses = []
        for epoch in range(1, 4):
            model.train()
            total_loss = 0
            for batch in train_loader:
                A_b = batch['A'].to(device)
                A_rec = model(A_b)
                loss = criterion(A_rec, A_b)

                # Test 6: NaN Monitoring (inline)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss detected at epoch {epoch}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"         Smoke epoch {epoch}/3 — loss: {avg_loss:.6f}")

        epoch1_loss = epoch_losses[0]
        epoch3_loss = epoch_losses[2]
        assert epoch3_loss < epoch1_loss, \
            f"Loss did not decrease: epoch1={epoch1_loss:.6f}, epoch3={epoch3_loss:.6f}"

        results[test_name] = "PASS"
        print(f"  [PASS] {test_name} (loss: {epoch1_loss:.4f} → {epoch3_loss:.4f})")
    except Exception as e:
        results[test_name] = "FAIL"
        all_passed = False
        print(f"  [FAIL] {test_name}")
        print(f"         Reason: {e}")
        _print_summary(results, all_passed)
        return False

    # ------------------------------------------------------------------
    # Test 7: Report Generation Validation
    # ------------------------------------------------------------------
    test_name = "Report Generation"
    try:
        report_dir = "reports/smoke_test"
        os.makedirs(report_dir, exist_ok=True)

        # Write test metrics.json
        test_metrics = {
            "smoke_test": True,
            "epoch1_loss": epoch_losses[0],
            "epoch3_loss": epoch_losses[2]
        }
        metrics_path = os.path.join(report_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=4)

        # Write test history.csv
        history_path = os.path.join(report_dir, "history.csv")
        history_df = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train_loss": epoch_losses
        })
        history_df.to_csv(history_path, index=False)

        # Verify files exist
        assert os.path.isfile(metrics_path), f"metrics.json not created at {metrics_path}"
        assert os.path.isfile(history_path), f"history.csv not created at {history_path}"

        results[test_name] = "PASS"
        print(f"  [PASS] {test_name}")
    except Exception as e:
        results[test_name] = "FAIL"
        all_passed = False
        print(f"  [FAIL] {test_name}")
        print(f"         Reason: {e}")
        _print_summary(results, all_passed)
        return False

    # ------------------------------------------------------------------
    # Test 8: Summary Table
    # ------------------------------------------------------------------
    _print_summary(results, all_passed)
    return all_passed


def _print_summary(results, all_passed):
    """Prints the final summary table of all smoke test results."""
    SEP = "=" * 41
    print(f"\n{SEP}")
    print("  Research Smoke Test Results")
    print(f"{SEP}")

    # Ensure consistent ordering
    test_order = [
        "Dataset Validation",
        "Rank Validation",
        "Normalization",
        "CUDA",
        "Forward Pass",
        "Training",
        "Report Generation"
    ]
    for test in test_order:
        status = results.get(test, "SKIP")
        print(f"  {test:<25s} {status}")

    print(f"{SEP}")
    if all_passed:
        print("  READY FOR FULL TRAINING")
    else:
        print("  NOT READY FOR FULL TRAINING")
    print(f"{SEP}\n")


if __name__ == "__main__":
    passed = run_research_smoke_test()
    if not passed:
        sys.exit(1)
    print("Smoke test passed. Safe to launch full training.")
