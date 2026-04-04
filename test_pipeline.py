#!/usr/bin/env python
"""Test data I/O and training pipeline with test dataset."""

from pathlib import Path

import torch

from src.dataset import (
    WSIPatchDataset,
    build_dataloader,
    build_eval_transform,
    build_patch_records,
    build_sampler,
    build_train_transform,
    split_records_by_fold,
)
from src.utils import PatchConfig, set_seed


def test_data_io():
    """Test that we can load patches from the test dataset."""
    print("=" * 60)
    print("TEST 1: Data Loading and Patch Record Building")
    print("=" * 60)

    set_seed(42)
    
    # Build patch records from test dataset
    patch_config = PatchConfig()
    records = build_patch_records(
        csv_path="data_test/data_overview.csv",
        data_root="data_test",
        patch_config=patch_config,
        split="development",
        num_folds=3,
        skip_blank=False,  # Keep all patches for testing
        max_wsis=3,  # Use all 3 test WSIs
    )
    
    print(f"✓ Built {len(records)} patch records")
    if records:
        print(f"  Sample record: patient={records[0].patient_id}, wsi={records[0].wsi_id}, pos=({records[0].x}, {records[0].y})")
    
    # Split into train/val
    train_records, val_records = split_records_by_fold(records, fold_index=0)
    print(f"✓ Train records: {len(train_records)}, Val records: {len(val_records)}")
    
    # Build datasets
    train_dataset = WSIPatchDataset(
        train_records,
        patch_size=patch_config.patch_size,
        transform=build_train_transform()
    )
    val_dataset = WSIPatchDataset(
        val_records,
        patch_size=patch_config.patch_size,
        transform=build_eval_transform()
    )
    print(f"✓ Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # Build dataloaders
    if len(train_records) > 0:
        train_loader = build_dataloader(
            train_dataset,
            batch_size=min(2, len(train_records)),
            sampler=build_sampler(train_records),
            num_workers=0,
        )
        print(f"✓ Created train dataloader")
    else:
        print("⚠ No train records, skipping train dataloader")
        train_loader = None
    
    if len(val_records) > 0:
        val_loader = build_dataloader(
            val_dataset,
            batch_size=min(2, len(val_records)),
            shuffle=False,
            num_workers=0,
        )
        print(f"✓ Created val dataloader")
    else:
        print("⚠ No val records, skipping val dataloader")
        val_loader = None
    
    return train_loader, val_loader


def test_batch_loading(train_loader, val_loader):
    """Test that we can actually load batches."""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Loading")
    print("=" * 60)
    
    if train_loader is None:
        print("⚠ Skipping train batch loading (no train data)")
    else:
        print("Loading a train batch...")
        try:
            batch = next(iter(train_loader))
            print(f"✓ Loaded batch: image shape={batch['image'].shape}, mask shape={batch['mask'].shape}")
            print(f"  Image dtype={batch['image'].dtype}, Mask dtype={batch['mask'].dtype}")
            print(f"  Image min/max={batch['image'].min():.3f}/{batch['image'].max():.3f}")
            print(f"  Mask unique values={torch.unique(batch['mask']).tolist()}")
        except Exception as e:
            print(f"✗ Failed to load train batch: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if val_loader is None:
        print("⚠ Skipping val batch loading (no val data)")
    else:
        print("Loading a val batch...")
        try:
            batch = next(iter(val_loader))
            print(f"✓ Loaded batch: image shape={batch['image'].shape}, mask shape={batch['mask'].shape}")
        except Exception as e:
            print(f"✗ Failed to load val batch: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_model_forward():
    """Test that the model can do a forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Forward Pass")
    print("=" * 60)
    
    from src.model import build_model
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(num_classes=5, pretrained=False).to(device)
        print(f"✓ Built model on {device}")
        
        # Create a fake batch
        fake_image = torch.randn(2, 3, 512, 512).to(device)
        fake_mask = torch.randint(0, 5, (2, 512, 512)).to(device)
        
        # Forward pass
        logits = model(fake_image)
        print(f"✓ Forward pass: input {fake_image.shape} → output {logits.shape}")
        
        # Test loss
        from src.loss import FocalDiceLoss
        criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, ignore_index=0)
        loss = criterion(logits, fake_mask)
        print(f"✓ Loss computed: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STAG-Unet Pipeline Test with Synthetic Data")
    print("=" * 60 + "\n")
    
    train_loader, val_loader = test_data_io()
    
    if not test_batch_loading(train_loader, val_loader):
        print("\n✗ Batch loading failed!")
        exit(1)
    
    if not test_model_forward():
        print("\n✗ Model forward pass failed!")
        exit(1)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run training with the test dataset:")
    print("  python ./src/train.py --csv-path data_test/data_overview.csv --data-root data_test --epochs 2 --batch-size 2 --max-wsis 3")
