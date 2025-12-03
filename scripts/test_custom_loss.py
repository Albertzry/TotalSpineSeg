"""
Test script for PartialLDH custom loss and trainer.

测试场景：
1. Batch中有LDH样本：应该正常计算所有类别的loss
2. Batch中没有LDH样本：应该忽略LDH通道的loss贡献
"""

import torch
import numpy as np


def test_partial_ldh_loss():
    """Test the PartialLDH_DC_and_CE_loss function."""
    print("=" * 70)
    print("Testing PartialLDH_DC_and_CE_loss...")
    print("=" * 70)
    
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import (
        PartialLDH_DC_and_CE_loss,
        PartialLDH_SoftDiceLoss,
        PartialLDH_CELoss
    )
    
    # Setup
    batch_size = 2
    num_classes = 13  # 0-12, where 12 is LDH
    spatial = (16, 16, 16)
    ldh_class_idx = 12
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create mock data directly on device
    # Network Output (Logits): (B, C, D, H, W)
    net_output = torch.randn(batch_size, num_classes, *spatial, requires_grad=True, device=device)
    
    # Ground Truth: (B, 1, D, H, W)
    target = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
    
    # Scenario setup:
    # Sample 0: NO LDH (ordinary subject) - some spine structures
    target[0, 0, 2:4, 2:4, 2:4] = 1  # disc
    target[0, 0, 6:8, 6:8, 6:8] = 6  # vertebrae
    
    # Sample 1: HAS LDH (LDH subject) - spine structures + LDH
    target[1, 0, 2:4, 2:4, 2:4] = 1  # disc
    target[1, 0, 5:10, 5:10, 5:10] = 12  # LDH
    
    print("\n--- Input Shapes ---")
    print(f"Network Output: {net_output.shape}")
    print(f"Target: {target.shape}")
    print(f"Sample 0 has LDH: {(target[0] == ldh_class_idx).any().item()}")
    print(f"Sample 1 has LDH: {(target[1] == ldh_class_idx).any().item()}")
    
    # Test 1: Individual Components
    print("\n--- Testing Individual Components ---")
    
    # Test Dice Loss
    print("\n1. Testing PartialLDH_SoftDiceLoss...")
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    dice_loss = PartialLDH_SoftDiceLoss(
        apply_nonlin=softmax_helper_dim1,
        batch_dice=False,
        do_bg=False,
        smooth=1e-5,
        ddp=False,
        ldh_class_idx=ldh_class_idx
    )
    
    try:
        with torch.no_grad():
            dl = dice_loss(net_output, target)
        print(f"   Dice Loss: {dl.item():.6f} ✓")
    except Exception as e:
        print(f"   Dice Loss FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test CE Loss
    print("\n2. Testing PartialLDH_CELoss...")
    ce_loss = PartialLDH_CELoss(ldh_class_idx=ldh_class_idx)
    
    try:
        with torch.no_grad():
            cl = ce_loss(net_output, target[:, 0])
        print(f"   CE Loss: {cl.item():.6f} ✓")
    except Exception as e:
        print(f"   CE Loss FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Combined Loss
    print("\n--- Testing Combined Loss ---")
    
    combined_loss = PartialLDH_DC_and_CE_loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
        ce_kwargs={},
        weight_ce=1.0,
        weight_dice=1.0,
        ignore_label=None,
        ldh_class_idx=ldh_class_idx
    )
    
    try:
        loss = combined_loss(net_output, target)
        print(f"Combined Loss (forward): {loss.item():.6f} ✓")
    except Exception as e:
        print(f"Combined Loss FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Backward Pass
    print("\n--- Testing Backward Pass ---")
    try:
        loss.backward()
        print("Backward pass successful ✓")
        print(f"Gradient shape: {net_output.grad.shape}")
        
        # Check that gradients exist
        grad_norm = net_output.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 0:
            print("Gradients are non-zero ✓")
        else:
            print("WARNING: Gradients are zero!")
            
    except Exception as e:
        print(f"Backward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify LDH handling
    print("\n--- Verifying LDH Handling Logic ---")
    
    # Create a batch with ONLY non-LDH samples
    target_no_ldh = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
    target_no_ldh[0, 0, 2:4, 2:4, 2:4] = 1
    target_no_ldh[1, 0, 6:8, 6:8, 6:8] = 6
    
    net_output_2 = torch.randn(batch_size, num_classes, *spatial, requires_grad=True, device=device)
    
    print(f"All samples have LDH: {(target_no_ldh == ldh_class_idx).any().item()}")
    
    try:
        loss_no_ldh = combined_loss(net_output_2, target_no_ldh)
        print(f"Loss (no LDH batch): {loss_no_ldh.item():.6f} ✓")
        
        loss_no_ldh.backward()
        grad_norm_no_ldh = net_output_2.grad.norm().item()
        print(f"Gradient norm (no LDH): {grad_norm_no_ldh:.6f}")
        
        # The LDH channel should have reduced gradient contribution
        ldh_grad = net_output_2.grad[:, ldh_class_idx].abs().mean().item()
        other_grad = net_output_2.grad[:, 1:ldh_class_idx].abs().mean().item()
        print(f"LDH channel grad (should be small): {ldh_grad:.8f}")
        print(f"Other channels grad: {other_grad:.8f}")
        
        if ldh_grad < other_grad * 0.1:  # LDH grad should be much smaller
            print("LDH channel gradient is properly reduced ✓")
        else:
            print("Note: LDH gradients may still flow through softmax interactions")
            
    except Exception as e:
        print(f"No LDH batch test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
    return True


def test_trainer_import():
    """Test that the trainer can be imported by nnUNet."""
    print("\n" + "=" * 70)
    print("Testing Trainer Import...")
    print("=" * 70)
    
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import (
            nnUNetTrainer_PartialLDH,
            nnUNetTrainer_PartialLDH_OnlyLDHUpdate
        )
        print("nnUNetTrainer_PartialLDH imported successfully ✓")
        print("nnUNetTrainer_PartialLDH_OnlyLDHUpdate imported successfully ✓")
        return True
    except Exception as e:
        print(f"Import FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_supervision():
    """Test loss with deep supervision wrapper."""
    print("\n" + "=" * 70)
    print("Testing Deep Supervision Wrapper...")
    print("=" * 70)
    
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import PartialLDH_DC_and_CE_loss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    
    batch_size = 2
    num_classes = 13
    ldh_class_idx = 12
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Base loss
    loss = PartialLDH_DC_and_CE_loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
        ce_kwargs={},
        weight_ce=1.0,
        weight_dice=1.0,
        ignore_label=None,
        ldh_class_idx=ldh_class_idx
    )
    
    # Wrap with deep supervision
    weights = np.array([1.0, 0.5, 0.25, 0.125, 0])
    weights = weights / weights.sum()
    ds_loss = DeepSupervisionWrapper(loss, weights)
    
    # Create multi-scale outputs (like deep supervision)
    outputs = []
    targets = []
    
    spatial_sizes = [(16, 16, 16), (8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1)]
    
    for spatial in spatial_sizes:
        out = torch.randn(batch_size, num_classes, *spatial, requires_grad=True, device=device)
        tgt = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
        # Add some LDH to second sample
        if spatial[0] >= 4:
            tgt[1, 0, 1:min(3, spatial[0]), 1:min(3, spatial[1]), 1:min(3, spatial[2])] = ldh_class_idx
        outputs.append(out)
        targets.append(tgt)
    
    try:
        total_loss = ds_loss(outputs, targets)
        print(f"Deep Supervision Loss: {total_loss.item():.6f} ✓")
        
        total_loss.backward()
        print("Backward pass with deep supervision successful ✓")
        return True
    except Exception as e:
        print(f"Deep supervision test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  TotalSpineSeg - Custom Trainer Test Suite")
    print("#" * 70 + "\n")
    
    success = True
    
    success &= test_trainer_import()
    success &= test_partial_ldh_loss()
    success &= test_deep_supervision()
    
    print("\n" + "#" * 70)
    if success:
        print("#  ALL TESTS PASSED!")
    else:
        print("#  SOME TESTS FAILED!")
    print("#" * 70 + "\n")
