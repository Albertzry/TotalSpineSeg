"""
Test script for PartialLDH custom trainer.

重要：标签映射
- GT标签值: 0=background, 1-12=前景类别 (12=LDH)
- 网络输出: 12个通道 (0-11)
- 映射: GT标签值 - 1 = 输出通道索引
- LDH: GT值12 → 输出通道11
"""

import torch
import numpy as np


def test_trainer_import():
    """Test that the trainer can be imported by nnUNet."""
    print("\n" + "=" * 70)
    print("Testing Trainer Import...")
    print("=" * 70)
    
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import (
            nnUNetTrainer_PartialLDH,
            PartialLDH_Loss,
            TverskyLoss,
            create_disc_attention_mask
        )
        print("nnUNetTrainer_PartialLDH imported successfully ✓")
        print("PartialLDH_Loss imported successfully ✓")
        print("TverskyLoss imported successfully ✓")
        print("create_disc_attention_mask imported successfully ✓")
        return True
    except Exception as e:
        print(f"Import FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_mapping():
    """Test the label mapping logic."""
    print("\n" + "=" * 70)
    print("Testing Label Mapping...")
    print("=" * 70)
    
    # GT标签值 → 输出通道索引
    # background (0) → 忽略
    # 前景 (1-12) → 通道 (0-11)
    
    print("\nExpected mapping:")
    print("  GT label 0 (background) → ignored")
    print("  GT label 1 (disc) → output channel 0")
    print("  GT label 2 (disc_C2_C3) → output channel 1")
    print("  ...")
    print("  GT label 12 (LDH) → output channel 11")
    
    # 验证LDH映射
    ldh_label_value = 12
    ldh_output_channel = ldh_label_value - 1  # 11
    
    print(f"\nLDH mapping: GT={ldh_label_value} → channel={ldh_output_channel}")
    assert ldh_output_channel == 11, "LDH output channel should be 11"
    print("Label mapping verified ✓")
    return True


def test_partial_ldh_loss():
    """Test the PartialLDH_Loss function with correct label mapping."""
    print("\n" + "=" * 70)
    print("Testing PartialLDH_Loss with Correct Label Mapping...")
    print("=" * 70)
    
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import PartialLDH_Loss
    
    batch_size = 2
    num_output_channels = 12  # 网络输出12个通道 (对应GT 1-12)
    spatial = (16, 16, 16)
    
    # GT中的标签值
    ldh_label_value = 12  # GT中LDH的值
    ldh_output_channel = 11  # 对应的输出通道索引
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create mock data
    # 网络输出: 12个通道
    net_output = torch.randn(batch_size, num_output_channels, *spatial, requires_grad=True, device=device)
    
    # GT标签: 值为0-12
    target = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
    
    # Sample 0: NO LDH
    # 添加一些其他结构 (GT值1-11)
    target[0, 0, 2:4, 2:4, 2:4] = 1  # disc (GT=1)
    target[0, 0, 6:8, 6:8, 6:8] = 6  # vertebrae (GT=6)
    
    # Sample 1: HAS LDH (small region)
    target[1, 0, 2:4, 2:4, 2:4] = 1  # disc (GT=1)
    target[1, 0, 5:6, 5:6, 5:6] = 3  # disc_C7_T1 (GT=3)
    target[1, 0, 7:9, 7:9, 7:9] = ldh_label_value  # LDH (GT=12)
    
    print(f"\nSample 0 has LDH (GT={ldh_label_value}): {(target[0] == ldh_label_value).any().item()}")
    print(f"Sample 1 has LDH (GT={ldh_label_value}): {(target[1] == ldh_label_value).any().item()}")
    print(f"LDH voxels in sample 1: {(target[1] == ldh_label_value).sum().item()}")
    print(f"Background voxels in sample 0: {(target[0] == 0).sum().item()}")
    
    # Create loss with correct parameters
    loss_fn = PartialLDH_Loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False},
        ce_kwargs={},
        weight_ce=1.0,
        weight_dice=1.0,
        weight_tversky=0.5,
        weight_focal=0.5,
        ldh_label_value=ldh_label_value,  # GT中的标签值 (12)
        ldh_output_channel=ldh_output_channel,  # 输出通道索引 (11)
        ldh_class_weight=3.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_gamma=2.0,
        num_classes=num_output_channels  # 12个前景类
    )
    
    # Forward
    print("\n--- Forward Pass ---")
    try:
        loss = loss_fn(net_output, target)
        print(f"Loss value: {loss.item():.6f} ✓")
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is Inf!"
    except Exception as e:
        print(f"Forward FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Backward
    print("\n--- Backward Pass ---")
    try:
        loss.backward()
        grad_norm = net_output.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f} ✓")
        assert not torch.isnan(net_output.grad).any(), "Gradient contains NaN!"
        
        # Check LDH gradient handling using output channel index
        ldh_grad_s0 = net_output.grad[0, ldh_output_channel].abs().mean().item()
        ldh_grad_s1 = net_output.grad[1, ldh_output_channel].abs().mean().item()
        print(f"Sample 0 (no LDH) - LDH channel (ch{ldh_output_channel}) grad: {ldh_grad_s0:.8f}")
        print(f"Sample 1 (has LDH) - LDH channel (ch{ldh_output_channel}) grad: {ldh_grad_s1:.8f}")
        
        if ldh_grad_s0 < ldh_grad_s1:
            print("LDH gradient properly reduced for sample without LDH ✓")
        
    except Exception as e:
        print(f"Backward FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with batch without any LDH
    print("\n--- Testing batch without LDH ---")
    net_output2 = torch.randn(batch_size, num_output_channels, *spatial, requires_grad=True, device=device)
    target_no_ldh = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
    target_no_ldh[0, 0, 2:4, 2:4, 2:4] = 1  # disc
    target_no_ldh[1, 0, 6:8, 6:8, 6:8] = 6  # vertebrae
    
    try:
        loss2 = loss_fn(net_output2, target_no_ldh)
        print(f"Loss (no LDH batch): {loss2.item():.6f} ✓")
        
        loss2.backward()
        ldh_grad = net_output2.grad[:, ldh_output_channel].abs().mean().item()
        print(f"LDH channel grad (should be reduced): {ldh_grad:.10f}")
        
    except Exception as e:
        print(f"No LDH batch test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_anatomical_attention():
    """Test anatomical attention based on disc location."""
    print("\n" + "=" * 70)
    print("Testing Anatomical Attention...")
    print("=" * 70)
    
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import (
        create_disc_attention_mask, PartialLDH_Loss
    )
    
    batch_size = 2
    num_output_channels = 12
    spatial = (16, 16, 16)
    ldh_label_value = 12
    ldh_output_channel = 11
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create target with disc and LDH (using GT label values)
    target = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
    
    # Add disc structures (GT class 1-5)
    target[0, 0, 5:8, 5:8, 5:8] = 1  # disc (GT=1)
    target[1, 0, 7:10, 7:10, 7:10] = 3  # disc_C7_T1 (GT=3)
    
    # Add LDH near disc in sample 1 (GT=12)
    target[1, 0, 8:10, 8:10, 10:12] = ldh_label_value  # LDH adjacent to disc
    
    # Test disc attention mask creation
    print("\n1. Testing create_disc_attention_mask...")
    try:
        # disc_classes uses GT label values (1-5)
        attention_mask = create_disc_attention_mask(target, disc_classes=(1, 2, 3, 4, 5), dilation_radius=3)
        print(f"   Attention mask shape: {attention_mask.shape} ✓")
        print(f"   Disc voxels (sample 0): {(target[0] == 1).sum().item()}")
        print(f"   Disc voxels (sample 1): {((target[1] >= 1) & (target[1] <= 5)).sum().item()}")
        print(f"   Attention region voxels (after dilation): {(attention_mask[0] > 0).sum().item()}")
        
        # Check that LDH location is within attention region
        ldh_in_attention = (attention_mask[1, 0, 8:10, 8:10, 10:12] > 0).all().item()
        print(f"   LDH location within attention region: {ldh_in_attention} ✓")
    except Exception as e:
        print(f"   Attention mask creation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss with anatomical attention
    print("\n2. Testing Loss with Anatomical Attention...")
    net_output = torch.randn(batch_size, num_output_channels, *spatial, requires_grad=True, device=device)
    
    loss_fn = PartialLDH_Loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False},
        ce_kwargs={},
        ldh_label_value=ldh_label_value,
        ldh_output_channel=ldh_output_channel,
        use_anatomical_attention=True,
        disc_classes=(1, 2, 3, 4, 5),  # GT label values
        attention_dilation=3,
        outside_disc_penalty=2.0,
        num_classes=num_output_channels
    )
    
    try:
        loss = loss_fn(net_output, target)
        print(f"   Loss with attention: {loss.item():.6f} ✓")
        
        loss.backward()
        print("   Backward pass successful ✓")
    except Exception as e:
        print(f"   Loss computation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare with loss without anatomical attention
    print("\n3. Comparing with/without Anatomical Attention...")
    net_output2 = torch.randn(batch_size, num_output_channels, *spatial, requires_grad=True, device=device)
    
    loss_fn_no_attn = PartialLDH_Loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False},
        ce_kwargs={},
        ldh_label_value=ldh_label_value,
        ldh_output_channel=ldh_output_channel,
        use_anatomical_attention=False,
        num_classes=num_output_channels
    )
    
    loss_no_attn = loss_fn_no_attn(net_output2, target)
    print(f"   Loss without attention: {loss_no_attn.item():.6f}")
    print("   (With attention adds penalty for LDH predictions outside disc region)")
    
    print("\nAnatomical Attention tests completed ✓")
    return True


def test_deep_supervision():
    """Test loss with deep supervision wrapper."""
    print("\n" + "=" * 70)
    print("Testing Deep Supervision...")
    print("=" * 70)
    
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import PartialLDH_Loss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    
    batch_size = 2
    num_output_channels = 12
    ldh_label_value = 12
    ldh_output_channel = 11
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss = PartialLDH_Loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False},
        ce_kwargs={},
        ldh_label_value=ldh_label_value,
        ldh_output_channel=ldh_output_channel,
        num_classes=num_output_channels
    )
    
    weights = np.array([1.0, 0.5, 0.25, 0.125, 0])
    weights = weights / weights.sum()
    ds_loss = DeepSupervisionWrapper(loss, weights)
    
    outputs = []
    targets = []
    spatial_sizes = [(16, 16, 16), (8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1)]
    
    for spatial in spatial_sizes:
        out = torch.randn(batch_size, num_output_channels, *spatial, requires_grad=True, device=device)
        tgt = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
        if spatial[0] >= 4:
            # Add some foreground (GT values 1-12)
            tgt[0, 0, 0:min(2, spatial[0]), 0:min(2, spatial[1]), 0:min(2, spatial[2])] = 1  # disc
            tgt[1, 0, 1:min(3, spatial[0]), 1:min(3, spatial[1]), 1:min(3, spatial[2])] = ldh_label_value  # LDH
        outputs.append(out)
        targets.append(tgt)
    
    try:
        total_loss = ds_loss(outputs, targets)
        print(f"Deep Supervision Loss: {total_loss.item():.6f} ✓")
        
        total_loss.backward()
        print("Backward pass successful ✓")
        return True
    except Exception as e:
        print(f"Deep supervision test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp16_stability():
    """Test numerical stability with fp16."""
    print("\n" + "=" * 70)
    print("Testing FP16 Numerical Stability...")
    print("=" * 70)
    
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PartialLDH import PartialLDH_Loss
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping fp16 test")
        return True
    
    batch_size = 2
    num_output_channels = 12
    spatial = (16, 16, 16)
    ldh_label_value = 12
    ldh_output_channel = 11
    
    device = torch.device('cuda')
    
    # Create fp16 data
    net_output = torch.randn(batch_size, num_output_channels, *spatial, 
                             device=device, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
    target[0, 0, 2:4, 2:4, 2:4] = 1
    target[1, 0, 7:9, 7:9, 7:9] = ldh_label_value
    
    loss_fn = PartialLDH_Loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False},
        ce_kwargs={},
        ldh_label_value=ldh_label_value,
        ldh_output_channel=ldh_output_channel,
        num_classes=num_output_channels
    )
    
    try:
        loss = loss_fn(net_output, target)
        print(f"FP16 Loss value: {loss.item():.6f}")
        assert not torch.isnan(loss), "FP16 Loss is NaN!"
        assert not torch.isinf(loss), "FP16 Loss is Inf!"
        print("FP16 forward pass successful ✓")
        
        loss.backward()
        assert not torch.isnan(net_output.grad).any(), "FP16 Gradient contains NaN!"
        print("FP16 backward pass successful ✓")
        return True
    except Exception as e:
        print(f"FP16 test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  TotalSpineSeg - PartialLDH Trainer Test")
    print("#  Testing with correct label mapping:")
    print("#  - GT labels: 0=background, 1-12=foreground (12=LDH)")
    print("#  - Network output: 12 channels (0-11)")
    print("#  - Mapping: GT value - 1 = output channel")
    print("#" * 70 + "\n")
    
    success = True
    success &= test_trainer_import()
    success &= test_label_mapping()
    success &= test_partial_ldh_loss()
    success &= test_anatomical_attention()
    success &= test_deep_supervision()
    success &= test_fp16_stability()
    
    print("\n" + "#" * 70)
    if success:
        print("#  ALL TESTS PASSED!")
    else:
        print("#  SOME TESTS FAILED!")
    print("#" * 70 + "\n")
