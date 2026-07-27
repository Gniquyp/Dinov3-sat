"""
DINOv3 + DPT + Mask2Former 模型验证脚本
验证模型可以正常前向传播
"""

import torch
import torch.nn as nn
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_model_with_dummy_backbone():
    """
    使用一个简单的 dummy backbone 测试模型结构
    """
    from dinov3.eval.segmentation.models.dinov3_dpt_mask2former import (
        DINOv3DPTMask2Former,
        DPTMultiScaleFeatureExtractor,
    )

    print("=" * 60)
    print("测试 1: DPTMultiScaleFeatureExtractor")
    print("=" * 60)

    # 创建一个简单的 dummy backbone 模拟 DINOv3 输出
    class DummyBackbone(nn.Module):
        def __init__(self, embed_dim=768, n_blocks=12, patch_size=16):
            super().__init__()
            self.embed_dim = embed_dim
            self.n_blocks = n_blocks
            self.patch_size = patch_size
            self.input_pad_size = patch_size

        def get_intermediate_layers(self, x, n, reshape=True, return_class_token=True, norm=False):
            B, C, H, W = x.shape
            h, w = H // self.patch_size, W // self.patch_size
            outputs = []
            for idx in n:
                # 模拟 patch features [B, C, h, w]
                patch_feat = torch.randn(B, self.embed_dim, h, w)
                # 模拟 class token [B, C]
                cls_token = torch.randn(B, self.embed_dim)
                outputs.append((patch_feat, cls_token))
            return outputs

    # 测试 DPTMultiScaleFeatureExtractor
    dummy_backbone = DummyBackbone(embed_dim=768, n_blocks=12, patch_size=16)

    feature_extractor = DPTMultiScaleFeatureExtractor(
        in_channels=[768, 768, 768, 768],
        channels=256,
        post_process_channels=[128, 256, 512, 1024],
        readout_type="ignore",
        use_batchnorm=False,
    )

    # 模拟输入
    x = torch.randn(2, 3, 512, 512)

    # 模拟 backbone 输出
    backbone_outputs = dummy_backbone.get_intermediate_layers(
        x, n=[2, 5, 8, 11], reshape=True, return_class_token=True
    )

    # DPT 特征提取
    multi_scale_features = feature_extractor(backbone_outputs)

    print(f"输入尺寸: {x.shape}")
    print(f"多尺度特征数量: {len(multi_scale_features)}")
    for i, feat in enumerate(multi_scale_features):
        print(f"  尺度 {i+1}: {feat.shape}")

    print("\n" + "=" * 60)
    print("测试 2: DINOv3DPTMask2Former 完整模型")
    print("=" * 60)

    # 构建完整模型
    model = DINOv3DPTMask2Former(
        backbone=dummy_backbone,
        backbone_out_layers=[2, 5, 8, 11],
        hidden_dim=256,
        num_classes=150,
        use_backbone_norm=False,
        use_batchnorm=False,
        autocast_dtype=torch.float32,
    )

    # 前向传播
    print(f"输入尺寸: {x.shape}")
    with torch.no_grad():
        output = model(x)

    print(f"\n输出键: {list(output.keys())}")
    print(f"pred_logits 尺寸: {output['pred_logits'].shape}")
    print(f"pred_masks 尺寸: {output['pred_masks'].shape}")
    print(f"aux_outputs 数量: {len(output['aux_outputs'])}")

    # 测试 predict 方法
    print("\n测试 predict 方法:")
    with torch.no_grad():
        pred_output = model.predict(x, rescale_to=(512, 512))
    print(f"pred_logits 尺寸: {pred_output['pred_logits'].shape}")
    print(f"pred_masks 尺寸: {pred_output['pred_masks'].shape}")

    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    print("\n所有测试通过!")


if __name__ == "__main__":
    test_model_with_dummy_backbone()
