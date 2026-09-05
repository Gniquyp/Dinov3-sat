"""
DINOv3 + ViT-CoMer + Mask2Former 模型验证脚本
验证模型可以正常前向传播、多尺度特征尺寸正确、CTI 交互的梯度可回传。
"""

import torch
import torch.nn as nn
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class DummyBlock(nn.Module):
    """模拟 DINOv3 block: identity, 但签名与真实 block 一致 (x, rope=None)。"""

    def forward(self, x, rope=None):
        return x


class DummyBackbone(nn.Module):
    """模拟 DINOv3 主干接口:
    - token 布局 [cls(1), storage(n_storage_tokens), patches];
    - get_intermediate_layers 内部按 blk(x, rope) 方式顺序运行全部 blocks
      (CoMer 的 CTI 交互 hook 挂在 blocks 上, 与真实 DINOv3 触发路径一致)。
    """

    def __init__(self, embed_dim=384, n_blocks=12, patch_size=16, n_storage_tokens=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_dims = [embed_dim] * n_blocks
        self.n_blocks = n_blocks
        self.patch_size = patch_size
        self.input_pad_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.blocks = nn.ModuleList([DummyBlock() for _ in range(n_blocks)])

    def get_intermediate_layers(self, x, n, reshape=True,
                                return_class_token=True, norm=False):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        prefix = 1 + self.n_storage_tokens

        tokens = torch.randn(B, prefix + h * w, self.embed_dim)
        outputs = []
        for i, blk in enumerate(self.blocks):
            # 与真实 DINOv3 一致: 位置参数传入 rope (此处为 None)
            tokens = blk(tokens, None)
            if i in n:
                outputs.append(tokens)

        result = []
        for out in outputs:
            patch_feat = out[:, prefix:].reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
            cls_token = out[:, 0]
            result.append((patch_feat, cls_token))
        return result


def test_model_with_dummy_backbone():
    from dinov3.eval.segmentation.models.dinov3_comer_mask2former import (
        DINOv3CoMerMask2Former,
        CoMerMultiScaleFeatureExtractor,
        build_dinov3_comer_mask2former,
    )

    torch.manual_seed(0)

    print("=" * 60)
    print("测试 1: CoMer 多尺度特征提取器 (stride 4/8/16/32)")
    print("=" * 60)

    dummy_backbone = DummyBackbone(embed_dim=384, n_blocks=12, patch_size=16, n_storage_tokens=1)

    model = DINOv3CoMerMask2Former(
        backbone=dummy_backbone,
        backbone_out_layers=[2, 5, 8, 11],
        hidden_dim=256,
        num_classes=8,
        use_backbone_norm=False,
        freeze_backbone=True,
        autocast_dtype=torch.float32,
    )

    # 12 blocks 默认均分 4 stage: [[0,2],[3,5],[6,8],[9,11]]
    print(f"CTI stage 划分: {model.comer.interaction_indexes}")
    assert model.comer.interaction_indexes == [[0, 2], [3, 5], [6, 8], [9, 11]]

    # 直接检查 finalize 输出的 4 个尺度 (256x256 输入)
    x = torch.randn(1, 3, 256, 256)
    x_pad = model.input_pad(x)
    model.comer.begin(x_pad)
    _ = model.encoder(x_pad)
    feats = model.comer.finalize()

    expected_shapes = [
        (1, 256, 64, 64),   # stride 4
        (1, 256, 32, 32),   # stride 8
        (1, 256, 16, 16),   # stride 16
        (1, 256, 8, 8),     # stride 32
    ]
    print(f"输入尺寸: {tuple(x.shape)}")
    for i, (feat, exp) in enumerate(zip(feats, expected_shapes), start=1):
        print(f"  尺度 {i}: {tuple(feat.shape)} (期望 {exp})")
        assert tuple(feat.shape) == exp, f"尺度 {i} shape mismatch: {feat.shape} vs {exp}"

    print("\n" + "=" * 60)
    print("测试 2: 完整模型前向 (含非 32 倍数尺寸的 padding)")
    print("=" * 60)

    # 240x240 会被 pad 到 256x256, 验证 CenterPadding(32) 路径
    x = torch.randn(2, 3, 240, 240)
    print(f"输入尺寸: {tuple(x.shape)} (pad 到 256x256)")
    with torch.no_grad():
        output = model(x)

    print(f"输出键: {list(output.keys())}")
    print(f"pred_logits 尺寸: {tuple(output['pred_logits'].shape)}")
    print(f"pred_masks 尺寸: {tuple(output['pred_masks'].shape)}")
    print(f"aux_outputs 数量: {len(output['aux_outputs'])}")
    assert output["pred_logits"].shape[0] == 2
    assert output["pred_logits"].shape[-1] == 8 + 1  # 8 类 + no-object
    assert output["pred_masks"].shape[0] == 2

    # 测试 predict 方法
    print("\n测试 predict 方法:")
    with torch.no_grad():
        pred_output = model.predict(x, rescale_to=(240, 240))
    print(f"pred_logits 尺寸: {tuple(pred_output['pred_logits'].shape)}")
    print(f"pred_masks 尺寸: {tuple(pred_output['pred_masks'].shape)}")
    assert tuple(pred_output["pred_masks"].shape[-2:]) == (240, 240)

    print("\n" + "=" * 60)
    print("测试 3: CTI 交互梯度回传 (backward)")
    print("=" * 60)

    # CPU 环境没有编译 MSDA CUDA 扩展 (其 backward 会直接报错);
    # 这里用可微的纯 PyTorch grid_sample 实现替换 custom Function,
    # 仅用于在 CPU 上验证整条计算图 (含 CTI 与 Mask2Former pixel decoder) 梯度连通。
    # GPU 训练时仍走编译扩展 (与 DPT 版 Mask2Former 的要求一致)。
    from dinov3.eval.segmentation.models.utils import ms_deform_attn as _msda_mod

    def _pytorch_msda_apply(value, spatial_shapes, level_start_index,
                            sampling_locations, attention_weights, im2col_step):
        return _msda_mod.ms_deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

    _msda_mod.MSDeformAttnFunction.apply = staticmethod(_pytorch_msda_apply)

    model.train()
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    loss = output["pred_logits"].mean() + output["pred_masks"].mean()
    for aux in output["aux_outputs"]:
        loss = loss + aux["pred_logits"].mean() + aux["pred_masks"].mean()
    loss.backward()

    # CoMer 新增模块的参数必须有梯度
    grad_ok = 0
    for name, p in model.comer.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grad_ok += 1
    print(f"CoMer 模块收到非零梯度的参数张量数: {grad_ok}")
    assert grad_ok > 0, "CoMer 参数没有收到梯度, CTI 交互可能未接入计算图"

    # Mask2Former 头参数也必须有梯度
    head_grad = sum(
        1 for p in model.mask2former_head.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"Mask2Former 头收到非零梯度的参数张量数: {head_grad}")
    assert head_grad > 0

    # 冻结的 backbone 不应有梯度
    backbone_grad = [n for n, p in model.named_parameters()
                     if n.startswith("encoder.backbone") and p.grad is not None]
    assert len(backbone_grad) == 0, "backbone 已冻结但收到了梯度"
    print("backbone 已正确冻结 (无梯度)")

    print("\n" + "=" * 60)
    print("测试 4: builder 接口与参数统计")
    print("=" * 60)

    model2 = build_dinov3_comer_mask2former(
        backbone=DummyBackbone(embed_dim=384, n_blocks=12),
        hidden_dim=256,
        num_classes=8,
        freeze_backbone=True,
    )
    assert isinstance(model2, DINOv3CoMerMask2Former)

    total_params = sum(p.numel() for p in model2.parameters())
    trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    comer_params = sum(p.numel() for p in model2.comer.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  其中 CoMer (SPM+MRFP+CTI) 参数量: {comer_params:,}")

    print("\n所有测试通过!")


if __name__ == "__main__":
    test_model_with_dummy_backbone()
