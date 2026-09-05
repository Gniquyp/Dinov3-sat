# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 + ViT-CoMer + Mask2Former 完整模型搭建

- DINOv3 作为主干网络提取特征 (权重和前向计算完全不变)
- ViT-CoMer (CVPR 2024 Highlight, arXiv:2403.07392) 替换原 DPT 作为
  多尺度特征桥接:
    * SPM (CNN 分支) 产生 stride 4/8/16/32 的多尺度特征;
    * MRFP (多感受野金字塔) 增强 CNN 特征;
    * CTI (CNN-Transformer Interaction) 通过多尺度可变形注意力
      在每个 ViT stage 边界做双向特征交互。
  交互通过 forward hook 在 DINOv3 block 边界触发, 不修改主干代码。
- Mask2Former 使用多尺度特征进行分割 (与 DPT 版完全一致)。
"""

import logging
import math
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_

from dinov3.eval.depth.models.embed import CenterPadding
from dinov3.eval.depth.models.encoder import (
    DinoVisionTransformerWrapper,
    PatchSizeAdaptationStrategy,
)
from dinov3.eval.segmentation.models.comer_modules import (
    CNN,
    CTIBlock,
    deform_inputs_3levels,
)
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead
from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn

logger = logging.getLogger("dinov3")


def _trunc_normal_(tensor, std=0.02):
    fn = getattr(torch.nn.init, "trunc_normal_", None)
    if fn is not None:
        fn(tensor, std=std)
    else:
        nn.init.normal_(tensor, std=std)


class CoMerMultiScaleFeatureExtractor(nn.Module):
    """ViT-CoMer 多尺度特征提取器 (SPM + MRFP + CTI)。

    CNN 分支 (SPM) 独立提取多尺度特征, 同时通过 forward hook 在 DINOv3
    每个 ViT stage 的 block 边界插入 CTI 双向交互:

      - stage 第一个 block 前 (pre_stage): MRFP 增强 CNN 特征 -> ViT 特征
        注入 CNN 中间尺度 -> CNN 多尺度特征经跨尺度交互后注入 ViT;
      - stage 最后一个 block 后 (post_stage): ViT stage 输出注入 CNN 中间
        尺度 -> CNN 跨尺度自交互 (最后一个 stage 额外做 4 次)。

    hook 只读写 token, DINOv3 主干的权重与计算保持原样。
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        hidden_dim: int,
        interaction_indexes: List[List[int]],
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: Optional[int] = None,
        init_values: float = 1e-6,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        dim_ratio: float = 6.0,
        add_vit_feature: bool = True,
        use_extra_CTI: bool = True,
        use_CTI_toV: bool = True,
        use_CTI_toC: bool = True,
        cnn_feature_interaction: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.patch_size = getattr(backbone, "patch_size", 16)
        self.prefix_len = 1 + getattr(backbone, "n_storage_tokens", 0)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature

        if deform_num_heads is None:
            deform_num_heads = max(1, embed_dim // 64)

        # SPM: CNN 多尺度分支
        self.spm = CNN(inplanes=conv_inplane, embed_dim=embed_dim)
        # CNN 三个 flatten 尺度 (stride 8/16/32) 的 level embedding
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        # 每个 ViT stage 一个 CTI 交互块
        self.interactions = nn.Sequential(
            *[
                CTIBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=True,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    use_CTI_toV=use_CTI_toV,
                    use_CTI_toC=use_CTI_toC,
                    dim_ratio=dim_ratio,
                    cnn_feature_interaction=cnn_feature_interaction,
                    extra_CTI=((i == len(interaction_indexes) - 1) and use_extra_CTI),
                )
                for i in range(len(interaction_indexes))
            ]
        )

        # stride 8 -> stride 4 的上采样 (与官方一致)
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        # 四个输出尺度的 BN
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        # 投影到 Mask2Former 的 hidden_dim
        self.proj = nn.ModuleList(
            [nn.Conv2d(embed_dim, hidden_dim, kernel_size=1) for _ in range(4)]
        )

        # 权重初始化 (与官方 ViTCoMer 一致)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        # 在 DINOv3 block 上注册交互 hook (注册到 block 模块, 不重复注册参数)
        for i, (start, end) in enumerate(interaction_indexes):
            backbone.blocks[start].register_forward_pre_hook(self._make_pre_hook(i))
            backbone.blocks[end].register_forward_hook(self._make_post_hook(i))
        self._reset_state()

    # ------------------------------------------------------------------ init
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _reset_state(self):
        self._state = {"active": False}

    # ----------------------------------------------------------------- hooks
    def _make_pre_hook(self, stage_idx: int):
        def hook(module, args):
            state = self._state
            if not state["active"]:
                return None
            x = args[0]
            H, W = state["H"], state["W"]
            patch_tokens = x[:, self.prefix_len :]
            new_patch, new_c = self.interactions[stage_idx].pre_stage(
                patch_tokens, state["c"], state["di3"], H, W
            )
            state["c"] = new_c
            x_new = torch.cat([x[:, : self.prefix_len], new_patch], dim=1)
            # DINOv3 block 以 (x, rope) 位置参数调用, 保持 rope 不变
            if len(args) > 1:
                return (x_new, args[1])
            return (x_new,)

        return hook

    def _make_post_hook(self, stage_idx: int):
        def hook(module, args, output):
            state = self._state
            if not state["active"]:
                return None
            H, W = state["H"], state["W"]
            patch_out = output[:, self.prefix_len :]
            state["c"] = self.interactions[stage_idx].post_stage(
                state["c"], patch_out, state["di3"], H, W
            )
            state["outs"][stage_idx] = patch_out
            return None

        return hook

    # --------------------------------------------------------------- forward
    def begin(self, x: torch.Tensor):
        """在 DINOv3 主干前向前调用: 输入需已 pad 到 32 的倍数。

        运行 SPM 得到 CNN 多尺度特征, 并初始化交互状态。
        """
        _, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(f"CoMer 输入尺寸需为 32 的倍数, 得到 {h}x{w}")

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        c = torch.cat([c2, c3, c4], dim=1)

        self._state = {
            "active": True,
            "c": c,
            "c1": c1,
            "l2": c2.shape[1],
            "l3": c3.shape[1],
            "l4": c4.shape[1],
            "H": H,
            "W": W,
            "di3": deform_inputs_3levels(H, W, x.device),
            "outs": [None] * len(self.interaction_indexes),
        }

    def finalize(self) -> List[torch.Tensor]:
        """主干前向后调用: 按官方 ViTCoMer 方式装配 4 个尺度的输出特征。

        Returns:
            [f1, f2, f3, f4]: stride 4/8/16/32, 通道均为 hidden_dim。
        """
        state = self._state
        if not state["active"]:
            raise RuntimeError("请先调用 begin() 再运行主干, 最后调用 finalize()")
        H, W = state["H"], state["W"]
        c, c1 = state["c"], state["c1"]
        l2, l3, l4 = state["l2"], state["l3"], state["l4"]
        B = c.shape[0]
        D = self.embed_dim

        # Split & Reshape
        c2 = c[:, 0:l2, :]
        c3 = c[:, l2 : l2 + l3, :]
        c4 = c[:, l2 + l3 :, :]
        c2 = c2.transpose(1, 2).reshape(B, D, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).reshape(B, D, H, W).contiguous()
        c4 = c4.transpose(1, 2).reshape(B, D, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = state["outs"]
            x1 = x1.transpose(1, 2).reshape(B, D, H, W).contiguous()
            x2 = x2.transpose(1, 2).reshape(B, D, H, W).contiguous()
            x3 = x3.transpose(1, 2).reshape(B, D, H, W).contiguous()
            x4 = x4.transpose(1, 2).reshape(B, D, H, W).contiguous()
            x1 = F.interpolate(x1, scale_factor=4, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm + 通道投影
        f1 = self.proj[0](self.norm1(c1))
        f2 = self.proj[1](self.norm2(c2))
        f3 = self.proj[2](self.norm3(c3))
        f4 = self.proj[3](self.norm4(c4))

        self._reset_state()
        return [f1, f2, f3, f4]


class DINOv3CoMerMask2Former(nn.Module):
    """
    DINOv3 + ViT-CoMer + Mask2Former 完整模型

    流程:
    1. 输入 pad 到 32 的倍数 (CoMer SPM 下采样 32x);
    2. SPM 提取 CNN 多尺度特征;
    3. DINOv3 主干前向, CTI 在每个 stage 边界做双向交互 (hook);
    4. 装配 stride 4/8/16/32 四个尺度的特征;
    5. Mask2Former 使用多尺度特征进行分割预测。
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_layers: List[int] = None,
        hidden_dim: int = 256,
        num_classes: int = 150,
        use_backbone_norm: bool = True,
        freeze_backbone: bool = True,
        autocast_dtype: torch.dtype = torch.float32,
        # ViT-CoMer 超参 (默认值为官方 CoMer-L 配置)
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: Optional[int] = None,
        interaction_indexes: Optional[List[List[int]]] = None,
        init_values: float = 1e-6,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        dim_ratio: float = 6.0,
        add_vit_feature: bool = True,
        use_extra_CTI: bool = True,
    ):
        super().__init__()

        # 1. 构建 DINOv3 特征提取器 (与 DPT 版相同的 wrapper)
        n_blocks = getattr(backbone, "n_blocks", 12)
        if backbone_out_layers is None:
            # 默认 FOUR_EVEN_INTERVALS
            if n_blocks == 24:
                backbone_out_layers = [4, 11, 17, 23]
            elif n_blocks == 40:
                backbone_out_layers = [9, 19, 29, 39]
            else:
                backbone_out_layers = [i * (n_blocks // 4) - 1 for i in range(1, 5)]

        self.encoder = DinoVisionTransformerWrapper(
            backbone_model=backbone,
            backbone_out_layers=backbone_out_layers,
            use_backbone_norm=use_backbone_norm,
            adapt_to_patch_size=PatchSizeAdaptationStrategy.CENTER_PADDING,
        )

        # 冻结/解冻 backbone (参数名前缀 encoder.backbone 与 DPT 版一致,
        # 训练脚本的优化器参数分组无需改动)
        self.encoder.backbone.requires_grad_(not freeze_backbone)

        embed_dim = self.encoder.embed_dims[0]

        # CTI stage 划分: 默认将 blocks 均分为 4 个 stage
        # 24 blocks (ViT-L): [[0,5],[6,11],[12,17],[18,23]] (官方 CoMer-L)
        if interaction_indexes is None:
            stage = n_blocks // 4
            interaction_indexes = [[i * stage, (i + 1) * stage - 1] for i in range(4)]

        # 2. 构建 ViT-CoMer 多尺度特征提取器
        self.comer = CoMerMultiScaleFeatureExtractor(
            backbone=self.encoder.backbone,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            interaction_indexes=interaction_indexes,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            init_values=init_values,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            dim_ratio=dim_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_CTI=use_extra_CTI,
        )

        # 输入 pad 到 32 的倍数 (SPM 需要; wrapper 内部还有 16 的 pad, 此处之后为 no-op)
        self.input_pad = CenterPadding(32)

        # 3. 构建 Mask2Former 头 (与 DPT 版完全一致)
        patch_size = getattr(backbone, "patch_size", 16)
        # CoMer 四个输出尺度相对输入图的 stride 为 4/8/16/32
        input_shape = {
            "1": [hidden_dim, None, None, 4],
            "2": [hidden_dim, None, None, 8],
            "3": [hidden_dim, None, None, 16],
            "4": [hidden_dim, None, None, 32],
        }
        _ = patch_size  # stride 由 CoMer 结构决定 (patch_size=16 时与 DPT 版相同)

        self.mask2former_head = Mask2FormerHead(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            ignore_value=255,
        )

        # autocast 上下文
        if torch.cuda.is_available():
            self.autocast_ctx = partial(
                torch.autocast, device_type="cuda", dtype=autocast_dtype, enabled=True
            )
        else:
            self.autocast_ctx = partial(torch.autocast, device_type="cpu", enabled=False)

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            predictions: 包含 pred_logits 和 pred_masks 的字典
        """
        with self.autocast_ctx():
            # 1. pad 到 32 的倍数
            x_pad = self.input_pad(x)

            # 2. SPM 提取 CNN 多尺度特征 + 初始化 CTI 交互状态
            self.comer.begin(x_pad)

            # 3. DINOv3 主干前向 (CTI 交互在 hook 中于 stage 边界触发)
            _ = self.encoder(x_pad)

            # 4. 装配 4 个尺度的多尺度特征 (stride 4/8/16/32, hidden_dim 通道)
            multi_scale_features = self.comer.finalize()

            # 5. 转换为 Mask2Former 需要的格式
            features = {
                "1": multi_scale_features[0],
                "2": multi_scale_features[1],
                "3": multi_scale_features[2],
                "4": multi_scale_features[3],
            }

            # 6. Mask2Former 预测
            predictions = self.mask2former_head(features)

        return predictions

    def predict(
        self, x: torch.Tensor, rescale_to: Tuple[int, int] = (512, 512)
    ) -> Dict[str, torch.Tensor]:
        """
        推理接口

        Args:
            x: 输入图像 [B, 3, H, W]
            rescale_to: 输出 mask 的大小

        Returns:
            predictions: 包含 pred_logits 和 pred_masks 的字典
        """
        with torch.inference_mode():
            output = self.forward(x)
            output["pred_masks"] = F.interpolate(
                output["pred_masks"],
                size=rescale_to,
                mode="bilinear",
                align_corners=False,
            )
        return output


def build_dinov3_comer_mask2former(
    backbone: nn.Module,
    backbone_out_layers: List[int] = None,
    hidden_dim: int = 256,
    num_classes: int = 150,
    use_backbone_norm: bool = True,
    freeze_backbone: bool = True,
    autocast_dtype: torch.dtype = torch.float32,
    conv_inplane: int = 64,
    n_points: int = 4,
    deform_num_heads: Optional[int] = None,
    init_values: float = 1e-6,
    cffn_ratio: float = 0.25,
    deform_ratio: float = 0.5,
    dim_ratio: float = 6.0,
    add_vit_feature: bool = True,
    use_extra_CTI: bool = True,
) -> DINOv3CoMerMask2Former:
    """
    构建 DINOv3 + ViT-CoMer + Mask2Former 模型

    Args:
        backbone: DINOv3 ViT 主干网络
        backbone_out_layers: 要提取的中间层索引
        hidden_dim: Mask2Former 的隐藏维度 (默认 256)
        num_classes: 分割类别数 (默认 150, ADE20K)
        use_backbone_norm: 是否使用 backbone 的 final norm
        freeze_backbone: 是否冻结主干
        autocast_dtype: 自动混合精度的数据类型
        conv_inplane: SPM CNN 分支基础通道数 (官方 CoMer-L: 64)
        n_points: 可变形注意力每尺度采样点数 (官方: 4)
        deform_num_heads: 可变形注意力头数 (默认 embed_dim // 64)
        init_values: CTI_toV 注入的可学习缩放初始值 (官方: 1e-6)
        cffn_ratio: ConvFFN 中间层比例 (官方: 0.25)
        deform_ratio: 可变形注意力采样偏移比例 (官方: 0.5)
        dim_ratio: MRFP 隐藏层比例 (官方: 6.0)
        add_vit_feature: 最终多尺度特征是否加回 ViT 特征 (官方: True)
        use_extra_CTI: 最后一个 stage 是否做 4 次额外 CNN 交互 (官方: True)

    Returns:
        DINOv3CoMerMask2Former 模型实例
    """
    model = DINOv3CoMerMask2Former(
        backbone=backbone,
        backbone_out_layers=backbone_out_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        use_backbone_norm=use_backbone_norm,
        freeze_backbone=freeze_backbone,
        autocast_dtype=autocast_dtype,
        conv_inplane=conv_inplane,
        n_points=n_points,
        deform_num_heads=deform_num_heads,
        init_values=init_values,
        cffn_ratio=cffn_ratio,
        deform_ratio=deform_ratio,
        dim_ratio=dim_ratio,
        add_vit_feature=add_vit_feature,
        use_extra_CTI=use_extra_CTI,
    )
    model.eval()
    return model
