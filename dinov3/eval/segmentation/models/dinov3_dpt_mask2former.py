# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 + DPT + Mask2Former 完整模型搭建
- DINOv3 作为主干网络提取特征
- DPT 生成多尺度特征
- Mask2Former 使用多尺度特征进行分割
"""

import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from dinov3.eval.depth.models.dpt_head import DPTHead
from dinov3.eval.depth.models.encoder import (
    DinoVisionTransformerWrapper,
    BackboneLayersSet,
    PatchSizeAdaptationStrategy,
)
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead

logger = logging.getLogger("dinov3")


class DPTMultiScaleFeatureExtractor(nn.Module):
    """
    使用 DPT 的前处理模块（ReassembleBlocks + Convs）将 DINOv3 的特征转换为多尺度特征。
    不进行最终的融合和上采样，而是输出多个尺度的特征图供 Mask2Former 使用。
    """

    def __init__(
        self,
        in_channels: List[int],
        channels: int = 256,
        post_process_channels: List[int] = [128, 256, 512, 1024],
        readout_type: str = "ignore",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        from dinov3.eval.depth.models.dpt_head import ReassembleBlocks, ConvModule

        self.in_channels = in_channels
        self.channels = channels

        # ReassembleBlocks: 将 ViT 特征重新组装为不同尺度的特征图
        self.reassemble_blocks = ReassembleBlocks(
            in_channels=in_channels,
            out_channels=post_process_channels,
            readout_type=readout_type,
            use_batchnorm=use_batchnorm,
        )

        # 每个尺度的卷积调整通道数
        self.convs = nn.ModuleList()
        for channel in post_process_channels:
            self.convs.append(
                ConvModule(channel, channels, kernel_size=3, padding=1, act_cfg=None, bias=False)
            )

    def forward(self, inputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Args:
            inputs: DINOv3 输出的列表，每个元素为 (patch_features, cls_token)
                    patch_features: [B, C, H, W]
                    cls_token: [B, C]

        Returns:
            multi_scale_features: 4个尺度的特征图列表，通道数统一为 self.channels
        """
        # ReassembleBlocks 处理：将 ViT 特征转换为 4 个尺度
        # 输出尺度依次为: x4(上采样), x2(上采样), x1(不变), x0.5(下采样)
        x = self.reassemble_blocks(inputs)

        # 每个尺度通过卷积统一通道数
        multi_scale_features = []
        for i, feature in enumerate(x):
            feature = self.convs[i](feature)
            multi_scale_features.append(feature)

        return multi_scale_features


class DINOv3DPTMask2Former(nn.Module):
    """
    DINOv3 + DPT + Mask2Former 完整模型

    流程:
    1. DINOv3 主干网络提取 4 层中间特征
    2. DPT 的 ReassembleBlocks 将特征转换为多尺度特征图
    3. Mask2Former 使用多尺度特征进行分割预测

    Args:
        backbone: DINOv3 ViT 主干网络
        backbone_out_layers: 要提取的中间层索引
        hidden_dim: Mask2Former 的隐藏维度
        num_classes: 分割类别数
        use_backbone_norm: 是否使用 backbone 的 norm
        use_batchnorm: DPT 是否使用 BatchNorm
        autocast_dtype: 自动混合精度的数据类型
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_layers: List[int] = None,
        hidden_dim: int = 256,
        num_classes: int = 150,
        use_backbone_norm: bool = True,
        use_batchnorm: bool = False,
        autocast_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # 1. 构建 DINOv3 特征提取器
        if backbone_out_layers is None:
            # 默认使用 FOUR_EVEN_INTERVALS
            n_blocks = getattr(backbone, "n_blocks", 12)
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

        # 冻结 backbone
        self.encoder.backbone.requires_grad_(False)

        # 获取 backbone 的 embed_dim
        embed_dims = self.encoder.embed_dims

        # 2. 构建 DPT 多尺度特征提取器
        self.dpt_feature_extractor = DPTMultiScaleFeatureExtractor(
            in_channels=embed_dims,
            channels=hidden_dim,
            post_process_channels=[hidden_dim // 2, hidden_dim, hidden_dim * 2, hidden_dim * 4],
            readout_type="ignore",
            use_batchnorm=use_batchnorm,
        )

        # 3. 构建 Mask2Former 头
        patch_size = getattr(backbone, "patch_size", 16)
        # 构建输入形状: [channels, height, width, stride]
        # DPT ReassembleBlocks 输出的 4 个尺度相对 patch_size 的倍数:
        # scale 0: x4 (上采样 4 倍)  -> stride = patch_size / 4
        # scale 1: x2 (上采样 2 倍)  -> stride = patch_size / 2
        # scale 2: x1 (不变)          -> stride = patch_size
        # scale 3: x0.5 (下采样 2 倍) -> stride = patch_size * 2
        input_shape = {
            "1": [hidden_dim, None, None, patch_size // 4],
            "2": [hidden_dim, None, None, patch_size // 2],
            "3": [hidden_dim, None, None, patch_size],
            "4": [hidden_dim, None, None, patch_size * 2],
        }

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
            # 1. DINOv3 提取特征
            # outputs: List of (patch_feats [B, C, h, w], class_token [B, C])
            backbone_outputs = self.encoder(x)

            # 2. DPT 生成多尺度特征
            # multi_scale_features: List of [B, hidden_dim, H_i, W_i]
            multi_scale_features = self.dpt_feature_extractor(backbone_outputs)

            # 3. 转换为 Mask2Former 需要的格式
            features = {
                "1": multi_scale_features[0],
                "2": multi_scale_features[1],
                "3": multi_scale_features[2],
                "4": multi_scale_features[3],
            }

            # 4. Mask2Former 预测
            predictions = self.mask2former_head(features)

        return predictions

    def predict(self, x: torch.Tensor, rescale_to: Tuple[int, int] = (512, 512)) -> Dict[str, torch.Tensor]:
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


def build_dinov3_dpt_mask2former(
    backbone: nn.Module,
    backbone_out_layers: List[int] = None,
    hidden_dim: int = 256,
    num_classes: int = 150,
    use_backbone_norm: bool = True,
    use_batchnorm: bool = False,
    autocast_dtype: torch.dtype = torch.float32,
) -> DINOv3DPTMask2Former:
    """
    构建 DINOv3 + DPT + Mask2Former 模型

    Args:
        backbone: DINOv3 ViT 主干网络
        backbone_out_layers: 要提取的中间层索引
        hidden_dim: Mask2Former 的隐藏维度 (默认 256)
        num_classes: 分割类别数 (默认 150, ADE20K)
        use_backbone_norm: 是否使用 backbone 的 final norm
        use_batchnorm: DPT 是否使用 BatchNorm
        autocast_dtype: 自动混合精度的数据类型

    Returns:
        DINOv3DPTMask2Former 模型实例
    """
    model = DINOv3DPTMask2Former(
        backbone=backbone,
        backbone_out_layers=backbone_out_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        use_backbone_norm=use_backbone_norm,
        use_batchnorm=use_batchnorm,
        autocast_dtype=autocast_dtype,
    )
    model.eval()
    return model
