# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ViT-CoMer 模块实现, 移植自官方代码 (CVPR 2024 Highlight):
# https://github.com/Traffic-X/ViT-CoMer  (segmentation/mmseg_custom/models/backbones/comer_modules.py)
#
# 相对官方实现的适配改动:
# 1. SyncBatchNorm -> BatchNorm2d (单卡/CPU 均可运行);
# 2. 多尺度可变形注意力复用仓库已有的 MSDeformAttn (纯 PyTorch 前向 / CUDA 反向);
# 3. 不依赖 timm, 自带 DropPath;
# 4. CTIBlock 不再内部运行 ViT blocks, 而是拆成 pre_stage / post_stage 两个阶段,
#    由 dinov3_comer_mask2former.py 通过 forward hook 在 DINOv3 block 边界处调用,
#    从而保持 DINOv3 主干的权重和前向完全不变。

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn

_logger = logging.getLogger("dinov3")


class DropPath(nn.Module):
    """timm 风格的 DropPath (stochastic depth), 训练时按样本丢弃。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for H_, W_ in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs_3levels(H, W, device):
    """CNN 特征金字塔 3 个尺度 (相对输入图 stride 8/16/32) 的可变形注意力输入。

    Args:
        H, W: ViT patch 网格尺寸 (输入图的 1/16)。则 CNN 三级特征网格为
              (2H, 2W) / (H, W) / (H//2, W//2)。

    Returns:
        [reference_points, spatial_shapes, level_start_index]
    """
    spatial_shapes = torch.as_tensor(
        [(2 * H, 2 * W), (H, W), (H // 2, W // 2)],
        dtype=torch.long,
        device=device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(2 * H, 2 * W), (H, W), (H // 2, W // 2)], device
    )
    return [reference_points, spatial_shapes, level_start_index]


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class MultiDWConv(nn.Module):
    """MRFP 中的多感受野深度可分离卷积: 每个尺度两个分支, 分别用 3x3 / 5x5 深度卷积。"""

    def __init__(self, dim=768):
        super().__init__()
        dim1 = dim
        dim = dim // 2
        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv4 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.dwconv5 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv6 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim1)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim1)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(dim1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()

        x11, x12 = x1[:, : C // 2, :, :], x1[:, C // 2 :, :, :]
        x11 = self.dwconv1(x11)
        x12 = self.dwconv2(x12)
        x1 = torch.cat([x11, x12], dim=1)
        x1 = self.act1(self.bn1(x1)).flatten(2).transpose(1, 2)

        x21, x22 = x2[:, : C // 2, :, :], x2[:, C // 2 :, :, :]
        x21 = self.dwconv3(x21)
        x22 = self.dwconv4(x22)
        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.act2(self.bn2(x2)).flatten(2).transpose(1, 2)

        x31, x32 = x3[:, : C // 2, :, :], x3[:, C // 2 :, :, :]
        x31 = self.dwconv5(x31)
        x32 = self.dwconv6(x32)
        x3 = torch.cat([x31, x32], dim=1)
        x3 = self.act3(self.bn3(x3)).flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MRFP(nn.Module):
    """Multi-Receptive Field Feature Pyramid 模块。

    两个线性投影层 + 一组多感受野 (3x3/5x5) 深度可分离卷积,
    对拼平的多尺度 CNN token 做增强。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiscaleExtractor(nn.Module):
    """CNN token 上的多尺度可变形自注意力 + ConvFFN。"""

    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads,
            n_points=n_points, ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        attn = self.attn(
            self.query_norm(query), reference_points,
            self.feat_norm(feat), spatial_shapes,
            level_start_index, None,
        )
        query = query + attn
        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
        return query


class CTI_toC(nn.Module):
    """Transformer -> CNN 方向的特征交互。

    先把 ViT 当前阶段的输出加到 CNN 金字塔的中间尺度 (stride 16) token 上,
    再在 CNN 多尺度 token 内部做一次可变形自注意力交互。
    """

    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)

        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(
                dim=dim, n_levels=3, num_heads=num_heads,
                n_points=n_points, norm_layer=norm_layer,
                deform_ratio=deform_ratio, with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
            )

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        B, N, C = query.shape
        n = N // 21
        x1 = query[:, 0 : 16 * n, :].contiguous()
        x2 = query[:, 16 * n : 20 * n, :].contiguous()
        x3 = query[:, 20 * n :, :].contiguous()
        x2 = x2 + feat
        query = torch.cat([x1, x2, x3], dim=1)

        if self.cnn_feature_interaction:
            deform_input = deform_inputs_3levels(H, W, query.device)
            query = self.cfinter(
                query=self.query_norm(query), reference_points=deform_input[0],
                feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                level_start_index=deform_input[2], H=H, W=W,
            )
        return query


class Extractor_CTI(nn.Module):
    """CTI_toC 的增强版 (额外带 ConvFFN), 官方用于最后一个阶段的 4 次额外交互。"""

    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(
                dim=dim, n_levels=3, num_heads=num_heads,
                n_points=n_points, norm_layer=norm_layer,
                deform_ratio=deform_ratio, with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
            )

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        B, N, C = query.shape
        n = N // 21
        x1 = query[:, 0 : 16 * n, :].contiguous()
        x2 = query[:, 16 * n : 20 * n, :].contiguous()
        x3 = query[:, 20 * n :, :].contiguous()
        x2 = x2 + feat
        query = torch.cat([x1, x2, x3], dim=1)
        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
        if self.cnn_feature_interaction:
            deform_input = deform_inputs_3levels(H, W, query.device)
            query = self.cfinter(
                query=self.query_norm(query), reference_points=deform_input[0],
                feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                level_start_index=deform_input[2], H=H, W=W,
            )
        return query


class CTI_toV(nn.Module):
    """CNN -> Transformer 方向的特征交互。

    CNN 多尺度 token 做一次跨尺度可变形自注意力 + ConvFFN 后,
    将 stride 8/16/32 三个尺度对齐到 stride 16 网格并求和,
    以可学习缩放因子 gamma 注入 ViT token。
    """

    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.0,
                 drop=0.0, drop_path=0.0, cffn_ratio=0.25):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads,
            n_points=n_points, ratio=deform_ratio,
        )
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        B, N, C = feat.shape
        c1 = self.attn(
            self.query_norm(feat), reference_points,
            self.feat_norm(feat), spatial_shapes,
            level_start_index, None,
        )
        c1 = c1 + self.drop_path(self.ffn(self.ffn_norm(c1), H, W))

        c_select1, c_select2, c_select3 = (
            c1[:, : H * W * 4, :],
            c1[:, H * W * 4 : H * W * 4 + H * W, :],
            c1[:, H * W * 4 + H * W :, :],
        )
        c_select1 = (
            F.interpolate(
                c_select1.permute(0, 2, 1).reshape(B, C, H * 2, W * 2),
                scale_factor=0.5, mode="bilinear", align_corners=False,
            )
            .flatten(2)
            .permute(0, 2, 1)
        )
        c_select3 = (
            F.interpolate(
                c_select3.permute(0, 2, 1).reshape(B, C, H // 2, W // 2),
                scale_factor=2, mode="bilinear", align_corners=False,
            )
            .flatten(2)
            .permute(0, 2, 1)
        )
        return query + self.gamma * (c_select1 + c_select2 + c_select3)


class CTIBlock(nn.Module):
    """一个阶段的 CNN-Transformer 双向交互。

    官方实现中 ViT blocks 也在本模块内顺序运行; 本适配版将其拆为:
      - pre_stage(): ViT 阶段 block 运行前调用 (MRFP + ViT->CNN 注入 + CNN->ViT 注入);
      - post_stage(): ViT 阶段 block 全部运行后调用 (CNN 吸收 ViT 输出 + 可选额外交互)。
    两者由外部通过 hook 在 DINOv3 block 边界触发。
    """

    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0.0, drop_path=0.0, with_cffn=True, cffn_ratio=0.25, init_values=0.0,
                 deform_ratio=1.0, extra_CTI=False,
                 use_CTI_toV=True, use_CTI_toC=True, dim_ratio=6.0,
                 cnn_feature_interaction=False):
        super().__init__()

        if use_CTI_toV:
            self.cti_tov = CTI_toV(
                dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                drop=drop, drop_path=drop_path, cffn_ratio=cffn_ratio,
            )
        if use_CTI_toC:
            self.cti_toc = CTI_toC(
                dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
                cnn_feature_interaction=cnn_feature_interaction,
            )

        if extra_CTI:
            self.extra_CTIs = nn.Sequential(
                *[
                    Extractor_CTI(
                        dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                        norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
                        cnn_feature_interaction=cnn_feature_interaction,
                    )
                    for _ in range(4)
                ]
            )
        else:
            self.extra_CTIs = None

        self.use_CTI_toV = use_CTI_toV
        self.use_CTI_toC = use_CTI_toC

        self.mrfp = MRFP(dim, hidden_features=int(dim * dim_ratio))

    def pre_stage(self, x, c, di3, H, W):
        """阶段开始时: 增强 CNN 特征, 并双向注入。

        Args:
            x: ViT patch token, [B, H*W, D] (阶段运行前)。
            c: CNN 金字塔 token, [B, 21*n, D]。
            di3: deform_inputs_3levels 返回的 [ref_points, spatial_shapes, level_start]。
        Returns:
            x_new, c_new
        """
        if self.use_CTI_toV:
            c = self.mrfp(c, H, W)
            c_select1, c_select2, c_select3 = (
                c[:, : H * W * 4, :],
                c[:, H * W * 4 : H * W * 4 + H * W, :],
                c[:, H * W * 4 + H * W :, :],
            )
            c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)

            x = self.cti_tov(
                query=x, reference_points=di3[0], feat=c,
                spatial_shapes=di3[1], level_start_index=di3[2], H=H, W=W,
            )
        return x, c

    def post_stage(self, c, x, di3, H, W):
        """阶段结束时: CNN token 吸收 ViT 阶段输出, 最后阶段额外做 4 次 CNN 自交互。

        Args:
            c: CNN 金字塔 token。
            x: ViT 阶段输出 patch token, [B, H*W, D]。
        """
        if self.use_CTI_toC:
            c = self.cti_toc(
                query=c, reference_points=di3[0], feat=x,
                spatial_shapes=di3[1], level_start_index=di3[2], H=H, W=W,
            )

        if self.extra_CTIs is not None:
            for cti in self.extra_CTIs:
                c = cti(
                    query=c, reference_points=di3[0], feat=x,
                    spatial_shapes=di3[1], level_start_index=di3[2], H=H, W=W,
                )
        return c


class CNN(nn.Module):
    """ViT-CoMer 的 CNN 分支 (论文中的 SPM/MRFP 金字塔茎干)。

    输入图像经过一串卷积得到 stride 4/8/16/32 四个尺度的特征图,
    再用 1x1 卷积统一到 embed_dim 维 (与 ViT token 维度一致)。

    Returns:
        c1: [B, D, 4H, 4W] 特征图 (stride 4),
        c2/c3/c4: [B, N, D] token (stride 8/16/32)。
    """

    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s, 保留为二维特征图
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
