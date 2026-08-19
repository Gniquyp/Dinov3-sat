import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps

from dinov3.hub.backbones import dinov3_vitl16, Weights
from dinov3.eval.segmentation.models.dinov3_dpt_mask2former import build_dinov3_dpt_mask2former

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_dinov3_dpt_m2f")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(inputs, targets, num_masks):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def batch_dice_loss(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs, targets):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_mask=1.0, cost_dice=1.0, num_points=12544):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        from scipy.optimize import linear_sum_assignment

        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets[b]["labels"]

            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]

            num_tgt = tgt_mask.shape[0]
            num_out = out_mask.shape[0]
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            grid = 2.0 * point_coords - 1.0
            grid = grid.unsqueeze(1)

            tgt_mask_sampled = F.grid_sample(tgt_mask, grid.expand(num_tgt, -1, -1, -1),
                                              mode="bilinear", padding_mode="zeros",
                                              align_corners=False).squeeze(2).squeeze(1)
            out_mask_sampled = F.grid_sample(out_mask, grid.expand(num_out, -1, -1, -1),
                                              mode="bilinear", padding_mode="zeros",
                                              align_corners=False).squeeze(2).squeeze(1)

            with torch.cuda.amp.autocast(enabled=False):
                out_mask_sampled = out_mask_sampled.float()
                tgt_mask_sampled = tgt_mask_sampled.float()
                cost_mask = batch_sigmoid_ce_loss(out_mask_sampled, tgt_mask_sampled)
                cost_dice = batch_dice_loss(out_mask_sampled, tgt_mask_sampled)

            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_masks):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)  # (batch_idx, tgt_within_image_idx)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        # 与 detectron2 nested_tensor 行为一致: 对每张图的 [N_b, H, W] 在
        # 目标数/高/宽三个维度上 pad 后 stack 为 [B, N_max, H_max, W_max],
        # 再用 (batch_idx, tgt_idx) 二元组索引取出匹配的目标
        masks = [t["masks"] for t in targets]
        max_n = max(m.shape[0] for m in masks)
        max_h = max(m.shape[1] for m in masks)
        max_w = max(m.shape[2] for m in masks)
        padded_masks = [
            F.pad(m, (0, max_w - m.shape[2], 0, max_h - m.shape[1], 0, max_n - m.shape[0]))
            for m in masks
        ]
        target_masks = torch.stack(padded_masks).to(src_masks.device)
        target_masks = target_masks[tgt_idx]  # [num_matches, H_max, W_max]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(target_masks, size=src_masks.shape[-2:],
                                         mode="bilinear", align_corners=False)

        N = src_masks.shape[0]
        point_coords = torch.rand(N, self.num_points, 2, device=src_masks.device)
        grid = 2.0 * point_coords - 1.0
        grid = grid.unsqueeze(1)

        with torch.no_grad():
            point_labels = F.grid_sample(target_masks, grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=False)
            point_labels = point_labels.squeeze(2).squeeze(1)
            point_labels = (point_labels > 0.5).float()

        point_logits = F.grid_sample(src_masks, grid, mode="bilinear",
                                      padding_mode="zeros", align_corners=False)
        point_logits = point_logits.squeeze(2).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_criterion(num_classes, mask_weight=20.0, dice_weight=1.0, cls_weight=2.0,
                    eos_coef=0.1, dec_layers=9):
    matcher = HungarianMatcher(
        cost_class=cls_weight,
        cost_mask=mask_weight,
        cost_dice=dice_weight,
        num_points=12544,
    )

    weight_dict = {
        "loss_ce": cls_weight,
        "loss_mask": mask_weight,
        "loss_dice": dice_weight,
    }

    aux_weight_dict = {}
    for i in range(dec_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    losses = ["labels", "masks"]

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )

    return criterion


class DroneSegDataset(Dataset):
    """低空无人机航拍语义分割数据集 (2026 低空图像语义分割赛道)

    目录结构:
        root/
          images/    *.png  RGB 图像
          train/     *.png  灰度标注 (L 模式), 像素值 0..8
    标签定义 (Label.txt):
        0=Ignore, 1=Background, 2=Building, 3=Road, 4=Water,
        5=Barren, 6=Vegetation, 7=Agricultural, 8=Vehicle
    标签映射: 原始 1..8 -> 0..7 (共 8 类), 原始 0 -> 255 (ignore_index)

    数据集不带验证集, 内部按固定种子做 train/val 划分, 不复制文件。
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root, split="train", transform=None,
                 images_subdir="images", masks_subdir="train",
                 val_ratio=0.05, seed=42, reduce_zero_label=True):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.reduce_zero_label = reduce_zero_label

        img_dir = self.root / images_subdir
        ann_dir = self.root / masks_subdir
        img_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in self.IMG_EXTENSIONS)
        ann_files = sorted(f for f in ann_dir.iterdir() if f.suffix.lower() == ".png")
        assert len(img_files) == len(ann_files), \
            f"图像({len(img_files)})和标注({len(ann_files)})数量不匹配"
        # 同名配对校验
        for i, a in zip(img_files, ann_files):
            assert i.stem == a.stem, f"图像与标注文件名不对应: {i.name} vs {a.name}"

        # 固定种子的确定性 train/val 划分
        n = len(img_files)
        perm = np.random.RandomState(seed).permutation(n)
        n_val = int(round(n * val_ratio))
        if split == "train":
            keep = perm[n_val:]
        elif split == "val":
            keep = perm[:n_val]
        elif split == "all":
            keep = perm
        else:
            raise ValueError(f"未知 split: {split}")
        self.img_files = [img_files[i] for i in keep]
        self.ann_files = [ann_files[i] for i in keep]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        ann = Image.open(self.ann_files[idx])

        if self.transform:
            img, ann = self.transform(img, ann)
        else:
            img = transforms.ToTensor()(img)
            ann = torch.from_numpy(np.array(ann)).long()

        if self.reduce_zero_label:
            # 原始 1..8 -> 0..7, 原始 0 (Ignore) -> 255
            ann = ann - 1
            ann[ann < 0] = 255

        return img, ann


class TrainTransform:
    """训练增广: 随机缩放 + 随机裁剪 img_size + 水平翻转 + 颜色抖动(仅图像)"""

    def __init__(self, img_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 flip_prob=0.5, scale_range=(0.75, 1.5), color_jitter=0.3):
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.jitter = (transforms.ColorJitter(color_jitter, color_jitter, color_jitter)
                       if color_jitter > 0 else None)

    def __call__(self, img, ann):
        # 随机缩放
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        ann = ann.resize((new_w, new_h), Image.NEAREST)

        # 不足 img_size 时右侧/下侧 padding (图像填0, 标注填255=ignore)
        pad_w = max(0, self.img_size - new_w)
        pad_h = max(0, self.img_size - new_h)
        if pad_w > 0 or pad_h > 0:
            img = ImageOps.expand(img, (0, 0, pad_w, pad_h), fill=0)
            ann = ImageOps.expand(ann, (0, 0, pad_w, pad_h), fill=255)
            new_w, new_h = img.size

        # 随机裁剪
        i = random.randint(0, new_h - self.img_size)
        j = random.randint(0, new_w - self.img_size)
        img = img.crop((j, i, j + self.img_size, i + self.img_size))
        ann = ann.crop((j, i, j + self.img_size, i + self.img_size))

        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ann = ann.transpose(Image.FLIP_LEFT_RIGHT)

        if self.jitter is not None:
            img = self.jitter(img)

        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        ann = torch.from_numpy(np.array(ann)).long()

        return img, ann


class ValTransform:
    """验证: 直接整体缩放到 img_size"""

    def __init__(self, img_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img, ann):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        ann = ann.resize((self.img_size, self.img_size), Image.NEAREST)

        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        ann = torch.from_numpy(np.array(ann)).long()

        return img, ann


def prepare_targets(gt_masks, ignore_index=255):
    targets = []
    batch_size = gt_masks.shape[0]

    for b in range(batch_size):
        mask = gt_masks[b]
        valid = mask != ignore_index
        unique_classes = torch.unique(mask[valid])
        unique_classes = unique_classes[unique_classes != ignore_index]

        if len(unique_classes) == 0:
            targets.append({
                "labels": torch.zeros(0, dtype=torch.long, device=mask.device),
                "masks": torch.zeros(0, mask.shape[0], mask.shape[1], device=mask.device),
            })
            continue

        masks_per_class = []
        labels_per_class = []
        for cls in unique_classes:
            cls_mask = (mask == cls).float()
            masks_per_class.append(cls_mask)
            labels_per_class.append(cls)

        targets.append({
            "labels": torch.stack(labels_per_class),
            "masks": torch.stack(masks_per_class),
        })

    return targets


def compute_miou(pred_masks, pred_logits, gt_masks, num_classes, ignore_index=255):
    with torch.no_grad():
        pred_cls = pred_logits.argmax(dim=-1)
        B, Q, H, W = pred_masks.shape
        pred_seg = torch.full((B, H, W), num_classes, dtype=torch.long, device=pred_masks.device)
        max_scores = torch.zeros(B, H, W, device=pred_masks.device) - 1

        for b in range(B):
            mask_scores = pred_masks[b].sigmoid()
            cls_scores = pred_logits[b].softmax(dim=-1)
            cls_ids = pred_cls[b]

            for q in range(Q):
                cls_idx = cls_ids[q]
                if cls_idx >= num_classes:
                    continue
                conf = cls_scores[q, cls_idx]
                binary_mask = mask_scores[q] > 0.5
                score_mask = mask_scores[q] * conf

                update_mask = binary_mask & (score_mask > max_scores[b])
                pred_seg[b][update_mask] = cls_idx
                max_scores[b][update_mask] = score_mask[update_mask]

        valid = gt_masks != ignore_index
        intersection = torch.zeros(num_classes, device=pred_masks.device)
        union = torch.zeros(num_classes, device=pred_masks.device)

        for cls in range(num_classes):
            pred_cls_mask = pred_seg == cls
            gt_cls_mask = gt_masks == cls
            intersection[cls] = (pred_cls_mask & gt_cls_mask & valid).sum().float()
            union[cls] = ((pred_cls_mask | gt_cls_mask) & valid).sum().float()

        valid_classes = union > 0
        if valid_classes.sum() > 0:
            iou = intersection[valid_classes] / (union[valid_classes] + 1e-6)
            miou = iou.mean().item()
        else:
            miou = 0.0

    return miou


def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, scaler=None, grad_clip=1.0):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (imgs, gt_masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        gt_masks = gt_masks.to(device)

        targets = prepare_targets(gt_masks)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            scaler.scale(losses).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {losses.item():.4f} "
                f"CE: {loss_dict.get('loss_ce', torch.tensor(0)).item():.4f} "
                f"Mask: {loss_dict.get('loss_mask', torch.tensor(0)).item():.4f} "
                f"Dice: {loss_dict.get('loss_dice', torch.tensor(0)).item():.4f}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, criterion, dataloader, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_miou = 0.0
    num_batches = 0

    for imgs, gt_masks in dataloader:
        imgs = imgs.to(device)
        gt_masks = gt_masks.to(device)

        targets = prepare_targets(gt_masks)

        outputs = model(imgs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # pred_masks 分辨率为输入的 1/4, 需上采样到 gt 尺寸再算 mIoU
        pred_masks = F.interpolate(
            outputs["pred_masks"], size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        miou = compute_miou(pred_masks, outputs["pred_logits"], gt_masks, num_classes)

        total_loss += losses.item()
        total_miou += miou
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches

    return avg_loss, avg_miou


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 + DPT + Mask2Former")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录 (包含 images/ 和标注子目录)")
    parser.add_argument("--images_subdir", type=str, default="images", help="图像子目录名")
    parser.add_argument("--masks_subdir", type=str, default="train", help="标注子目录名")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集划分比例")
    parser.add_argument("--output_dir", type=str, default="./outputs/dinov3_dpt_m2f", help="输出目录")
    parser.add_argument("--img_size", type=int, default=512, help="训练裁剪大小")
    parser.add_argument("--batch_size", type=int, default=2, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--num_classes", type=int, default=8, help="类别数 (赛道为 8 类)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--eval_interval", type=int, default=5, help="验证间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="保存间隔")
    parser.add_argument("--backbone_pretrained", type=str, default="", help="backbone 预训练权重路径")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的 checkpoint 路径")
    parser.add_argument("--unfreeze_backbone", action="store_true", default=False,
                        help="放开主干微调 (默认冻结主干只训练头)")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1,
                        help="放开主干时 backbone 学习率 = lr x 该系数 (防止冲垮预训练特征)")
    parser.add_argument("--use_amp", action="store_true", default=False, help="使用混合精度训练")

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info("构建 DINOv3 backbone (SAT493M 权重)...")
    backbone = dinov3_vitl16(pretrained=not args.backbone_pretrained, weights=Weights.SAT493M)
    if args.backbone_pretrained:
        state_dict = torch.load(args.backbone_pretrained, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        backbone.load_state_dict(state_dict, strict=False)
        logger.info(f"从 {args.backbone_pretrained} 加载 backbone 权重")

    logger.info("构建 DINOv3 + DPT + Mask2Former 模型...")
    model = build_dinov3_dpt_mask2former(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        freeze_backbone=not args.unfreeze_backbone,
    )
    logger.info("Backbone 已放开微调" if args.unfreeze_backbone else "Backbone 已冻结")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    logger.info("构建损失函数...")
    criterion = build_criterion(num_classes=args.num_classes)
    criterion = criterion.to(device)

    logger.info("构建数据集...")
    train_transform = TrainTransform(img_size=args.img_size)
    val_transform = ValTransform(img_size=args.img_size)

    train_dataset = DroneSegDataset(
        root=args.data_root,
        split="train",
        transform=train_transform,
        images_subdir=args.images_subdir,
        masks_subdir=args.masks_subdir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    val_dataset = DroneSegDataset(
        root=args.data_root,
        split="val",
        transform=val_transform,
        images_subdir=args.images_subdir,
        masks_subdir=args.masks_subdir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    do_eval = len(val_dataset) > 0
    if not do_eval:
        logger.warning("验证集为空, 将跳过验证 (val_ratio 过小?)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info("构建优化器...")
    if args.unfreeze_backbone:
        # 分层学习率: backbone 用小 lr, 头部用正常 lr
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("encoder.backbone")]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("encoder.backbone")]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
                {"params": head_params, "lr": args.lr},
            ],
            lr=args.lr, weight_decay=args.weight_decay,
        )
        logger.info(f"分层学习率: backbone {args.lr * args.backbone_lr_mult:.2e}, "
                    f"head {args.lr:.2e}")
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and torch.cuda.is_available() else None

    start_epoch = 0
    best_miou = 0.0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)
        logger.info(f"从 epoch {start_epoch} 恢复训练, 最佳 mIoU: {best_miou:.4f}")

    logger.info("开始训练...")
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"{'='*50}")

        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch + 1, scaler=scaler, grad_clip=args.grad_clip,
        )

        scheduler.step()

        logger.info(f"Epoch {epoch + 1} 训练平均 Loss: {train_loss:.4f}")

        if do_eval and ((epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1):
            val_loss, val_miou = validate(model, criterion, val_loader, device, args.num_classes)
            logger.info(f"Epoch {epoch + 1} 验证 Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                best_path = os.path.join(args.output_dir, "model_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "args": vars(args),
                }, best_path)
                logger.info(f"保存最佳模型到 {best_path}, mIoU: {best_miou:.4f}")

        if (epoch + 1) % args.save_interval == 0 or epoch == args.num_epochs - 1:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_miou": best_miou,
                "args": vars(args),
            }, save_path)
            logger.info(f"保存 checkpoint 到 {save_path}")

    logger.info("\n训练完成!")
    logger.info(f"最佳 mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
