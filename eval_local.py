"""在本地验证集上全分辨率评测 mIoU (不花比赛提交次数)。

验证集划分与训练时完全一致 (seed=42, val_ratio=0.05 -> 350 张),
评测在原始 1024x1024 分辨率上进行 (与比赛评测口径一致),
因此这里的分数比训练日志里的 mIoU 更接近排行榜真实成绩。

用法:
    python eval_local.py \
        --checkpoint outputs/model_best.pth \
        --data_root /hy-tmp/bi_sai/train --masks_subdir masks \
        --img_sizes 512 768 1024 --tta

输出: 每个推理尺寸下的逐类 IoU 和 mIoU, 用于选最佳推理配置。
注意: img_size 必须能被 32 整除 (pixel decoder 的要求)。
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dinov3.hub.backbones import dinov3_vitl16, Weights
from dinov3.eval.segmentation.models.dinov3_dpt_mask2former import build_dinov3_dpt_mask2former
from train_dinov3_dpt_mask2former import DroneSegDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_local")

CLASS_NAMES = ["Background", "Building", "Road", "Water",
               "Barren", "Vegetation", "Agricultural", "Vehicle"]


@torch.no_grad()
def predict_scores(model, x, num_classes, out_size):
    """返回 [C, H_out, W_out] 逐类得分 (未 argmax), 便于 TTA 累加."""
    outputs = model(x)
    cls_probs = outputs["pred_logits"].softmax(-1)[0, :, :num_classes]  # [Q, C]
    mask_probs = outputs["pred_masks"].sigmoid()[0]                     # [Q, h, w]
    scores = torch.einsum("qc,qhw->chw", cls_probs, mask_probs)         # [C, h, w]
    return F.interpolate(scores[None], size=out_size, mode="bilinear", align_corners=False)[0]


def update_confusion(conf, gt, pred, num_classes):
    """gt: [H, W] int64, 值 0..C-1 或 255(忽略); pred: [H, W] int64, 值 0..C-1."""
    valid = gt != 255
    conf += np.bincount(
        num_classes * gt[valid] + pred[valid],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def report(conf, num_classes, tag):
    """由混淆矩阵计算逐类 IoU 与 mIoU 并打印."""
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    logger.info(f"--- {tag} ---")
    for c in range(num_classes):
        logger.info(f"  {CLASS_NAMES[c]:<14s} IoU = {iou[c]:.4f}")
    logger.info(f"  mIoU = {np.nanmean(iou):.4f}")
    return np.nanmean(iou)


def main():
    parser = argparse.ArgumentParser(description="Local full-resolution val evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True, help="训练集根目录 (含 images/ 和标注子目录)")
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--masks_subdir", type=str, default="masks")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_sizes", type=int, nargs="+", default=[512],
                        help="要对比的推理尺寸, 均需被 32 整除, 如 512 768 1024")
    parser.add_argument("--tta", action="store_true", help="加测水平翻转 TTA (两版分数都会输出)")
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()

    for s in args.img_sizes:
        assert s % 32 == 0, f"img_size {s} 必须能被 32 整除"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DroneSegDataset(
        args.data_root, split="val", transform=None,
        images_subdir=args.images_subdir, masks_subdir=args.masks_subdir,
        val_ratio=args.val_ratio, seed=args.seed,
    )
    logger.info(f"验证集 {len(dataset)} 张 (与训练时划分一致)")

    backbone = dinov3_vitl16(pretrained=False, weights=Weights.SAT493M)
    model = build_dinov3_dpt_mask2former(
        backbone=backbone, hidden_dim=args.hidden_dim, num_classes=args.num_classes,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    logger.info(f"已加载 checkpoint (epoch {checkpoint.get('epoch')}, "
                f"best mIoU {checkpoint.get('best_miou', 0):.4f})")

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    C = args.num_classes
    conf = {s: np.zeros((C, C), dtype=np.int64) for s in args.img_sizes}
    conf_tta = {s: np.zeros((C, C), dtype=np.int64) for s in args.img_sizes}

    for idx in range(len(dataset)):
        img = Image.open(dataset.img_files[idx]).convert("RGB")
        orig_size = (img.size[1], img.size[0])  # (H, W)
        ann = np.array(Image.open(dataset.ann_files[idx]))
        gt = ann.astype(np.int64) - 1          # 1..8 -> 0..7
        gt[ann == 0] = 255                     # Ignore

        for s in args.img_sizes:
            x = img.resize((s, s), Image.BILINEAR)
            x = normalize(transforms.ToTensor()(x)).unsqueeze(0).to(device)

            pred = predict_scores(model, x, C, orig_size).argmax(0).cpu().numpy()
            update_confusion(conf[s], gt, pred, C)

            if args.tta:
                scores = predict_scores(model, x, C, orig_size)
                scores = scores + torch.flip(
                    predict_scores(model, torch.flip(x, dims=[3]), C, orig_size), dims=[2])
                update_confusion(conf_tta[s], gt, scores.argmax(0).cpu().numpy(), C)

        if (idx + 1) % 50 == 0:
            logger.info(f"进度 {idx + 1}/{len(dataset)}")

    logger.info("================ 评测结果 (原始分辨率) ================")
    for s in args.img_sizes:
        report(conf[s], C, f"img_size={s}")
        if args.tta:
            report(conf_tta[s], C, f"img_size={s} + hflip TTA")


if __name__ == "__main__":
    main()
