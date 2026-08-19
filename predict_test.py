"""在测试集上推理并生成提交用标注图。

用法:
    python predict_test.py \
        --checkpoint outputs/dinov3_dpt_m2f_drone/model_best.pth \
        --test_dir /path/to/test_1/images \
        --output_dir ./predictions

输出: 与测试图同名的灰度 PNG, 像素值 1..8 (与 Label.txt 定义一致, 无 0)。
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict")

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@torch.no_grad()
def predict_semantic(model, img_tensor, num_classes, out_size):
    """单张图像语义分割推理.

    Args:
        img_tensor: [1, 3, H, W] 已归一化
        out_size: (H_out, W_out) 输出分辨率

    Returns:
        pred: [H_out, W_out] int64, 值域 0..num_classes-1
    """
    outputs = model(img_tensor)
    cls_probs = outputs["pred_logits"].softmax(-1)[0, :, :num_classes]  # [Q, C] 去掉 no-object
    mask_probs = outputs["pred_masks"].sigmoid()[0]                     # [Q, h, w]
    # Mask2Former 语义推理: 逐类得分 = sum_q cls_prob * mask_prob
    scores = torch.einsum("qc,qhw->chw", cls_probs, mask_probs)         # [C, h, w]
    scores = F.interpolate(scores[None], size=out_size, mode="bilinear", align_corners=False)[0]
    return scores.argmax(0)


def main():
    parser = argparse.ArgumentParser(description="Predict test set with DINOv3 + DPT + Mask2Former")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练保存的 model_best.pth")
    parser.add_argument("--test_dir", type=str, required=True, help="测试图像目录 (test_1/images)")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="预测结果输出目录")
    parser.add_argument("--img_size", type=int, default=512, help="推理输入尺寸")
    parser.add_argument("--num_classes", type=int, default=8, help="类别数")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度 (需与训练一致)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("构建模型 (无需下载预训练权重, 权重全部来自 checkpoint)...")
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
    test_files = sorted(f for f in Path(args.test_dir).iterdir() if f.suffix.lower() in IMG_EXTENSIONS)
    logger.info(f"共 {len(test_files)} 张测试图像")

    for idx, img_path in enumerate(test_files):
        img = Image.open(img_path).convert("RGB")
        orig_size = (img.size[1], img.size[0])  # (H, W)
        x = img.resize((args.img_size, args.img_size), Image.BILINEAR)
        x = normalize(transforms.ToTensor()(x)).unsqueeze(0).to(device)

        pred = predict_semantic(model, x, args.num_classes, orig_size)  # 0..7
        pred = (pred + 1).byte().cpu().numpy()  # 还原为 1..8

        Image.fromarray(pred, mode="L").save(out_dir / img_path.name)
        if (idx + 1) % 50 == 0:
            logger.info(f"已完成 {idx + 1}/{len(test_files)}")

    logger.info(f"全部完成, 结果保存在 {out_dir}")


if __name__ == "__main__":
    main()
