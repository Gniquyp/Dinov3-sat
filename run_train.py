import argparse
import sys
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 + ViT-CoMer/DPT + Mask2Former")
    parser.add_argument("--config", type=str, default="configs/train_dinov3_comer_mask2former.yaml",
                        help="配置文件路径 (文件名含 comer 时走 ViT-CoMer, 否则走 DPT)")
    parser.add_argument("--data_root", type=str, default=None, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练 checkpoint")
    parser.add_argument("--backbone_pretrained", type=str, default=None, help="backbone 预训练权重")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None and key != "config":
            config[key] = value

    sys.argv = [sys.argv[0]]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append(f"--{key}")
        else:
            sys.argv.extend([f"--{key}", str(value)])

    if "comer" in os.path.basename(args.config).lower():
        from train_dinov3_comer_mask2former import main as train_main
    else:
        from train_dinov3_dpt_mask2former import main as train_main
    train_main()


if __name__ == "__main__":
    main()
