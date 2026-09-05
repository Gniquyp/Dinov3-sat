"""训练入口: DINOv3 + ViT-CoMer + Mask2Former

数据加载、损失函数、训练循环、checkpoint 保存等逻辑与 DPT 版完全一致,
仅把多尺度特征桥接模块从 DPT 替换为 ViT-CoMer (CVPR 2024 Highlight)。
因此这里直接复用 train_dinov3_dpt_mask2former.py 的全部实现,
只替换模型构建函数。

用法 (推荐通过 run_train.py + 配置文件):
    python run_train.py --config configs/train_dinov3_comer_mask2former.yaml
或:
    python train_dinov3_comer_mask2former.py --data_root <path> --output_dir outputs/...
"""

import train_dinov3_dpt_mask2former as _base
from dinov3.eval.segmentation.models.dinov3_comer_mask2former import (
    build_dinov3_comer_mask2former,
)

# 替换 DPT 版训练脚本使用的模型构建函数 (CoMer builder 兼容其全部调用参数)
_base.build_dinov3_dpt_mask2former = build_dinov3_comer_mask2former

if __name__ == "__main__":
    _base.main()
