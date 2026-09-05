"""在本地验证集上全分辨率评测 mIoU (DINOv3 + ViT-CoMer + Mask2Former)。

评测流程与 eval_local.py 完全一致 (同样的验证集划分、TTA、img_size 要求),
仅模型构建函数替换为 ViT-CoMer 版。

用法:
    python eval_local_comer.py \
        --checkpoint outputs/dinov3_comer_m2f_drone/model_best.pth \
        --data_root /hy-tmp/bi_sai/train --masks_subdir masks \
        --img_sizes 512 768 1024 --tta

注意: img_size 必须能被 32 整除 (CoMer SPM 下采样 32x)。
"""

import eval_local as _base
from dinov3.eval.segmentation.models.dinov3_comer_mask2former import (
    build_dinov3_comer_mask2former,
)

# 替换 DPT 版评测脚本使用的模型构建函数
_base.build_dinov3_dpt_mask2former = build_dinov3_comer_mask2former

if __name__ == "__main__":
    _base.main()
