"""在测试集上推理并生成提交用标注图 (DINOv3 + ViT-CoMer + Mask2Former)。

推理流程与 predict_test.py 完全一致, 仅模型构建函数替换为 ViT-CoMer 版。

用法:
    python predict_comer.py \
        --checkpoint outputs/dinov3_comer_m2f_drone/model_best.pth \
        --test_dir /path/to/test_1/images \
        --output_dir ./predictions

输出: 与测试图同名的灰度 PNG, 像素值 1..8 (与 Label.txt 定义一致, 无 0)。
"""

import predict_test as _base
from dinov3.eval.segmentation.models.dinov3_comer_mask2former import (
    build_dinov3_comer_mask2former,
)

# 替换 DPT 版推理脚本使用的模型构建函数
_base.build_dinov3_dpt_mask2former = build_dinov3_comer_mask2former

if __name__ == "__main__":
    _base.main()
