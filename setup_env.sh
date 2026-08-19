#!/bin/bash
# ============================================================
# DINOv3-SAT + DPT + Mask2Former 一键环境部署脚本
# 适配: Driver 550.163.01 (最高 CUDA 12.4), Linux
# 组合: Python 3.10 + PyTorch 2.5.1 (cu124) + CUDA Toolkit 12.4 (含 nvcc)
# 用法: bash setup_env.sh
# ============================================================
set -e

echo "============================================"
echo " Step 1/6: 创建 Conda 环境"
echo "============================================"
conda create -n dinov3-sat python=3.10 -y

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dinov3-sat

echo ""
echo "============================================"
echo " Step 2/6: 安装 CUDA Toolkit 12.4 (含 nvcc, 编译扩展用)"
echo "============================================"
if command -v nvcc >/dev/null 2>&1; then
    echo "检测到系统已有 nvcc: $(nvcc --version | grep release), 跳过安装"
else
    # 注意: conda-forge 的 cudatoolkit 包不含 nvcc! 必须用 nvidia 源的 cuda-toolkit
    conda install -c nvidia cuda-toolkit=12.4 -y
fi
# 让 setup.py 找到编译器
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}

echo ""
echo "============================================"
echo " Step 3/6: 安装 PyTorch 2.5.1 (CUDA 12.4)"
echo "============================================"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "============================================"
echo " Step 4/6: 安装 Python 依赖包"
echo "============================================"
pip install numpy scipy Pillow omegaconf PyYAML termcolor torchmetrics

echo ""
echo "============================================"
echo " Step 5/6: 编译 CUDA 扩展 (可变形注意力)"
echo "============================================"
cd dinov3/eval/segmentation/models/utils/ops
python setup.py build install
cd -

echo ""
echo "============================================"
echo " Step 6/6: 验证环境"
echo "============================================"
if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] nvcc: $(nvcc --version | grep release)"
else
    echo "[警告] 未找到 nvcc! 请执行: export CUDA_HOME=\$CONDA_PREFIX 并确认 cuda-toolkit 已安装"
fi
python -c "
import torch
print(f'[OK] PyTorch {torch.__version__}')
print(f'[OK] CUDA {torch.version.cuda}')
print(f'[OK] GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'[OK] GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB)')

import numpy;       print('[OK] numpy')
import scipy;       print('[OK] scipy')
from PIL import Image; print('[OK] Pillow')
import omegaconf;   print('[OK] omegaconf')
import yaml;        print('[OK] PyYAML')
import termcolor;   print('[OK] termcolor')

from MultiScaleDeformableAttention import MultiScaleDeformableAttention
print('[OK] MultiScaleDeformableAttention (CUDA扩展)')

print('')
print('============================================')
print(' 环境就绪!')
print('')
print(' 数据集目录结构 (2026 低空语义分割赛道):')
print('   <data_root>/images/*.png   (图像)')
print('   <data_root>/train/*.png    (标注, 值 0..8, 0=Ignore)')
print('')
print(' 启动训练 (单卡):')
print('   CUDA_VISIBLE_DEVICES=2 python train_dinov3_dpt_mask2former.py \\')
print('       --data_root /path/to/train/train \\')
print('       --backbone_pretrained /path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \\')
print('       --use_amp --output_dir ./outputs/')
print(' 注意: 训练脚本当前仅支持单卡 (无 DDP 代码)')
print('============================================')
"

echo ""
echo "Done!"
