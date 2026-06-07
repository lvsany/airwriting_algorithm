# PalmPad 复现

CHI 2025 论文复现：**Palmpad: Enabling Real-Time Index-to-Palm Touch Interaction with a Single RGB Camera**  
He et al., doi:[10.1145/3706598.3714130](https://dl.acm.org/doi/10.1145/3706598.3714130)

---

## 架构

```
RGB Camera (120fps)
       │
  MediaPipe Hands  →  Palm landmarks + Index fingertip
       │
  ┌────┴─────────────────────────┐
  │  time_steps=2 frame window   │
  │  palm crop  (128×128×3)      │  ResNet18 → 1000-d
  │  index crop (128×128×3)      │  ResNet18 → 1000-d
  │  optical flow (128×128×2)    │  ResNet18 → 1000-d
  └──────────────────────────────┘
       │ concat → 3000-d per time step
       │
     LSTM (hidden=512)
       │
     MLP → touch / no-touch
```

## 环境安装（服务器 4090）

```bash
pip install -r requirements.txt
```

## Step 1：下载数据集（~96 GB）

```bash
python download_dataset.py --out_dir /data/palmpad_raw
```

## Step 2：离线预处理

MediaPipe 检测手部关键点，裁剪 palm/index 区域，计算光流，保存为 `.npy`。

```bash
python preprocess.py \
    --data_root /data/palmpad_raw \
    --out_root  /data/palmpad_proc \
    --workers   8
```

预计耗时：~2-4 小时（取决于 CPU 核心数）。

## Step 3：训练

```bash
python train.py \
    --processed_root /data/palmpad_proc \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-4 \
    --frame_interval 2 \
    --workers 8
```

`frame_interval` 对应论文的时间间隔设置：
| frame_interval | 实际时间间隔（120fps）|
|---|---|
| 1  | 1/120s |
| 2  | 1/60s  |
| 4  | 1/30s  |
| 6  | 1/20s  |

监控训练：
```bash
tensorboard --logdir runs/
```

## Step 4：实时推理

```bash
python inference.py --checkpoint checkpoints/best.pt --camera 0
```

## 论文指标

| 指标 | 论文结果 | 说明 |
|------|---------|------|
| Accuracy | 97.0% | 16 用户 |
| F1-score | 96.1% | macro |

## 文件说明

| 文件 | 功能 |
|------|------|
| `model.py` | PalmPad 模型（3×ResNet18 + LSTM + MLP） |
| `dataset.py` | PyTorch Dataset，滑窗采样 |
| `preprocess.py` | 离线预处理（MediaPipe + 光流） |
| `train.py` | 训练脚本，bf16 混合精度，torch.compile |
| `inference.py` | 实时推理，异步多线程流水线 |
| `download_dataset.py` | 从 HuggingFace 下载数据 |
