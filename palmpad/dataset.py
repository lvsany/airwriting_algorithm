"""
PyTorch Dataset for the preprocessed PalmPad data.

Each sample is a window of `time_steps` consecutive frames:
  palm  : (T, 3, 128, 128)  float32, ImageNet-normalised
  index : (T, 3, 128, 128)  float32, ImageNet-normalised
  flow  : (T, 2, 128, 128)  float32
  label : int   (0 = no-touch, 1 = touch)

The label of the window is taken from the last frame.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalise = transforms.Compose([
    transforms.ToTensor(),           # HWC uint8 → CHW float [0,1]
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class PalmPadDataset(Dataset):
    def __init__(
        self,
        processed_root: str,
        time_steps: int = 2,
        frame_interval: int = 1,   # gap between consecutive time steps (frames)
        augment: bool = True,
        user_ids: list | None = None,  # None = all users
    ):
        self.time_steps = time_steps
        self.frame_interval = frame_interval
        self.augment = augment

        self._segments = []   # list of (palm_arr, index_arr, flow_arr, label_arr)
        self._windows = []    # list of (seg_idx, start_frame_idx)

        root = Path(processed_root)
        seg_dirs = sorted(d for d in root.rglob("labels.npy"))

        for lp in seg_dirs:
            seg_dir = lp.parent
            if user_ids is not None:
                # Filter by user id embedded in path
                uid = int(seg_dir.parts[-2].split("_")[-1])
                if uid not in user_ids:
                    continue

            palm   = np.load(seg_dir / "palm.npy",   mmap_mode="r")
            index_ = np.load(seg_dir / "index.npy",  mmap_mode="r")
            flow   = np.load(seg_dir / "flow.npy",   mmap_mode="r")
            labels = np.load(seg_dir / "labels.npy", mmap_mode="r")

            N = len(labels)
            seg_idx = len(self._segments)
            self._segments.append((palm, index_, flow, labels))

            # Build sliding window indices; need `frame_interval*(time_steps-1)` look-back
            min_start = frame_interval * (time_steps - 1)
            for fi in range(min_start, N):
                self._windows.append((seg_idx, fi))

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        seg_idx, end = self._windows[idx]
        palm_arr, index_arr, flow_arr, labels = self._segments[seg_idx]

        # Frame indices for this window (oldest → newest)
        frame_ids = [end - self.frame_interval * (self.time_steps - 1 - t)
                     for t in range(self.time_steps)]

        palm_tensors  = []
        index_tensors = []
        flow_tensors  = []

        for fi in frame_ids:
            p = palm_arr[fi]   # (128, 128, 3) uint8
            i = index_arr[fi]
            f = flow_arr[fi]   # (128, 128, 2) float32

            if self.augment:
                p, i, f = _augment(p, i, f)

            palm_tensors.append(_normalise(p.copy()))
            index_tensors.append(_normalise(i.copy()))
            flow_tensors.append(torch.from_numpy(f.copy()).permute(2, 0, 1))  # (2,H,W)

        palm_out  = torch.stack(palm_tensors)   # (T, 3, 128, 128)
        index_out = torch.stack(index_tensors)
        flow_out  = torch.stack(flow_tensors)   # (T, 2, 128, 128)
        label     = int(labels[end])

        return palm_out, index_out, flow_out, label


# ---------------------------------------------------------------------------
# Simple augmentation (spatial only — no temporal jitter needed for time_steps=2)
# ---------------------------------------------------------------------------

import cv2

def _augment(palm, index, flow):
    # Random horizontal flip (same transform applied to all)
    if np.random.rand() < 0.5:
        palm  = palm[:, ::-1]
        index = index[:, ::-1]
        flow  = flow[:, ::-1]
        flow[..., 0] *= -1   # flip x component of optical flow

    # Random brightness / contrast jitter on RGB crops
    alpha = np.random.uniform(0.8, 1.2)
    beta  = np.random.randint(-20, 20)
    palm  = np.clip(palm.astype(np.int32) * alpha + beta, 0, 255).astype(np.uint8)
    index = np.clip(index.astype(np.int32) * alpha + beta, 0, 255).astype(np.uint8)

    return palm, index, flow
