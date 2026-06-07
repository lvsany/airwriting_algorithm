"""
Offline preprocessing pipeline:
  1. Read each .avi + .txt annotation pair from the HuggingFace dataset.
  2. Detect hand landmarks with MediaPipe (left palm + right index fingertip).
  3. Crop 128x128 palm and index regions from every frame.
  4. Compute Farneback optical flow between consecutive frames.
  5. Save crops and flows as compressed numpy archives (.npz) for fast training.

Output layout:
  processed/
    group{g}/user{u}/seg{s}/
      palm.npy    shape (N, 128, 128, 3)  uint8
      index.npy   shape (N, 128, 128, 3)  uint8
      flow.npy    shape (N-1, 128, 128, 2) float32
      labels.npy  shape (N,)              int8
"""

import argparse
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

CROP_SIZE = 128
PALM_LANDMARKS = [0, 5, 9, 13, 17]   # wrist + 4 base knuckles
INDEX_TIP = 8                         # right hand index tip

mp_hands = mp.solutions.hands


def _bbox_from_landmarks(lms, indices, h, w, pad: float = 0.3):
    """Bounding box (x1,y1,x2,y2) around a set of landmark indices, with padding."""
    pts = np.array([[lms[i].x * w, lms[i].y * h] for i in indices])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + pad)
    half = side / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))
    return x1, y1, x2, y2


def _safe_crop(frame, x1, y1, x2, y2):
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


def process_video(video_path: str, label_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Read labels (frame-level)
    labels_raw = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels_raw.append(int(float(parts[1])))

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    palms, indices, flows, labels_out = [], [], [], []
    prev_gray = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for fi in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if fi >= len(labels_raw):
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Default crops (black) when detection fails
            palm_crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            index_crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

            if result.multi_hand_landmarks and result.multi_handedness:
                left_lm = right_lm = None
                for lm, info in zip(
                    result.multi_hand_landmarks, result.multi_handedness
                ):
                    label = info.classification[0].label
                    if label == "Left":
                        left_lm = lm.landmark   # user's left = palm pad
                    else:
                        right_lm = lm.landmark  # user's right = writing hand

                if left_lm is not None:
                    x1, y1, x2, y2 = _bbox_from_landmarks(left_lm, PALM_LANDMARKS, h, w)
                    palm_crop = _safe_crop(frame, x1, y1, x2, y2)

                if right_lm is not None:
                    tip = right_lm[INDEX_TIP]
                    cx, cy = int(tip.x * w), int(tip.y * h)
                    half = int(CROP_SIZE * 0.8)
                    x1 = max(0, cx - half)
                    y1 = max(0, cy - half)
                    x2 = min(w, cx + half)
                    y2 = min(h, cy + half)
                    index_crop = _safe_crop(frame, x1, y1, x2, y2)

            palms.append(palm_crop)
            indices.append(index_crop)
            labels_out.append(labels_raw[fi])

            # Optical flow (filled with zeros for the first frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                # Crop flow to palm region (same bbox as palm)
                flow_crop = cv2.resize(flow, (CROP_SIZE, CROP_SIZE))
            else:
                flow_crop = np.zeros((CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)

            flows.append(flow_crop)
            prev_gray = gray

    cap.release()

    if len(palms) < 2:
        return

    np.save(os.path.join(out_dir, "palm.npy"), np.stack(palms).astype(np.uint8))
    np.save(os.path.join(out_dir, "index.npy"), np.stack(indices).astype(np.uint8))
    np.save(os.path.join(out_dir, "flow.npy"), np.stack(flows).astype(np.float32))
    np.save(os.path.join(out_dir, "labels.npy"), np.array(labels_out, dtype=np.int8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to downloaded dataset root")
    parser.add_argument("--out_root", default="processed", help="Output directory")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    # Dataset structure: group{g}/user_{u}/  with *.avi + *.txt pairs
    video_files = sorted(data_root.rglob("*.avi"))
    print(f"Found {len(video_files)} videos")

    for vp in tqdm(video_files, desc="Preprocessing"):
        lp = vp.with_suffix(".txt")
        if not lp.exists():
            continue
        # Reconstruct relative path for output
        rel = vp.relative_to(data_root)
        out_dir = out_root / rel.with_suffix("")
        if (out_dir / "labels.npy").exists():
            continue  # already done
        process_video(str(vp), str(lp), str(out_dir))


if __name__ == "__main__":
    main()
