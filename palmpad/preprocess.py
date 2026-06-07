"""
Offline preprocessing pipeline:
  1. Read each .avi + .txt annotation pair from the HuggingFace dataset.
  2. Detect hand landmarks with MediaPipe (left palm + right index fingertip).
  3. Crop 128x128 palm and index regions from every frame.
  4. Compute Farneback optical flow between consecutive frames.
  5. Save crops and flows as numpy arrays (.npy) for fast training.

Output layout:
  processed/
    group_1/user-0/user-0_0/
      palm.npy    shape (N, 128, 128, 3)  uint8
      index.npy   shape (N, 128, 128, 3)  uint8
      flow.npy    shape (N, 128, 128, 2)  float32
      labels.npy  shape (N,)              int8
"""

import argparse
import os
import sys

# ── Fix: redirect matplotlib/tmp dirs BEFORE importing mediapipe ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TMP  = os.path.join(_HERE, ".tmp")
os.makedirs(_TMP, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _TMP)
os.environ.setdefault("TMPDIR",       _TMP)
os.environ.setdefault("TEMP",         _TMP)
os.environ.setdefault("TMP",          _TMP)
import tempfile
tempfile.tempdir = _TMP
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

CROP_SIZE = 128
PALM_LANDMARKS = [0, 5, 9, 13, 17]   # wrist + 4 base knuckles
INDEX_TIP = 8                         # right hand index fingertip


def _bbox_from_landmarks(lms, indices, h, w, pad: float = 0.3):
    pts = np.array([[lms[i].x * w, lms[i].y * h] for i in indices])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + pad)
    half = side / 2
    x1 = max(0, int(cx - half));  y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half));  y2 = min(h, int(cy + half))
    return x1, y1, x2, y2


def _safe_crop(frame, x1, y1, x2, y2):
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


def process_video(args_tuple):
    """Worker function — runs in a separate process (one MediaPipe instance each)."""
    video_path, label_path, out_dir = args_tuple

    # Skip if already done
    if os.path.exists(os.path.join(out_dir, "labels.npy")):
        return video_path, "skip", 0

    # Label file format: "Timestamp downsample raw"
    # col[0]=timestamp  col[1]=downsample  col[2]=raw(contact label)
    labels_raw = []
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    labels_raw.append(int(float(parts[2])))  # raw = contact label
                except ValueError:
                    continue   # skip header line
    except Exception as e:
        return video_path, "error", str(e)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_path, "error", "cannot open video"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    palms, indices, flows, labels_out = [], [], [], []
    prev_gray = None

    # Explicit subpackage import required for mediapipe >= 0.10
    import mediapipe.python.solutions.hands as _mp_hands
    _Hands = _mp_hands.Hands

    with _Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for fi in range(total):
            ok, frame = cap.read()
            if not ok or fi >= len(labels_raw):
                break

            h, w = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            palm_crop  = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            index_crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

            if result.multi_hand_landmarks and result.multi_handedness:
                left_lm = right_lm = None
                for lm, info in zip(result.multi_hand_landmarks,
                                    result.multi_handedness):
                    side = info.classification[0].label
                    if side == "Left":
                        left_lm = lm.landmark
                    else:
                        right_lm = lm.landmark

                if left_lm is not None:
                    x1, y1, x2, y2 = _bbox_from_landmarks(
                        left_lm, PALM_LANDMARKS, h, w)
                    palm_crop = _safe_crop(frame, x1, y1, x2, y2)

                if right_lm is not None:
                    tip = right_lm[INDEX_TIP]
                    cx, cy = int(tip.x * w), int(tip.y * h)
                    half = int(CROP_SIZE * 0.8)
                    x1 = max(0, cx - half);  y1 = max(0, cy - half)
                    x2 = min(w, cx + half);  y2 = min(h, cy + half)
                    index_crop = _safe_crop(frame, x1, y1, x2, y2)

            palms.append(palm_crop)
            indices.append(index_crop)
            labels_out.append(labels_raw[fi])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                flow_crop = cv2.resize(flow, (CROP_SIZE, CROP_SIZE))
            else:
                flow_crop = np.zeros((CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)

            flows.append(flow_crop)
            prev_gray = gray

    cap.release()

    if len(palms) < 2:
        return video_path, "skip", "too few frames"

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "palm.npy"),
            np.stack(palms).astype(np.uint8))
    np.save(os.path.join(out_dir, "index.npy"),
            np.stack(indices).astype(np.uint8))
    np.save(os.path.join(out_dir, "flow.npy"),
            np.stack(flows).astype(np.float32))
    np.save(os.path.join(out_dir, "labels.npy"),
            np.array(labels_out, dtype=np.int8))

    return video_path, "ok", len(palms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root",  default="data/palmpad_proc")
    parser.add_argument("--workers",   type=int,
                        default=min(8, cpu_count()),
                        help="Parallel worker processes")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)

    # Collect all (video, label, out_dir) triples
    video_files = sorted(data_root.rglob("*.avi"))
    print(f"Found {len(video_files)} .avi files")

    tasks = []
    for vp in video_files:
        lp = vp.with_suffix(".txt")
        if not lp.exists():
            print(f"  [warn] no label for {vp.name}, skipping")
            continue
        rel     = vp.relative_to(data_root)
        out_dir = str(out_root / rel.with_suffix(""))
        tasks.append((str(vp), str(lp), out_dir))

    print(f"Tasks to process: {len(tasks)}  (workers={args.workers})")

    errors = []
    with Pool(processes=args.workers) as pool:
        for vpath, status, info in tqdm(
            pool.imap_unordered(process_video, tasks),
            total=len(tasks),
            desc="Preprocessing",
        ):
            if status == "error":
                errors.append((vpath, info))
                tqdm.write(f"  ERROR {Path(vpath).name}: {info}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for p, e in errors:
            print(f"  {p}: {e}")
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
