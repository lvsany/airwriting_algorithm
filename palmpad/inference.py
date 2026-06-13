"""
Real-time PalmPad inference from a USB/webcam RGB camera.

Architecture matches the paper's asynchronous pipeline:
  - Camera thread   : reads frames at full camera fps (up to 120 fps)
  - MediaPipe thread: detects hand landmarks at ~20 fps
  - Model thread    : runs PalmPad CNN+LSTM at ~100 fps using cached landmarks

Usage:
  python inference.py --checkpoint checkpoints/best.pt --camera 0
"""

import argparse
import time
import threading
import queue
import collections
import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms

from model import PalmPadModel

CROP_SIZE = 128
PALM_LANDMARKS = [0, 5, 9, 13, 17]
INDEX_TIP = 8

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

def _bbox_from_landmarks(lms, indices, h, w, pad=0.3):
    pts = np.array([[lms[i].x * w, lms[i].y * h] for i in indices])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + pad)
    half = side / 2
    x1 = max(0, int(cx - half)); y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half)); y2 = min(h, int(cy + half))
    return x1, y1, x2, y2


def _safe_crop(frame, x1, y1, x2, y2):
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


class PalmPadInference:
    def __init__(self, model: PalmPadModel, device: torch.device, time_steps=2):
        self.model = model
        self.device = device
        self.time_steps = time_steps

        # Ring buffers for crops and flows
        self._palms  = collections.deque(maxlen=time_steps)
        self._indices = collections.deque(maxlen=time_steps)
        self._flows  = collections.deque(maxlen=time_steps)
        self._prev_gray = None

        self.touch = False
        self.confidence = 0.0

    def push_frame(self, frame, palm_lm, index_lm):
        h, w = frame.shape[:2]

        palm_crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        index_crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

        if palm_lm is not None:
            x1, y1, x2, y2 = _bbox_from_landmarks(palm_lm, PALM_LANDMARKS, h, w)
            palm_crop = _safe_crop(frame, x1, y1, x2, y2)

        if index_lm is not None:
            tip = index_lm[INDEX_TIP]
            cx, cy = int(tip.x * w), int(tip.y * h)
            half = int(CROP_SIZE * 0.8)
            x1 = max(0, cx - half); y1 = max(0, cy - half)
            x2 = min(w, cx + half); y2 = min(h, cy + half)
            index_crop = _safe_crop(frame, x1, y1, x2, y2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            flow_crop = cv2.resize(flow, (CROP_SIZE, CROP_SIZE))
        else:
            flow_crop = np.zeros((CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)

        self._prev_gray = gray
        self._palms.append(palm_crop)
        self._indices.append(index_crop)
        self._flows.append(flow_crop)

        if len(self._palms) == self.time_steps:
            self._run_model()

    @torch.no_grad()
    def _run_model(self):
        def prep_rgb(arr):
            return _normalise(arr.copy())

        palm_t  = torch.stack([prep_rgb(p) for p in self._palms]).unsqueeze(0)
        index_t = torch.stack([prep_rgb(i) for i in self._indices]).unsqueeze(0)
        flow_t  = torch.stack([
            torch.from_numpy(f.copy()).permute(2, 0, 1) for f in self._flows
        ]).unsqueeze(0)

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(
                    palm_t.to(self.device),
                    index_t.to(self.device),
                    flow_t.to(self.device),
                )
        else:
            logits = self.model(
                palm_t.to(self.device),
                index_t.to(self.device),
                flow_t.to(self.device),
            )
        probs = torch.softmax(logits, dim=-1)[0].cpu().float()
        self.touch      = probs[1].item() > 0.5
        self.confidence = probs[1].item()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PalmPadModel(time_steps=args.time_steps)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    inferrer = PalmPadInference(model, device, time_steps=args.time_steps)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)

    frame_q: queue.Queue = queue.Queue(maxsize=4)
    landmark_q: queue.Queue = queue.Queue(maxsize=4)

    # MediaPipe thread
    mp_hands = mp.solutions.hands
    def mp_worker():
        with mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as hands:
            while True:
                try:
                    frame = frame_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                left_lm = right_lm = None
                if result.multi_hand_landmarks and result.multi_handedness:
                    for lm, info in zip(
                        result.multi_hand_landmarks, result.multi_handedness
                    ):
                        label = info.classification[0].label
                        if label == "Left":
                            left_lm = lm.landmark
                        else:
                            right_lm = lm.landmark
                try:
                    landmark_q.put_nowait((frame, left_lm, right_lm))
                except queue.Full:
                    pass

    mp_thread = threading.Thread(target=mp_worker, daemon=True)
    mp_thread.start()

    # Cached landmarks (updated by mp_thread output)
    cached_palm_lm  = None
    cached_index_lm = None

    fps_counter = collections.deque(maxlen=60)
    print("Running — press Q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Push to MediaPipe queue (non-blocking)
        try:
            frame_q.put_nowait(frame.copy())
        except queue.Full:
            pass

        # Consume landmark results
        while not landmark_q.empty():
            _, cached_palm_lm, cached_index_lm = landmark_q.get_nowait()

        t0 = time.perf_counter()
        inferrer.push_frame(frame, cached_palm_lm, cached_index_lm)
        fps_counter.append(time.perf_counter() - t0)

        # Overlay
        status_color = (0, 255, 0) if inferrer.touch else (0, 0, 255)
        status_text  = f"TOUCH {inferrer.confidence:.2f}" if inferrer.touch \
                       else f"no-touch {inferrer.confidence:.2f}"
        cv2.putText(frame, status_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, status_color, 3)
        if fps_counter:
            inf_fps = 1.0 / (sum(fps_counter) / len(fps_counter))
            cv2.putText(frame, f"inf {inf_fps:.0f}fps", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("PalmPad", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_q.put(None)
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--camera",      type=int, default=0)
    parser.add_argument("--time_steps",  type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
