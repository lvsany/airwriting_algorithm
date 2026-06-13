"""
Contact detector: PalmPad (CHI 2025).

Uses PalmPadInference (ResNet18 + LSTM) for contact detection.
Hand tracking uses mediapipe >= 0.10 Tasks API (HandLandmarker),
which replaces the deprecated mp.solutions.hands.

palm_coordinate_system is still used to produce (u, v) stroke coordinates
so the data format matches own_framework exactly.
"""

import os
import sys
import urllib.request
import yaml
import numpy as np
import cv2
import torch

# --- path setup (self-contained, CWD-independent) --------------------------
_THIS    = os.path.dirname(os.path.abspath(__file__))
_DATASETS = os.path.dirname(_THIS)
_PROJECT = os.path.dirname(_DATASETS)
_PALMPAD = os.path.join(_PROJECT, "palmpad")
for _p in (_PROJECT, _PALMPAD):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

from model import PalmPadModel
from inference import PalmPadInference
from src.hand_track.palm_coordinate_system import PalmLocalFrame
from src.utils.geometry_utils import get_landmark_3d
from .base import ContactDetectorBase

# hand_landmarker.task — shared with exp3_eval_palmpad.py
_DEFAULT_TASK_PATH = os.path.join(_DATASETS, "hand_landmarker.task")
_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Match DualHandDetector's still-hold thresholds
_STILL_END_SEC = 1.0
_STILL_END_PX  = 20.0


def _read_fps() -> int:
    cfg_path = os.path.join(_PROJECT, "src", "config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("video", {}).get("fps", 60)
    except Exception:
        return 60


# ---------------------------------------------------------------------------
# Mediapipe Tasks API wrapper
# ---------------------------------------------------------------------------

class _LandmarkListWrapper:
    """
    Wraps a new-API NormalizedLandmarkList so it exposes a `.landmark`
    attribute, making it compatible with PalmLocalFrame / get_landmark_3d /
    PalmPadInference which all expect `obj.landmark[i].{x,y,z}`.
    """
    __slots__ = ("landmark",)

    def __init__(self, lm_list):
        self.landmark = lm_list  # list of NormalizedLandmark with .x .y .z


class _HandTracker:
    """
    Per-frame hand tracker backed by mediapipe >= 0.10 Tasks API.

    Assigns roles based on image-x centroid (same heuristic as DualHandDetector):
    - right-dominant user: image-right hand → writing, image-left → palm
    - single hand visible: assigned by centroid side

    Exposes:
      palm_lm   : _LandmarkListWrapper | None
      write_lm  : _LandmarkListWrapper | None
      screen_pos: (x, y) of index fingertip in pixels
    """

    def __init__(self, task_path: str):
        if not os.path.exists(task_path):
            print(f"[HandLandmarker] Model not found. Downloading to {task_path} ...")
            os.makedirs(os.path.dirname(task_path) or ".", exist_ok=True)
            urllib.request.urlretrieve(_TASK_URL, task_path)
            print("[HandLandmarker] Download complete.")

        import mediapipe as mp
        from mediapipe.tasks import python as _mp_py
        from mediapipe.tasks.python import vision as _mp_vis

        options = _mp_vis.HandLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=task_path),
            running_mode=_mp_vis.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._det = _mp_vis.HandLandmarker.create_from_options(options)

        self.palm_lm   = None
        self.write_lm  = None
        self.screen_pos = (0, 0)

    def process(self, frame: np.ndarray, dominant: str = "right"):
        import mediapipe as mp

        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._det.detect(mp_img)

        self.palm_lm    = None
        self.write_lm   = None
        self.screen_pos = (0, 0)

        if not result.hand_landmarks:
            return

        # Sort hands by image-x centroid (left → right)
        hands = sorted(
            [(_LandmarkListWrapper(lm_list),
              float(np.mean([lm.x for lm in lm_list])))
             for lm_list in result.hand_landmarks],
            key=lambda t: t[1],
        )

        if len(hands) == 1:
            wrapped, cx = hands[0]
            if dominant == "right":
                if cx >= 0.5:
                    self.write_lm = wrapped
                else:
                    self.palm_lm  = wrapped
            else:
                if cx < 0.5:
                    self.write_lm = wrapped
                else:
                    self.palm_lm  = wrapped
        else:
            # image-left hand → palm for right-dominant; image-right → writing
            if dominant == "right":
                self.palm_lm  = hands[0][0]
                self.write_lm = hands[-1][0]
            else:
                self.write_lm = hands[0][0]
                self.palm_lm  = hands[-1][0]

        if self.write_lm is not None:
            tip = self.write_lm.landmark[8]  # index fingertip
            self.screen_pos = (int(tip.x * w), int(tip.y * h))

    def reset(self):
        self.palm_lm    = None
        self.write_lm   = None
        self.screen_pos = (0, 0)


# ---------------------------------------------------------------------------
# ContactDetectorBase implementation
# ---------------------------------------------------------------------------

class PalmPadDetector(ContactDetectorBase):
    name = "palmpad"
    needs_calibration = False   # no hover-anchor calibration; skip CALIB phase
    hover_result = None

    def __init__(self, checkpoint_path: str, time_steps: int = 2,
                 task_path: str = None, dominant: str = "right"):
        # Load PalmPad model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = PalmPadModel(time_steps=time_steps)
        ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state  = ckpt.get("state_dict", ckpt)
        state  = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval().to(device)
        self._inferrer = PalmPadInference(model, device, time_steps=time_steps)
        print(f"[PalmPad] Loaded {checkpoint_path} on {device}  "
              f"(val_f1={ckpt.get('val_f1', float('nan')):.1f}%)")

        # Hand tracker (mediapipe Tasks API)
        self._tracker  = _HandTracker(task_path or _DEFAULT_TASK_PATH)
        self._dominant = dominant

        # Palm local frame for (u, v) stroke coordinates
        self._palm_frame     = PalmLocalFrame()
        self._write_pos      = (0, 0)
        self._write_pos_palm = None

        # Still-hold state (driven by PalmPad's own contact signal)
        self._still_threshold  = int(round(_read_fps() * _STILL_END_SEC))
        self._last_screen_pos  = None
        self._still_frames     = 0
        self._still_hold       = False
        self._still_hold_event = False
        self._is_writing       = False

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> bool:
        self._still_hold_event = False

        # Hand tracking
        self._tracker.process(frame, self._dominant)
        self._write_pos = self._tracker.screen_pos

        # Palm coordinate frame
        if self._tracker.palm_lm is not None:
            self._palm_frame.update(self._tracker.palm_lm)

        # Writing position in palm frame
        self._write_pos_palm = None
        if self._tracker.write_lm is not None and self._palm_frame.is_valid:
            tip_3d = get_landmark_3d(self._tracker.write_lm, 8)
            uc, vc, _ = self._palm_frame.to_local(tip_3d)
            self._write_pos_palm = (uc, vc)

        # PalmPad contact inference
        palm_lm  = (self._tracker.palm_lm.landmark
                    if self._tracker.palm_lm  is not None else None)
        index_lm = (self._tracker.write_lm.landmark
                    if self._tracker.write_lm is not None else None)
        self._inferrer.push_frame(frame, palm_lm, index_lm)
        self._is_writing = self._inferrer.touch

        self._update_still_hold()
        if self._still_hold:
            self._is_writing = False

        return self._is_writing

    def get_screen_position(self) -> tuple:
        return self._write_pos

    def get_writing_position(self):
        return self._write_pos_palm

    def consume_still_hold_event(self) -> bool:
        triggered = self._still_hold_event
        self._still_hold_event = False
        return triggered

    def reset(self):
        self._tracker.reset()
        self._palm_frame.reset()
        self._write_pos      = (0, 0)
        self._write_pos_palm = None
        inf = self._inferrer
        inf._palms.clear()
        inf._indices.clear()
        inf._flows.clear()
        inf._prev_gray  = None
        inf.touch       = False
        inf.confidence  = 0.0
        self._is_writing       = False
        self._last_screen_pos  = None
        self._still_frames     = 0
        self._still_hold       = False
        self._still_hold_event = False

    # ------------------------------------------------------------------
    def _update_still_hold(self):
        pos = self._write_pos

        if pos == (0, 0):
            self._last_screen_pos = None
            self._still_frames    = 0
            self._still_hold      = False
            return

        if self._last_screen_pos is None:
            self._last_screen_pos = pos
            return

        dx    = pos[0] - self._last_screen_pos[0]
        dy    = pos[1] - self._last_screen_pos[1]
        moved = (dx * dx + dy * dy) > (_STILL_END_PX ** 2)

        prev_hold = self._still_hold
        if moved:
            self._still_frames = 0
            self._still_hold   = False
        else:
            if self._is_writing or self._still_hold:
                self._still_frames += 1
                if self._still_frames >= self._still_threshold:
                    self._still_hold = True
            else:
                self._still_frames = 0

        if not prev_hold and self._still_hold:
            self._still_hold_event = True

        self._last_screen_pos = pos
