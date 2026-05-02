"""
双手检测器模块
扩展HandWritingDetector以支持双手检测和角色分配
"""

import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from collections import deque
from typing import Optional
from enum import Enum

from src.hand_track.palm_coordinate_system import PalmPlaneTracker
from src.hand_track.contact_detector import MultiFeatureContactDetector
from src.utils.geometry_utils import get_landmark_3d


# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)


class HandRole(Enum):
    WRITING = "writing"
    PALM    = "palm"
    UNKNOWN = "unknown"


# ── 特征缓冲区（计算 v_n / sigma_d） ────────────────────────────────────────
class _FeatureBuffer:
    _DIST_WIN  = 5
    _SIGMA_WIN = 8

    def __init__(self):
        self._d = deque(maxlen=self._DIST_WIN + 2)
        self._s = deque(maxlen=self._SIGMA_WIN)

    def push(self, dist_mm: float):
        self._d.append(dist_mm)
        self._s.append(dist_mm)

    def v_n(self) -> float:
        if len(self._d) < 2:
            return 0.0
        return float(np.mean(np.diff(list(self._d)[-self._DIST_WIN:])))

    def sigma_d(self) -> float:
        return float(np.std(list(self._s))) if len(self._s) > 1 else 0.0

    def reset(self):
        self._d.clear()
        self._s.clear()


def _compute_dist2d_palm(write_lm, palm_lm, frame_shape: tuple) -> dict:
    """指尖到各掌心关键点的 2D 像素距离"""
    if write_lm is None or palm_lm is None:
        return {}
    h, w = frame_shape[:2]
    tip = write_lm.landmark[8]
    tx, ty = tip.x * w, tip.y * h
    return {i: float(np.hypot(palm_lm.landmark[i].x * w - tx,
                               palm_lm.landmark[i].y * h - ty))
            for i in (0, 5, 9, 13, 17)}


def _compute_brightness(frame_gray: np.ndarray, cx: int, cy: int, r: int = 18) -> float:
    h, w = frame_gray.shape
    roi = frame_gray[max(cy - r, 0):min(cy + r, h),
                     max(cx - r, 0):min(cx + r, w)]
    return float(np.mean(roi)) if roi.size else 128.0


class DualHandDetector:
    def __init__(self):
        hc = CONFIG['hand_detection']
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=hc['static_image_mode'],
            max_num_hands=hc['max_num_hands'],
            min_detection_confidence=hc['min_detection_confidence'],
            min_tracking_confidence=hc['min_tracking_confidence']
        )
        self.mp_drawing = mp.solutions.drawing_utils

        pc = CONFIG.get('palm_writing', {})
        self.palm_enabled = pc.get('enabled', True)
        self.dominant     = pc.get('dominant_hand', 'right')
        self.role_mode    = pc.get('role_assignment_mode', 'position')

        self.left_lm = self.right_lm = self.write_lm = self.palm_lm = None
        self.left_role = self.right_role = HandRole.UNKNOWN

        self.palm_tracker = PalmPlaneTracker(CONFIG)
        self.contact_sm   = MultiFeatureContactDetector(mode="rule")
        self._feat_buf    = _FeatureBuffer()

        self.is_writing      = False
        self.write_pos       = (0, 0)
        self.write_pos_palm  = None
        self.dist_palm       = None
        self.frame_shape     = None
        self.frame_cnt       = 0

    def process(self, frame, ts):
        self.frame_shape = frame.shape
        self.frame_cnt  += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.left_lm = self.right_lm = self.write_lm = self.palm_lm = None
        self.left_role = self.right_role = HandRole.UNKNOWN

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hd.classification[0].label
                if label == "Left":
                    self.right_lm = lm
                elif label == "Right":
                    self.left_lm = lm
                self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
            self._assign_roles()

        if self.palm_enabled and self.palm_lm:
            palm_sys = self.palm_tracker.update(self.palm_lm)

            if self.write_lm and palm_sys:
                tip_3d   = get_landmark_3d(self.write_lm, 8)
                in_bound = palm_sys.is_within_palm_boundary(tip_3d)
                x, y, z  = palm_sys.get_2d_coordinates(tip_3d)
                self.write_pos_palm = (x, y)
                self.dist_palm = z if in_bound else None

                if in_bound:
                    self._feat_buf.push(z)
                    vn    = self._feat_buf.v_n()
                    sigma = self._feat_buf.sigma_d()
                    d2d   = _compute_dist2d_palm(self.write_lm, self.palm_lm, frame.shape)

                    proj = palm_sys.project_to_plane(tip_3d)
                    self.write_pos = (int(proj[0] * frame.shape[1]),
                                      int(proj[1] * frame.shape[0]))
                    brightness = _compute_brightness(frame_gray, *self.write_pos)

                    self.contact_sm.update(z, vn, sigma, d2d, brightness)
                else:
                    self._feat_buf.reset()
                    self.contact_sm.update(None)

                    proj = palm_sys.project_to_plane(tip_3d)
                    self.write_pos = (int(proj[0] * frame.shape[1]),
                                      int(proj[1] * frame.shape[0]))
            else:
                self._feat_buf.reset()
                self.contact_sm.update(None)
                self.write_pos_palm = None
                self.dist_palm = None
        else:
            self._feat_buf.reset()
            self.contact_sm.update(None)
            self.write_pos_palm = None
            self.dist_palm = None

        self.is_writing = self.contact_sm.is_contact()
        return self.is_writing

    def _assign_roles(self):
        if not self.left_lm or not self.right_lm:
            return
        if self.role_mode == "position":
            if self.dominant == "right":
                self.right_role, self.left_role = HandRole.WRITING, HandRole.PALM
                self.write_lm,   self.palm_lm   = self.right_lm, self.left_lm
            else:
                self.left_role, self.right_role = HandRole.WRITING, HandRole.PALM
                self.write_lm,  self.palm_lm    = self.left_lm, self.right_lm

    def _draw_debug(self, frame, ps):
        det   = self.contact_sm
        score = det.get_score()
        state = det.get_state().value

        if self.dist_palm is not None:
            cv2.putText(frame, f"Dist: {self.dist_palm:.1f}mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {state}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Score: {score:.3f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 230, 120), 2)
        if self.write_pos_palm:
            cv2.putText(frame,
                        f"Coord: ({self.write_pos_palm[0]:.3f}, {self.write_pos_palm[1]:.3f})",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.is_writing and self.write_pos:
            cv2.circle(frame, self.write_pos, 8, (0, 0, 255), -1)

    def just_started_writing(self):
        return self.contact_sm.just_started()

    def just_stopped_writing(self):
        return self.contact_sm.just_stopped()

    def get_writing_position(self):
        return self.write_pos_palm

    def get_screen_position(self):
        return self.write_pos

    def get_debug_info(self):
        return {
            'palm_enabled':   self.palm_enabled,
            'left_detected':  self.left_lm is not None,
            'right_detected': self.right_lm is not None,
            'left_role':      self.left_role.value,
            'right_role':     self.right_role.value,
            'is_writing':     self.is_writing,
            'dist_palm':      self.dist_palm,
            'palm_tracker':   self.palm_tracker.get_debug_info(),
            'contact_state':  self.contact_sm.get_debug_info(),
        }

    def reset(self):
        self.palm_tracker.reset()
        self.contact_sm.reset()
        self._feat_buf.reset()
        self.left_lm = self.right_lm = self.write_lm = self.palm_lm = None
        self.is_writing     = False
        self.write_pos      = (0, 0)
        self.write_pos_palm = None
        self.dist_palm      = None
