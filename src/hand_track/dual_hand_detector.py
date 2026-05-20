"""
双手检测器模块
集成 PalmLocalFrame / HandFeatureExtractor / HoverAnchorDetector / ContactStateMachine
实现基于掌面坐标系的 hover-anchored 在线接触检测
"""

import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from enum import Enum

from src.hand_track.palm_coordinate_system import PalmLocalFrame
from src.hand_track.feature_extractor import HandFeatureExtractor
from src.hand_track.hover_anchor_detector import HoverAnchorDetector, HoverDetectResult
from src.hand_track.contact_state_machine import (
    ContactStateMachine,
    SmoothContactResult,
    ContactState,
)
from src.utils.geometry_utils import get_landmark_3d


config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)


class HandRole(Enum):
    WRITING = "writing"
    PALM    = "palm"
    UNKNOWN = "unknown"


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

        self._palm_frame     = PalmLocalFrame()
        self._feat_extractor = HandFeatureExtractor()
        self._hover_det      = HoverAnchorDetector()
        self._contact_sm     = ContactStateMachine()

        # public alias for visualization
        self.palm_local_frame: PalmLocalFrame = self._palm_frame

        self.hover_result:   HoverDetectResult   = None
        self.contact_result: SmoothContactResult = None

        self.is_writing      = False
        self.write_pos       = (0, 0)
        self.write_pos_palm  = None   # (u, v) in palm local frame
        self.dist_palm       = None   # n component (signed distance to palm plane)
        self.last_feat: np.ndarray = np.full(10, np.nan)  # last extracted feature vector
        self.frame_shape     = None
        self.frame_cnt       = 0

    def process(self, frame):
        self.frame_shape = frame.shape
        self.frame_cnt  += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.left_lm = self.right_lm = self.write_lm = self.palm_lm = None
        self.left_role = self.right_role = HandRole.UNKNOWN
        self.write_pos      = (0, 0)
        self.write_pos_palm = None
        self.dist_palm      = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hd.classification[0].label
                # MediaPipe uses anatomical labels: "Right" = user's right hand
                if label == "Right":
                    self.right_lm = lm
                elif label == "Left":
                    self.left_lm = lm
                self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
            self._assign_roles()

        if self.palm_enabled and self.palm_lm:
            self._palm_frame.update(self.palm_lm)

        if self.write_lm:
            tip = self.write_lm.landmark[8]
            self.write_pos = (int(tip.x * w), int(tip.y * h))

            if self._palm_frame.is_valid:
                tip_3d = get_landmark_3d(self.write_lm, 8)
                uc, vc, nc = self._palm_frame.to_local(tip_3d)
                self.write_pos_palm = (uc, vc)
                self.dist_palm = nc

            feat = self._feat_extractor.extract(
                self.write_lm, self.palm_lm, frame_gray, self._palm_frame
            )
        else:
            feat = np.full(10, np.nan)
            self._feat_extractor.reset()

        self.last_feat = feat

        self.hover_result  = self._hover_det.update(feat)
        self.contact_result = self._contact_sm.update(
            self.hover_result.raw_contact,
            self.hover_result.distance,
            self.frame_cnt,
        )
        self.is_writing = self.contact_result.state == ContactState.CONTACT
        return self.is_writing

    def _assign_roles(self):
        if not self.left_lm or not self.right_lm:
            return

        # Heuristic correction for MediaPipe label confusion (e.g. hands close/crossed).
        # For a mirrored front-facing camera (standard Mac setup): the user's
        # anatomical right hand appears on the IMAGE RIGHT side (larger x centroid).
        # If the centroid check disagrees with MediaPipe labels, trust x-position.
        right_cx = float(np.mean([lm.x for lm in self.right_lm.landmark]))
        left_cx  = float(np.mean([lm.x for lm in self.left_lm.landmark]))
        if right_cx < left_cx:          # right hand should be on image RIGHT (larger x)
            self.left_lm, self.right_lm = self.right_lm, self.left_lm

        if self.role_mode == "position":
            if self.dominant == "right":
                self.right_role, self.left_role = HandRole.WRITING, HandRole.PALM
                self.write_lm,   self.palm_lm   = self.right_lm, self.left_lm
            else:
                self.left_role, self.right_role = HandRole.WRITING, HandRole.PALM
                self.write_lm,  self.palm_lm    = self.left_lm, self.right_lm

    def just_started_writing(self):
        if self.contact_result is None:
            return False
        return self.contact_result.changed and self.contact_result.state == ContactState.CONTACT

    def just_stopped_writing(self):
        if self.contact_result is None:
            return False
        return self.contact_result.changed and self.contact_result.state == ContactState.IDLE

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
            'palm_frame':     {'valid': self._palm_frame.is_valid},
            'hover':          {
                'phase':    self.hover_result.phase,
                'progress': self.hover_result.progress,
                'distance': self.hover_result.distance,
                'threshold': self.hover_result.threshold,
            } if self.hover_result else None,
            'contact': {
                'state':   self.contact_result.state.value,
                'changed': self.contact_result.changed,
                'pending': self._contact_sm.get_debug_info()['pending'],
            } if self.contact_result else None,
        }

    def reset(self):
        self._palm_frame.reset()
        self._feat_extractor.reset()
        self._hover_det.reset()
        self._contact_sm.reset()
        self.left_lm = self.right_lm = self.write_lm = self.palm_lm = None
        self.is_writing      = False
        self.write_pos       = (0, 0)
        self.write_pos_palm  = None
        self.dist_palm       = None
        self.hover_result    = None
        self.contact_result  = None
        self.last_feat       = np.full(10, np.nan)
