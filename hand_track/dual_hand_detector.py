"""
双手检测器模块
扩展HandWritingDetector以支持双手检测和角色分配
"""

import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict
from enum import Enum

from hand_track.palm_coordinate_system import PalmPlaneTracker
from hand_track.contact_state_machine import ContactStateMachine
from utils.geometry_utils import get_landmark_3d


# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)


class HandRole(Enum):
    """手部角色"""
    WRITING = "writing"  # 书写手
    PALM = "palm"  # 画布手（手掌）
    UNKNOWN = "unknown"  # 未知


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
        self.dominant = pc.get('dominant_hand', 'right')
        self.role_mode = pc.get('role_assignment_mode', 'position')
        
        self.left_lm = self.right_lm = self.write_lm = self.palm_lm = None
        self.left_role = self.right_role = HandRole.UNKNOWN
        
        self.palm_tracker = PalmPlaneTracker(CONFIG)
        self.contact_sm = ContactStateMachine(CONFIG)
        
        self.is_writing = False
        self.write_pos = (0, 0)
        self.write_pos_palm = None
        self.dist_palm = None
        self.frame_shape = None
        self.frame_cnt = 0
        
    def process(self, frame, ts):
        self.frame_shape = frame.shape
        self.frame_cnt += 1
        
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
                tip_3d = get_landmark_3d(self.write_lm, 8)
                in_bound = palm_sys.is_within_palm_boundary(tip_3d)
                x, y, z = palm_sys.get_2d_coordinates(tip_3d)
                self.write_pos_palm = (x, y)
                self.dist_palm = z if in_bound else None
                
                if in_bound:
                    self.contact_sm.update(z, (x, y), ts)
                    self.is_writing = self.contact_sm.is_writing()
                else:
                    self.contact_sm.update(None, None, ts)
                    self.is_writing = False
                
                proj = palm_sys.project_to_plane(tip_3d)
                self.write_pos = (int(proj[0] * self.frame_shape[1]), int(proj[1] * self.frame_shape[0]))
            else:
                self.is_writing = False
                self.write_pos_palm = None
                self.dist_palm = None
                self.contact_sm.update(None, None, ts)
        else:
            self.is_writing = False
            self.write_pos_palm = None
            self.dist_palm = None
        return self.is_writing
    
    def _assign_roles(self):
        if not self.left_lm or not self.right_lm:
            return
        if self.role_mode == "position":
            if self.dominant == "right":
                self.right_role, self.left_role = HandRole.WRITING, HandRole.PALM
                self.write_lm, self.palm_lm = self.right_lm, self.left_lm
            else:
                self.left_role, self.right_role = HandRole.WRITING, HandRole.PALM
                self.write_lm, self.palm_lm = self.left_lm, self.right_lm
    
    def _draw_debug(self, frame, ps):
        sm = self.contact_sm
        thr = sm.writing_threshold
        buf_size = len(sm.adaptive_buffer)
        is_adaptive = sm.adaptive_enabled and buf_size >= sm.adaptive_min_samples

        if self.dist_palm is not None:
            cv2.putText(frame, f"Dist: {self.dist_palm:.1f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {sm.get_state_name()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        thr_label = f"Threshold: {thr:.1f}mm  [{'ADAPTIVE' if is_adaptive else f'WARMUP {buf_size}/{sm.adaptive_min_samples}'}]"
        thr_color = (80, 230, 120) if is_adaptive else (0, 165, 255)
        cv2.putText(frame, thr_label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, thr_color, 2)
        if self.write_pos_palm:
            cv2.putText(frame, f"Coord: ({self.write_pos_palm[0]:.3f}, {self.write_pos_palm[1]:.3f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.write_lm and not ps.is_within_palm_boundary(get_landmark_3d(self.write_lm, 8)):
            cv2.putText(frame, "Out of Boundary", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.is_writing and self.write_pos:
            cv2.circle(frame, self.write_pos, 8, (0, 0, 255), -1)
    
    def just_started_writing(self):
        return self.contact_sm.just_started_writing()
    
    def just_stopped_writing(self):
        return self.contact_sm.just_stopped_writing()
    
    def get_writing_position(self):
        return self.write_pos_palm
    
    def get_screen_position(self):
        return self.write_pos
    
    def get_debug_info(self):
        return {
            'palm_enabled': self.palm_enabled,
            'left_detected': self.left_lm is not None,
            'right_detected': self.right_lm is not None,
            'left_role': self.left_role.value,
            'right_role': self.right_role.value,
            'is_writing': self.is_writing,
            'dist_palm': self.dist_palm,
            'palm_tracker': self.palm_tracker.get_debug_info(),
            'contact_state': self.contact_sm.get_debug_info()
        }
    
    def reset(self):
        self.palm_tracker.reset()
        self.contact_sm.reset()
        self.left_lm = self.right_lm = self.write_lm = self.palm_lm = None
        self.is_writing = False
        self.write_pos = (0, 0)
        self.write_pos_palm = None
        self.dist_palm = None
