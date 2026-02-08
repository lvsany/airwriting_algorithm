"""
接触检测状态机模块
管理手指与手掌平面的接触状态转换
"""

from enum import Enum
from typing import Optional, Tuple
from collections import deque
import numpy as np


class ContactState(Enum):
    """接触状态枚举"""
    IDLE = "idle"  # 空闲（无手或手距离太远）
    HOVERING = "hovering"  # 悬停（手接近但未接触）
    TOUCHING = "touching"  # 触碰（刚接触平面）
    WRITING = "writing"  # 书写中（持续接触并移动）
    LIFTED = "lifted"  # 抬起（刚离开平面）


class ContactStateMachine:
    def __init__(self, config: dict):
        cc = config.get('palm_writing', {}).get('contact_detection', {})
        self.thresh_idle = cc.get('idle_to_hover_threshold', 100)
        self.thresh_hover = cc.get('hover_to_touch_threshold', 30)
        self.thresh_touch = cc.get('touch_to_write_threshold', 15)
        self.thresh_lift = cc.get('write_to_lift_threshold', 25)
        self.min_speed = cc.get('min_writing_speed', 50)
        self.max_hover_speed = cc.get('max_hover_speed', 200)
        self.smooth_win = cc.get('temporal_smoothing_window', 3)
        
        self.state = ContactState.IDLE
        self.prev = ContactState.IDLE
        self.frames = 0
        self.dist_hist = deque(maxlen=self.smooth_win)
        self.vel_hist = deque(maxlen=self.smooth_win)
        self.pos_hist = deque(maxlen=10)
        self.time_hist = deque(maxlen=10)
        
    def update(self, dist, pos, ts):
        if dist is None:
            self._trans(ContactState.IDLE)
            return self.state
        
        self.dist_hist.append(dist)
        d = np.median(list(self.dist_hist))
        
        v = 0.0
        if pos:
            self.pos_hist.append(pos)
            self.time_hist.append(ts)
            if len(self.pos_hist) >= 2:
                v = self._calc_vel()
                self.vel_hist.append(v)
        
        v_smooth = np.mean(list(self.vel_hist)) if self.vel_hist else 0.0
        new = self._next_state(d, v_smooth)
        
        if new != self.state:
            self._trans(new)
        else:
            self.frames += 1
        return self.state
    
    def _next_state(self, d, v):
        if d > self.thresh_idle:
            return ContactState.IDLE
        if d > self.thresh_hover:
            if self.state in [ContactState.WRITING, ContactState.TOUCHING] and v > self.min_speed:
                return self.state
            return ContactState.HOVERING
        if d <= self.thresh_touch:
            if v >= self.min_speed:
                return ContactState.WRITING
            if self.state == ContactState.WRITING and v > self.min_speed * 0.5:
                return ContactState.WRITING
            return ContactState.TOUCHING
        if self.thresh_touch < d <= self.thresh_hover:
            if self.state in [ContactState.WRITING, ContactState.TOUCHING]:
                return ContactState.LIFTED if d > self.thresh_lift else self.state
            return ContactState.HOVERING
        return self.state
    
    def _calc_vel(self):
        if len(self.pos_hist) < 2:
            return 0.0
        n = min(5, len(self.pos_hist))
        tot_d, tot_t = 0.0, 0.0
        for i in range(len(self.pos_hist) - n, len(self.pos_hist) - 1):
            p1, p2 = np.array(self.pos_hist[i]), np.array(self.pos_hist[i+1])
            t1, t2 = self.time_hist[i], self.time_hist[i+1]
            dt = t2 - t1
            if dt > 0:
                tot_d += np.linalg.norm(p2 - p1)
                tot_t += dt
        return 0.0 if tot_t < 1e-6 else (tot_d / tot_t) * 1000.0
    
    def _trans(self, new):
        self.prev, self.state, self.frames = self.state, new, 0
    
    def is_writing(self):
        return self.state in [ContactState.TOUCHING, ContactState.WRITING]
    
    def just_started_writing(self):
        return (self.prev in [ContactState.IDLE, ContactState.HOVERING, ContactState.LIFTED] and
                self.state in [ContactState.TOUCHING, ContactState.WRITING])
    
    def just_stopped_writing(self):
        return (self.prev in [ContactState.TOUCHING, ContactState.WRITING] and
                self.state in [ContactState.LIFTED, ContactState.HOVERING, ContactState.IDLE])
    
    def get_state_name(self):
        return self.state.value
    
    def reset(self):
        self.state = self.prev = ContactState.IDLE
        self.frames = 0
        self.dist_hist.clear()
        self.vel_hist.clear()
        self.pos_hist.clear()
        self.time_hist.clear()
    
    def get_debug_info(self):
        return {
            'current_state': self.state.value,
            'previous_state': self.prev.value,
            'state_frames': self.frames,
            'distance_buffer': list(self.dist_hist),
            'velocity_buffer': list(self.vel_hist),
            'smoothed_distance': np.median(list(self.dist_hist)) if self.dist_hist else None,
            'smoothed_velocity': np.mean(list(self.vel_hist)) if self.vel_hist else None
        }
