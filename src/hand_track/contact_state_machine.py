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
    WRITING = "writing"  # 书写中（持续接触并移动）


class ContactStateMachine:
    def __init__(self, config: dict):
        cc = config.get('palm_writing', {}).get('contact_detection', {})
        self.writing_threshold = cc.get('writing_threshold', cc.get('touch_to_write_threshold', 15))
        self.smooth_win = cc.get('temporal_smoothing_window', 3)

        # --- 自适应阈值配置 ---
        ac = cc.get('adaptive_threshold', {})
        self.adaptive_enabled = ac.get('enabled', True)
        self.adaptive_buffer = deque(maxlen=ac.get('buffer_size', 300))
        self.adaptive_min_samples = ac.get('min_samples', 60)
        self.adaptive_update_interval = ac.get('update_interval', 30)
        self.threshold_min = ac.get('threshold_min', 3.0)
        self.threshold_max = ac.get('threshold_max', 30.0)
        self._frames_since_update = 0

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

        # 自适应阈值更新
        if self.adaptive_enabled:
            self._update_adaptive_threshold(dist)

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
    
    def _update_adaptive_threshold(self, dist: float):
        """将新距离样本加入缓冲区，达到更新间隔后用 Otsu 方法重新计算阈值。"""
        self.adaptive_buffer.append(dist)
        self._frames_since_update += 1
        if (len(self.adaptive_buffer) >= self.adaptive_min_samples
                and self._frames_since_update >= self.adaptive_update_interval):
            self._frames_since_update = 0
            new_thresh = self._otsu_threshold(list(self.adaptive_buffer))
            if new_thresh is not None:
                self.writing_threshold = new_thresh

    def _otsu_threshold(self, distances: list) -> Optional[float]:
        """Otsu 最优阈值：最大化两类（书写 / 非书写）的类间方差。

        当距离分布为单峰（偏差 < 1mm）时返回 None，沿用当前阈值。
        """
        arr = np.asarray(distances, dtype=float)
        min_d, max_d = arr.min(), arr.max()
        if max_d - min_d < 1.0:          # 分布过窄，无法区分两类
            return None

        n_bins = 100
        hist, edges = np.histogram(arr, bins=n_bins, range=(min_d, max_d))
        hist = hist.astype(float) / hist.sum()   # 归一化为概率
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        best_thresh: float = self.writing_threshold
        best_var: float = -1.0

        for i in range(1, n_bins):
            w0 = hist[:i].sum()
            w1 = hist[i:].sum()
            if w0 < 1e-6 or w1 < 1e-6:
                continue
            mu0 = (hist[:i] * bin_centers[:i]).sum() / w0
            mu1 = (hist[i:] * bin_centers[i:]).sum() / w1
            var_between = w0 * w1 * (mu0 - mu1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thresh = float(edges[i])

        return float(np.clip(best_thresh, self.threshold_min, self.threshold_max))

    def _next_state(self, d, v):
        return ContactState.WRITING if d <= self.writing_threshold else ContactState.IDLE
    
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
        return self.state == ContactState.WRITING
    
    def just_started_writing(self):
        return self.prev == ContactState.IDLE and self.state == ContactState.WRITING
    
    def just_stopped_writing(self):
        return self.prev == ContactState.WRITING and self.state == ContactState.IDLE
    
    def get_state_name(self):
        return self.state.value
    
    def reset(self):
        self.state = self.prev = ContactState.IDLE
        self.frames = 0
        self.dist_hist.clear()
        self.vel_hist.clear()
        self.pos_hist.clear()
        self.time_hist.clear()
        self.adaptive_buffer.clear()
        self._frames_since_update = 0
    
    def get_debug_info(self):
        return {
            'current_state': self.state.value,
            'previous_state': self.prev.value,
            'state_frames': self.frames,
            'distance_buffer': list(self.dist_hist),
            'velocity_buffer': list(self.vel_hist),
            'smoothed_distance': np.median(list(self.dist_hist)) if self.dist_hist else None,
            'smoothed_velocity': np.mean(list(self.vel_hist)) if self.vel_hist else None,
            'adaptive_threshold': self.writing_threshold,
            'adaptive_buffer_size': len(self.adaptive_buffer),
        }
