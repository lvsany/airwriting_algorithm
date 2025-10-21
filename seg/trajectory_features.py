# 计算轨迹的基础量

import cv2
import numpy as np

from utils.online_trace_converter import TracePoint

class TrajectoryFeatures:
    def __init__(self, trajectory: list[list[TracePoint]]):
        """
        初始化轨迹特征计算类
        
        Args:
            trajectory: 形状为(N, 2)的轨迹点数组
        """
        self.trajectory = trajectory
        self.xy = self.trace_to_array()

    def trace_to_array(self) -> np.ndarray:
        pts = []
        for p in self.trajectory:
            if p is None:
                continue
            if isinstance(p, (list, tuple, np.ndarray)):
                x, y = p[0], p[1]
            elif hasattr(p, 'x') and hasattr(p, 'y'):
                x, y = p.x, p.y
            else:
                # 如有其它属性名（例如 pos 或 to_tuple），按需扩展
                raise TypeError(f"不能识别的轨迹点类型: {p!r}")
            pts.append((float(x), float(y)))
        return np.array(pts, dtype=float)

    # 按弧长重采样轨迹点
    def resample_by_arclength(self, ds: float) -> np.ndarray:
        """
        按弧长重采样轨迹点，使得相邻点之间的距离约为ds
        
        Args:
            xy: 形状为(N, 2)的轨迹点数组
            ds: 目标点间距

        Returns:
            重采样后的轨迹点数组
        """
        # 计算累积弧长
        deltas = np.diff(self.xy, axis=0)
        dists = np.sqrt((deltas ** 2).sum(axis=1))  
        cumulative_length = np.insert(np.cumsum(dists), 0, 0)

        # 计算新的采样点位置
        num_points = int(cumulative_length[-1] / ds) + 1
        new_lengths = np.linspace(0, cumulative_length[-1], num_points)

        # 重采样轨迹点
        new_xy = np.zeros((num_points, 2))
        new_xy[:, 0] = np.interp(new_lengths, cumulative_length, xy[:, 0])
        new_xy[:, 1] = np.interp(new_lengths, cumulative_length, xy[:, 1])

        return new_xy

    def smooth_savgol(self, window_size:int, polyorder:int) -> np.ndarray:
        """
        使用Savitzky-Golay滤波器对轨迹点进行平滑处理
        
        Args:
            xy: 形状为(N, 2)的轨迹点数组
            window_size: 滑动窗口大小，必须为奇数
            polyorder: 多项式阶数，必须小于window_size
            
        Returns:
            平滑后的轨迹点数组
        """
        from scipy.signal import savgol_filter
        
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数
        
        smoothed_x = savgol_filter(self.xy[:, 0], window_size, polyorder)
        smoothed_y = savgol_filter(self.xy[:, 1], window_size, polyorder)
        
        smoothed_xy = np.vstack((smoothed_x, smoothed_y)).T

        return smoothed_xy
    
    # 计算直线度，平均曲率，角度等
    def calculate_features(self) -> dict:
        features = {}

        # straightness
        start_point = self.xy[0]
        end_point = self.xy[-1]
        # 计算弦长
        chord_length = np.linalg.norm(end_point - start_point)
        # 计算路径长度
        path_length = np.sum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))
        features['straightness'] = chord_length / path_length if path_length > 0 else 0.0


        # 平均曲率：单位弧长的转角
        v = np.diff(self.xy, axis=0)
        L = np.linalg.norm(v, axis=1) + 1e-9
        t = v / L[:, None]
        # 相邻切向量夹角
        cosang = np.clip(np.sum(t[:-1] * t[1:], axis=1), -1.0, 1.0)
        dtheta = np.arccos(cosang)  # [0,pi]
        # 局部弧长取相邻两段均值
        segL = (L[:-1] + L[1:]) * 0.5 + 1e-9
        features['mean_curvature'] = float(np.mean(dtheta / segL)) if len(segL) else 0.0

        # angles
        dy = end_point[1] - start_point[1]
        dx = end_point[0] - start_point[0]
        angle = np.degrees(np.arctan2(-dy, dx))
        features['angle'] = angle

        return features