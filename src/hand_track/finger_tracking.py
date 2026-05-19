import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from src.hand_track.hand_writing_detector import HandWritingDetector

# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

# 从配置文件读取参数
trajectory_config = CONFIG['trajectory']
detector = HandWritingDetector()  # 使用配置文件中的默认参数
trajectory = []
MAX_TRAJECTORY_POINTS = trajectory_config['max_trajectory_points']
prev_index_pos = None
WRITING_MOTION_THRESHOLD = trajectory_config['writing_motion_threshold']
JITTER_THRESHOLD = trajectory_config['jitter_threshold']
history_points = []  # 用于存储历史轨迹点以计算速度和曲率

def smart_smooth(current_pos, prev_pos, history_points=history_points, max_window=None, min_window=None, fps=60):
    """自适应平滑函数，根据速度和曲率动态调整窗口大小和权重"""
    # 从配置文件读取平滑参数
    smoothing_config = CONFIG['trajectory']['smoothing']
    if max_window is None:
        max_window = smoothing_config['max_window']
    if min_window is None:
        min_window = smoothing_config['min_window']
    
    if prev_pos is None:
        history_points.clear()  # 重置历史点
        history_points.append(current_pos)
        return current_pos

    # 添加当前点到历史缓冲区
    history_points.append(current_pos)
    if len(history_points) > max_window:
        history_points.pop(0)

    # 计算速度（像素/秒）
    distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    speed = distance * fps  # 假设每帧时间为 1/fps 秒

    # 计算曲率（基于三点法）
    curvature = 0
    if len(history_points) >= 3:
        p1 = np.array(history_points[-3])
        p2 = np.array(history_points[-2])
        p3 = np.array(history_points[-1])
        v1 = p2 - p1
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 > 0 and norm_v2 > 0:
            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
            curvature = 1 - cos_theta  # 曲率越高，值越大（范围 [0, 2]）

    # 根据速度和曲率动态调整窗口大小
    speed_threshold_low = smoothing_config['speed_threshold_low']
    speed_threshold_high = smoothing_config['speed_threshold_high']
    curvature_threshold = smoothing_config['curvature_threshold']
    
    if speed > speed_threshold_high or curvature > curvature_threshold:
        window_size = min_window  # 快速或高曲率区域用小窗口
    elif speed < speed_threshold_low and curvature < curvature_threshold:
        window_size = max_window  # 慢速且低曲率区域用大窗口
    else:
        # 线性插值确定窗口大小
        speed_factor = (speed - speed_threshold_low) / (speed_threshold_high - speed_threshold_low)
        curvature_factor = curvature / curvature_threshold
        factor = max(speed_factor, curvature_factor)
        factor = np.clip(factor, 0, 1)
        window_size = int(min_window + (max_window - min_window) * (1 - factor))

    # 确保窗口大小不超过历史点数量
    window_size = min(window_size, len(history_points))
    # return current_pos
    # 计算加权移动平均
    if window_size > 1:
        points = np.array(history_points[-window_size:])
        # 权重根据时间衰减（最近的点权重更高）
        weights = np.exp(np.linspace(-1, 0, window_size))  # 指数衰减权重
        weights /= np.sum(weights)  # 归一化
        smoothed_x = int(np.sum(points[:, 0] * weights))
        smoothed_y = int(np.sum(points[:, 1] * weights))
        return (smoothed_x, smoothed_y)
    else:
        return current_pos

