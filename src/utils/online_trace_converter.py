import cv2
import numpy as np
import time
import yaml
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

@dataclass
class TracePoint:
    """在线轨迹点数据结构"""
    x: int
    y: int
    timestamp: float
    velocity: float = 0.0  # 像素/秒
    curvature: float = 0.0  # 曲率（1/像素）
    pressure: float = 1.0  # 压力值（预留）
    
class OnlineTraceConverter:
    """在线轨迹转换器 - 将轨迹转换为带有时间和速度信息的在线轨迹"""
    
    def __init__(self, smoothing_window: int = None):
        """
        初始化在线轨迹转换器
        
        Args:
            smoothing_window: 速度平滑窗口大小，如果为None则从配置文件读取
        """
        if smoothing_window is None:
            smoothing_window = CONFIG['online_trace']['smoothing_window']
        self.smoothing_window = smoothing_window
        self.current_trace: List[TracePoint] = []
        self.completed_traces: List[List[TracePoint]] = []
        self.last_point_time = None
        self.velocity_history: List[float] = []
        
    def add_point(self, x: int, y: int, timestamp: Optional[float] = None) -> TracePoint:
        """
        添加轨迹点并计算速度
        
        Args:
            x: x坐标
            y: y坐标
            timestamp: 时间戳（如果为None则使用当前时间）
            
        Returns:
            添加的轨迹点
        """
        if timestamp is None:
            timestamp = time.time()
            
        velocity = 0.0
        if len(self.current_trace) > 0:
            prev_point = self.current_trace[-1]
            distance = np.sqrt((x - prev_point.x)**2 + (y - prev_point.y)**2)
            time_diff = timestamp - prev_point.timestamp
            
            if time_diff > 0:
                velocity = distance / time_diff
                # 添加到速度历史中进行平滑
                self.velocity_history.append(velocity)
                if len(self.velocity_history) > self.smoothing_window:
                    self.velocity_history.pop(0)
                # 使用平滑后的速度
                velocity = np.mean(self.velocity_history)
        
        point = TracePoint(x=x, y=y, timestamp=timestamp, velocity=velocity, curvature=0.0)
        self.current_trace.append(point)
        self.last_point_time = timestamp
        
        # 计算并更新最近几个点的曲率
        self._update_curvature()
        
        return point
    
    def end_trace(self) -> List[TracePoint]:
        """
        结束当前轨迹并返回完整的轨迹
        
        Returns:
            完成的轨迹点列表
        """
        if len(self.current_trace) > 0:
            # 完成轨迹前重新计算所有点的曲率
            self._calculate_all_curvatures(self.current_trace)
            completed_trace = self.current_trace.copy()
            self.completed_traces.append(completed_trace)
            self.current_trace.clear()
            self.velocity_history.clear()
            return completed_trace
        return []
    
    def get_current_trace(self) -> List[TracePoint]:
        """获取当前正在绘制的轨迹"""
        return self.current_trace.copy()
    
    def get_all_traces(self) -> List[List[TracePoint]]:
        """获取所有完成的轨迹"""
        return self.completed_traces.copy()
    
    def clear_traces(self):
        """清除所有轨迹"""
        self.current_trace.clear()
        self.completed_traces.clear()
        self.velocity_history.clear()
        self.last_point_time = None
    
    def get_velocity_stats(self) -> Dict[str, float]:
        """
        获取当前轨迹的速度统计信息
        
        Returns:
            包含速度统计的字典
        """
        if len(self.current_trace) == 0:
            return {'current': 0.0, 'average': 0.0, 'max': 0.0, 'min': 0.0}
        
        velocities = [point.velocity for point in self.current_trace if point.velocity > 0]
        if len(velocities) == 0:
            return {'current': 0.0, 'average': 0.0, 'max': 0.0, 'min': 0.0}
        
        return {
            'current': self.current_trace[-1].velocity,
            'average': np.mean(velocities),
            'max': np.max(velocities),
            'min': np.min(velocities)
        }
    
    @staticmethod
    def convert_legacy_trajectory(trajectory: List[Tuple[int, int]], 
                                fps: float = 30.0) -> List[TracePoint]:
        """
        将传统轨迹格式转换为在线轨迹格式
        
        Args:
            trajectory: 传统轨迹点列表 [(x, y), ...]
            fps: 假设的帧率用于计算时间戳
            
        Returns:
            在线轨迹点列表
        """
        online_trace = []
        base_time = time.time()
        frame_duration = 1.0 / fps
        
        for i, (x, y) in enumerate(trajectory):
            if x is None or y is None:
                continue
                
            timestamp = base_time + i * frame_duration
            velocity = 0.0
            
            if i > 0 and len(online_trace) > 0:
                prev_point = online_trace[-1]
                distance = np.sqrt((x - prev_point.x)**2 + (y - prev_point.y)**2)
                time_diff = timestamp - prev_point.timestamp
                if time_diff > 0:
                    velocity = distance / time_diff
            
            point = TracePoint(x=x, y=y, timestamp=timestamp, velocity=velocity)
            online_trace.append(point)
        
        return online_trace
    
    @staticmethod
    def get_velocity_color(velocity: float, max_velocity: float = 500.0) -> Tuple[int, int, int]:
        """
        根据速度获取对应的颜色（BGR格式）
        
        Args:
            velocity: 当前速度
            max_velocity: 最大速度用于归一化
            
        Returns:
            BGR颜色元组
        """
        # 将速度归一化到0-1范围
        normalized_velocity = min(velocity / max_velocity, 1.0)
        
        # 使用颜色映射：蓝色(慢) -> 绿色(中) -> 红色(快)
        if normalized_velocity < 0.5:
            # 蓝色到绿色
            blue = int(255 * (1 - 2 * normalized_velocity))
            green = int(255 * 2 * normalized_velocity)
            red = 0
        else:
            # 绿色到红色
            blue = 0
            green = int(255 * (2 - 2 * normalized_velocity))
            red = int(255 * (2 * normalized_velocity - 1))
        
        return (blue, green, red)
    
    @staticmethod
    def draw_velocity_trace(frame: np.ndarray, trace: List[TracePoint], 
                          thickness: int = 2, show_points: bool = True) -> np.ndarray:
        """
        在帧上绘制带速度颜色的轨迹
        
        Args:
            frame: 要绘制的帧
            trace: 轨迹点列表
            thickness: 线条粗细
            show_points: 是否显示轨迹点
            
        Returns:
            绘制后的帧
        """
        if len(trace) < 2:
            return frame
        
        # 计算最大速度用于颜色映射
        velocities = [point.velocity for point in trace if point.velocity > 0]
        max_velocity = max(velocities) if velocities else 100.0
        
        # 绘制轨迹线段
        for i in range(1, len(trace)):
            pt1 = (trace[i-1].x, trace[i-1].y)
            pt2 = (trace[i].x, trace[i].y)
            color = OnlineTraceConverter.get_velocity_color(trace[i].velocity, max_velocity)
            cv2.line(frame, pt1, pt2, color, thickness)
            
            if show_points:
                cv2.circle(frame, pt2, 3, color, -1)
        
        return frame
    
    @staticmethod
    def draw_velocity_info(frame: np.ndarray, stats: Dict[str, float], 
                         position: Tuple[int, int] = (20, 120)) -> np.ndarray:
        """
        在帧上绘制速度信息
        
        Args:
            frame: 要绘制的帧
            stats: 速度统计信息
            position: 文本位置
            
        Returns:
            绘制后的帧
        """
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        texts = [
            f"Current Speed: {stats['current']:.1f} px/s",
            f"Average Speed: {stats['average']:.1f} px/s",
            f"Max Speed: {stats['max']:.1f} px/s"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (x, y + i * 25), font, font_scale, color, thickness)
        
        return frame
    
    def _update_curvature(self):
        """更新当前轨迹最近几个点的曲率"""
        if len(self.current_trace) < 3:
            return
        
        # 更新最后几个点的曲率
        start_idx = max(0, len(self.current_trace) - 5)
        for i in range(start_idx, len(self.current_trace)):
            self.current_trace[i].curvature = self._calculate_curvature_at_point(self.current_trace, i)
    
    def _calculate_all_curvatures(self, trace: List[TracePoint]):
        """计算轨迹中所有点的曲率"""
        for i in range(len(trace)):
            trace[i].curvature = self._calculate_curvature_at_point(trace, i)
    
    def _calculate_curvature_at_point(self, trace: List[TracePoint], index: int) -> float:
        """
        计算指定点的曲率
        
        Args:
            trace: 轨迹点列表
            index: 要计算曲率的点索引
            
        Returns:
            曲率值（1/像素）
        """
        if len(trace) < 3 or index < 1 or index >= len(trace) - 1:
            return 0.0
        
        # 获取相邻三个点
        p1 = trace[index - 1]
        p2 = trace[index]
        p3 = trace[index + 1]
        
        # 计算向量
        v1 = np.array([p2.x - p1.x, p2.y - p1.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        # 避免除零错误
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 1e-6 or len_v2 < 1e-6:
            return 0.0
        
        # 计算夹角（使用向量叉积）
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        dot_product = np.dot(v1, v2)
        
        # 计算曲率 κ = |dθ/ds|
        # 其中 sin(θ) = cross_product / (|v1| * |v2|)
        # ds ≈ (|v1| + |v2|) / 2
        
        sin_theta = cross_product / (len_v1 * len_v2)
        ds = (len_v1 + len_v2) / 2.0
        
        if ds < 1e-6:
            return 0.0
        
        # 曲率的绝对值
        curvature = abs(sin_theta) / ds
        
        return curvature
    
    def get_curvature_stats(self) -> Dict[str, float]:
        """
        获取当前轨迹的曲率统计信息
        
        Returns:
            包含曲率统计的字典
        """
        if len(self.current_trace) == 0:
            return {'current': 0.0, 'average': 0.0, 'max': 0.0, 'min': 0.0}
        
        curvatures = [point.curvature for point in self.current_trace if point.curvature > 0]
        if len(curvatures) == 0:
            return {'current': 0.0, 'average': 0.0, 'max': 0.0, 'min': 0.0}
        
        return {
            'current': self.current_trace[-1].curvature,
            'average': np.mean(curvatures),
            'max': np.max(curvatures),
            'min': np.min(curvatures)
        }
    
    @staticmethod
    def draw_curvature_info(frame: np.ndarray, stats: Dict[str, float], 
                          position: Tuple[int, int] = (20, 300)) -> np.ndarray:
        """
        在帧上绘制曲率信息
        
        Args:
            frame: 要绘制的帧
            stats: 曲率统计信息
            position: 文本位置
            
        Returns:
            绘制后的帧
        """
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        texts = [
            f"Current Curvature: {stats['current']:.4f} 1/px",
            f"Average Curvature: {stats['average']:.4f} 1/px",
            f"Max Curvature: {stats['max']:.4f} 1/px"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (x, y + i * 25), font, font_scale, color, thickness)
        
        return frame
