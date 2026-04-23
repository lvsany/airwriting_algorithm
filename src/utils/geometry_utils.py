"""
几何工具模块
提供3D平面拟合、坐标变换、投影等几何计算功能
"""

import numpy as np
from typing import List, Tuple, Optional
import random


def fit_plane_ransac(
    points: np.ndarray,
    n_iterations: int = 100,
    threshold: float = 0.01,
    min_inliers: int = 3
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    使用RANSAC算法拟合3D平面
    
    Args:
        points: (N, 3) 数组，N个3D点
        n_iterations: RANSAC迭代次数
        threshold: 内点距离阈值（米）
        min_inliers: 最小内点数量
        
    Returns:
        (normal, point): 平面法向量和平面上的一点，如果拟合失败返回None
    """
    if points.shape[0] < 3:
        return None
        
    best_normal = None
    best_point = None
    best_inliers = 0
    
    for _ in range(n_iterations):
        # 随机选择3个点
        idx = random.sample(range(points.shape[0]), 3)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        
        # 计算平面法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # 法向量归一化
        norm = np.linalg.norm(normal)
        if norm < 1e-6:  # 三点共线，跳过
            continue
        normal = normal / norm
        
        # 计算所有点到平面的距离
        distances = np.abs(np.dot(points - p1, normal))
        
        # 统计内点
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_point = p1
    
    if best_inliers < min_inliers:
        return None
        
    return best_normal, best_point


def build_coordinate_frame(
    origin: np.ndarray,
    normal: np.ndarray,
    reference_direction: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建局部坐标系
    
    Args:
        origin: 坐标系原点 (3,)
        normal: Z轴方向（平面法向量） (3,)
        reference_direction: 参考方向，用于确定X轴 (3,)
        
    Returns:
        (u_axis, v_axis, w_axis): X, Y, Z轴单位向量
    """
    # 确保法向量归一化
    w_axis = normal / np.linalg.norm(normal)
    
    # 如果没有提供参考方向，使用默认方向
    if reference_direction is None:
        # 选择一个不与法向量平行的向量
        if abs(w_axis[0]) < 0.9:
            reference_direction = np.array([1.0, 0.0, 0.0])
        else:
            reference_direction = np.array([0.0, 1.0, 0.0])
    
    # 计算U轴（通过叉乘获得垂直于法向量的方向）
    u_axis = np.cross(reference_direction, w_axis)
    u_norm = np.linalg.norm(u_axis)
    if u_norm < 1e-6:
        # 如果参考方向与法向量平行，换一个方向
        reference_direction = np.array([0.0, 0.0, 1.0])
        u_axis = np.cross(reference_direction, w_axis)
        u_norm = np.linalg.norm(u_axis)
    u_axis = u_axis / u_norm
    
    # 计算V轴（右手坐标系）
    v_axis = np.cross(w_axis, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)
    
    return u_axis, v_axis, w_axis


def build_transform_matrix(
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    w_axis: np.ndarray
) -> np.ndarray:
    """
    构建从世界坐标系到局部坐标系的变换矩阵
    
    Args:
        origin: 坐标系原点 (3,)
        u_axis, v_axis, w_axis: 局部坐标系的三个轴 (3,)
        
    Returns:
        4x4齐次变换矩阵
    """
    # 旋转矩阵（将世界坐标系的xyz轴对齐到局部坐标系的uvw轴）
    R = np.column_stack([u_axis, v_axis, w_axis])
    
    # 构建齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = R.T  # 转置因为我们要从世界到局部
    T[:3, 3] = -R.T @ origin  # 平移部分
    
    return T


def transform_point(point: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    使用变换矩阵变换一个点
    
    Args:
        point: 3D点 (3,)
        transform_matrix: 4x4变换矩阵
        
    Returns:
        变换后的点 (3,)
    """
    # 转换为齐次坐标
    point_homo = np.append(point, 1.0)
    
    # 应用变换
    transformed = transform_matrix @ point_homo
    
    # 返回3D坐标
    return transformed[:3]


def project_point_to_plane(
    point: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray
) -> np.ndarray:
    """
    将点投影到平面上
    
    Args:
        point: 3D点 (3,)
        plane_origin: 平面上的一点 (3,)
        plane_normal: 平面法向量（归一化） (3,)
        
    Returns:
        投影后的点 (3,)
    """
    # 计算点到平面的距离
    distance = np.dot(point - plane_origin, plane_normal)
    
    # 沿法向量移动到平面上
    projected = point - distance * plane_normal
    
    return projected


def point_to_plane_distance(
    point: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray
) -> float:
    """
    计算点到平面的带符号距离
    
    Args:
        point: 3D点 (3,)
        plane_origin: 平面上的一点 (3,)
        plane_normal: 平面法向量（归一化） (3,)
        
    Returns:
        带符号距离（正值表示在法向量方向）
    """
    return np.dot(point - plane_origin, plane_normal)


def extract_landmarks_3d(landmarks, indices: List[int]) -> np.ndarray:
    """
    从MediaPipe landmarks中提取指定索引的3D坐标
    
    Args:
        landmarks: MediaPipe手部地标
        indices: 地标索引列表
        
    Returns:
        (N, 3) 数组，N个3D点的世界坐标
    """
    points = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        points.append([lm.x, lm.y, lm.z])
    return np.array(points)


def get_landmark_3d(landmarks, index: int) -> np.ndarray:
    """
    获取单个地标的3D世界坐标
    
    Args:
        landmarks: MediaPipe手部地标
        index: 地标索引
        
    Returns:
        3D坐标 (3,)
    """
    lm = landmarks.landmark[index]
    return np.array([lm.x, lm.y, lm.z])


def calculate_palm_reference_direction(landmarks) -> np.ndarray:
    """
    计算手掌的参考方向（从手腕指向中指根部）
    用于确定手掌坐标系的X轴方向
    
    Args:
        landmarks: MediaPipe手部地标
        
    Returns:
        方向向量 (3,)
    """
    wrist = get_landmark_3d(landmarks, 0)
    middle_base = get_landmark_3d(landmarks, 9)
    
    direction = middle_base - wrist
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0])
    
    return direction / norm


def smooth_plane_parameters(
    current_normal: np.ndarray,
    current_origin: np.ndarray,
    previous_normal: Optional[np.ndarray],
    previous_origin: Optional[np.ndarray],
    alpha: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    平滑平面参数（指数移动平均）
    
    Args:
        current_normal: 当前帧的平面法向量
        current_origin: 当前帧的平面原点
        previous_normal: 上一帧的平面法向量
        previous_origin: 上一帧的平面原点
        alpha: 平滑系数 (0-1)，越大越接近当前值
        
    Returns:
        (smoothed_normal, smoothed_origin): 平滑后的参数
    """
    if previous_normal is None or previous_origin is None:
        return current_normal, current_origin
    
    # 确保法向量方向一致（避免180度翻转）
    if np.dot(current_normal, previous_normal) < 0:
        current_normal = -current_normal
    
    # 指数移动平均
    smoothed_normal = alpha * current_normal + (1 - alpha) * previous_normal
    smoothed_normal = smoothed_normal / np.linalg.norm(smoothed_normal)
    
    smoothed_origin = alpha * current_origin + (1 - alpha) * previous_origin
    
    return smoothed_normal, smoothed_origin
