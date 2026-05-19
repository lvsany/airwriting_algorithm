"""
手掌坐标系模块
管理手掌平面拟合、坐标系构建和坐标变换
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import time

from src.utils.geometry_utils import (
    fit_plane_ransac,
    build_coordinate_frame,
    build_transform_matrix,
    transform_point,
    project_point_to_plane,
    point_to_plane_distance,
    extract_landmarks_3d,
    get_landmark_3d,
    calculate_palm_reference_direction
)


class PalmLocalFrame:
    """
    掌面局部坐标系（基于关键点直接构建，左手/画板手专用）

    坐标系定义（逐帧更新）:
        O = landmark 0 (手腕)
        u = normalize(lm5 - lm17)     食指MCP → 小指MCP
        v_raw = normalize(lm9 - lm0)  手腕 → 中指MCP
        n = normalize(cross(u, v_raw))
        v = normalize(cross(n, u))    正交化 v

    关键点缺失或几何退化时保持上一帧坐标系不变。
    MediaPipe z 轴深度精度有限，n 分量仅用于方向判断。
    """

    def __init__(self):
        self._origin: Optional[np.ndarray] = None
        self._u:      Optional[np.ndarray] = None
        self._v:      Optional[np.ndarray] = None
        self._n:      Optional[np.ndarray] = None

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def update(self, lm) -> bool:
        """
        用当前帧左手 landmarks 更新坐标系。
        几何退化（向量共线或长度为零）时保留上一帧结果，返回 False。
        """
        try:
            o   = get_landmark_3d(lm, 0)
            p5  = get_landmark_3d(lm, 5)
            p9  = get_landmark_3d(lm, 9)
            p17 = get_landmark_3d(lm, 17)
        except (IndexError, AttributeError):
            return False

        u = p5 - p17
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            return False
        u = u / u_norm

        v_raw = p9 - o
        v_norm = np.linalg.norm(v_raw)
        if v_norm < 1e-6:
            return False
        v_raw = v_raw / v_norm

        n = np.cross(u, v_raw)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            return False
        n = n / n_norm

        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        self._origin = o
        self._u = u
        self._v = v
        self._n = n
        return True

    def to_local(self, point: np.ndarray) -> Tuple[float, float, float]:
        """
        将全局坐标点投影到掌面局部坐标系，返回 (u分量, v分量, n分量)。
        坐标系未初始化时返回 (0.0, 0.0, 0.0)。
        """
        if self._origin is None:
            return (0.0, 0.0, 0.0)
        d = point - self._origin
        return (float(np.dot(d, self._u)),
                float(np.dot(d, self._v)),
                float(np.dot(d, self._n)))

    def get_normal(self) -> Optional[np.ndarray]:
        """返回当前法向量副本；未初始化返回 None。"""
        return self._n.copy() if self._n is not None else None

    @property
    def is_valid(self) -> bool:
        return self._origin is not None

    def reset(self):
        self._origin = self._u = self._v = self._n = None

    @property
    def origin(self) -> Optional[np.ndarray]:
        return self._origin.copy() if self._origin is not None else None

    @property
    def u_axis(self) -> Optional[np.ndarray]:
        return self._u.copy() if self._u is not None else None

    @property
    def v_axis(self) -> Optional[np.ndarray]:
        return self._v.copy() if self._v is not None else None

    @property
    def n_axis(self) -> Optional[np.ndarray]:
        return self._n.copy() if self._n is not None else None


@dataclass
class PalmCoordinateSystem:
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    w_axis: np.ndarray
    plane_normal: np.ndarray
    plane_origin: np.ndarray
    world_to_local: np.ndarray
    timestamp: float
    confidence: float
    n_inliers: int
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0

    def transform_to_local(self, point_world: np.ndarray) -> np.ndarray:
        return transform_point(point_world, self.world_to_local)

    def get_distance_to_plane(self, point_world: np.ndarray) -> float:
        return point_to_plane_distance(point_world, self.plane_origin, self.plane_normal) * 1000.0

    def project_to_plane(self, point_world: np.ndarray) -> np.ndarray:
        return project_point_to_plane(point_world, self.plane_origin, self.plane_normal)

    def get_2d_coordinates(self, point_world: np.ndarray) -> Tuple[float, float, float]:
        local = self.transform_to_local(point_world)
        return local[0], local[1], local[2] * 1000.0

    def is_within_palm_boundary(self, point_world: np.ndarray) -> bool:
        x, y = self.transform_to_local(point_world)[:2]
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class PalmPlaneTracker:
    def __init__(self, config: dict):
        pc = config.get('palm_writing', {}).get('plane_fitting', {})
        self.ransac_iters = pc.get('ransac_iterations', 100)
        self.ransac_thresh = pc.get('ransac_threshold', 0.01)
        self.ref_lm = pc.get('reference_landmarks', [0, 5, 17])
        self.origin_lm = pc.get('origin_landmark', 0)
        self.current = None

    def update(self, lm):
        if lm is None:
            self.current = None
            return None
        self.current = self._fit(lm)
        return self.current

    def _fit(self, lm):
        try:
            pts = extract_landmarks_3d(lm, self.ref_lm)
        except:
            return None

        res = fit_plane_ransac(pts, self.ransac_iters, self.ransac_thresh, 2)
        if not res:
            return None

        n, po = res
        o = get_landmark_3d(lm, self.origin_lm)
        ref = calculate_palm_reference_direction(lm)
        u, v, w = build_coordinate_frame(o, n, ref)
        T = build_transform_matrix(o, u, v, w)

        n_in = np.sum(np.abs(np.dot(pts - po, n)) < self.ransac_thresh)

        all_pts = extract_landmarks_3d(lm, list(range(21)))
        local = np.array([transform_point(p, T) for p in all_pts])
        m = 0.005
        xmin, xmax = local[:, 0].min() - m, local[:, 0].max() + m
        ymin, ymax = local[:, 1].min() - m, local[:, 1].max() + m

        return PalmCoordinateSystem(o, u, v, w, n, po, T, time.time(), 1.0, n_in, xmin, xmax, ymin, ymax)

    def get_current_system(self):
        return self.current

    def reset(self):
        self.current = None

    def get_debug_info(self):
        if not self.current:
            return {'valid': False}
        return {'valid': True, 'n_inliers': self.current.n_inliers,
                'origin': self.current.origin.tolist(), 'normal': self.current.plane_normal.tolist()}
