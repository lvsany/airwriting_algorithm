"""
手掌坐标系模块
管理手掌平面拟合、坐标系构建和坐标变换
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import time

from utils.geometry_utils import (
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
