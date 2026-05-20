"""
手掌坐标系模块
管理手掌平面拟合、坐标系构建和坐标变换
"""

import numpy as np
from typing import Optional, Tuple

from src.utils.geometry_utils import get_landmark_3d


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

