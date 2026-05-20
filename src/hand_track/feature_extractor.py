"""
统一手部特征提取模块
每帧从左右手 landmarks 和灰度图中提取 10 维特征向量
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple

from src.utils.geometry_utils import get_landmark_3d
from src.hand_track.palm_coordinate_system import PalmLocalFrame


# 特征名称——顺序与 extract() 返回向量严格对齐，可直接用于日志列标题
FEATURE_NAMES: Tuple[str, ...] = (
    'dist2d_lm0',      # 0: 右手指尖 → 左手 lm0（手腕）2D 像素距离
    'dist2d_lm5',      # 1: 右手指尖 → 左手 lm5（食指MCP）2D 像素距离
    'dist2d_lm9',      # 2: 右手指尖 → 左手 lm9（中指MCP）2D 像素距离
    'dist2d_lm13',     # 3: 右手指尖 → 左手 lm13（无名指MCP）2D 像素距离
    'dist2d_lm17',     # 4: 右手指尖 → 左手 lm17（小指MCP）2D 像素距离
    'hull_iou',        # 5: 双手凸包重叠面积 / 并集面积
    'local_n',         # 6: 指尖在掌面局部坐标系下的 n 分量
    'roi_brightness',  # 7: 指尖 32×32 ROI 灰度均值
    'roi_shadow',      # 8: 指尖 32×32 ROI 拉普拉斯方差（越小越可能是阴影区）
    'approach_theta',  # 9: 指尖运动方向与掌面法向量的夹角（度，0°=垂直靠近）
)

_PALM_DIST_IDS = (0, 5, 9, 13, 17)  # dims 0-4 对应的左手关键点索引
_ROI_HALF      = 16                  # 32×32 ROI 的半边长（像素）
_APPROACH_WIN  = 4                   # 历史深度（需要 t 与 t-3 → 存 4 帧）
_STILL_PX      = 2.0                 # 像素位移低于此值视为静止


class HandFeatureExtractor:
    """
    逐帧提取 10 维手部接触特征向量。

    每次调用 extract() 返回 shape=(10,) 的 float64 数组，顺序见 FEATURE_NAMES。
    无法计算的维度填 np.nan；此处不做归一化或裁剪。

    内部维护一个最近 4 帧的指尖坐标缓存，用于计算接近角（dim 9）。
    手部消失时缓存自动清空，避免跨段误算。
    """

    def __init__(self):
        self._tip_px: deque = deque(maxlen=_APPROACH_WIN)  # (cx, cy) int
        self._tip_3d: deque = deque(maxlen=_APPROACH_WIN)  # np.ndarray (3,)

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def extract(
        self,
        write_lm,
        palm_lm,
        frame_gray: np.ndarray,
        palm_frame: Optional[PalmLocalFrame] = None,
    ) -> np.ndarray:
        """
        提取 10 维特征向量。

        Parameters
        ----------
        write_lm   : MediaPipe 右手（书写手）landmarks；None → 返回全 NaN。
        palm_lm    : MediaPipe 左手（画板手）landmarks；None → 返回全 NaN。
        frame_gray : 当前帧灰度图，shape (H, W)，dtype uint8。
        palm_frame : PalmLocalFrame 实例；None 时 dims 6/9 为 NaN。

        Returns
        -------
        np.ndarray, shape (10,), dtype float64
        """
        feat = np.full(10, np.nan)

        if write_lm is None or palm_lm is None:
            self._tip_px.clear()
            self._tip_3d.clear()
            return feat

        h, w = frame_gray.shape[:2]
        tip  = write_lm.landmark[8]
        cx   = int(tip.x * w)
        cy   = int(tip.y * h)

        # dims 0-4: 右手指尖到左手各关键点的 2D 像素欧氏距离
        for k, idx in enumerate(_PALM_DIST_IDS):
            p = palm_lm.landmark[idx]
            feat[k] = np.hypot(p.x * w - cx, p.y * h - cy)

        # dim 5: 双手凸包 IoU
        feat[5] = self._hull_iou(write_lm, palm_lm, w, h)

        # dim 6: 指尖在掌面局部坐标系中的 n 分量
        if palm_frame is not None and palm_frame.is_valid:
            tip_3d = get_landmark_3d(write_lm, 8)
            _, _, nc = palm_frame.to_local(tip_3d)
            feat[6] = nc

        # dims 7-8: 指尖 ROI 亮度 + 拉普拉斯方差
        roi = self._crop_roi(frame_gray, cx, cy)
        if roi.size:
            feat[7] = float(np.mean(roi))
            lap     = cv2.Laplacian(roi, cv2.CV_64F)
            feat[8] = float(np.var(lap))

        # dim 9: 接近角——先入队再计算，保证当前帧已在历史中
        tip_3d_now = get_landmark_3d(write_lm, 8)
        self._tip_px.append((cx, cy))
        self._tip_3d.append(tip_3d_now)
        feat[9] = self._approach_theta(palm_frame)

        return feat

    def reset(self):
        """清空历史缓存（角色切换或场景重置时调用）。"""
        self._tip_px.clear()
        self._tip_3d.clear()

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _hull_iou(write_lm, palm_lm, w: int, h: int) -> float:
        """计算双手凸包 intersection / union 面积比（IoU）。"""
        w_pts = np.array([[lm.x * w, lm.y * h] for lm in write_lm.landmark],
                         dtype=np.float32)
        p_pts = np.array([[lm.x * w, lm.y * h] for lm in palm_lm.landmark],
                         dtype=np.float32)
        hull_w = cv2.convexHull(w_pts)
        hull_p = cv2.convexHull(p_pts)
        inter_area, _ = cv2.intersectConvexConvex(hull_w, hull_p)
        area_w    = cv2.contourArea(hull_w)
        area_p    = cv2.contourArea(hull_p)
        union_area = area_w + area_p - float(inter_area)
        return float(inter_area / union_area) if union_area > 1e-6 else 0.0

    @staticmethod
    def _crop_roi(gray: np.ndarray, cx: int, cy: int) -> np.ndarray:
        """裁剪以 (cx,cy) 为中心的 32×32 ROI，边界自动裁剪。"""
        r  = _ROI_HALF
        y0 = max(cy - r, 0)
        y1 = min(cy + r, gray.shape[0])
        x0 = max(cx - r, 0)
        x1 = min(cx + r, gray.shape[1])
        return gray[y0:y1, x0:x1]

    def _approach_theta(self, palm_frame: Optional[PalmLocalFrame]) -> float:
        """
        计算指尖运动方向（t 相对 t-3）与掌面法向量的夹角（度）。

        - 历史不足 4 帧或坐标系无效 → NaN
        - 像素位移 < 2 px（静止）→ 90.0
        - 正常 → [0, 180] 度，0° 表示垂直靠近掌面
        """
        if palm_frame is None:
            return np.nan
        normal = palm_frame.get_normal()
        if normal is None or len(self._tip_3d) < _APPROACH_WIN:
            return np.nan

        px_now, px_old = self._tip_px[-1], self._tip_px[0]
        if np.hypot(px_now[0] - px_old[0], px_now[1] - px_old[1]) < _STILL_PX:
            return 90.0

        motion = self._tip_3d[-1] - self._tip_3d[0]
        m_norm = np.linalg.norm(motion)
        if m_norm < 1e-9:
            return 90.0
        motion    = motion / m_norm
        cos_theta = float(np.clip(np.dot(motion, normal), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))
