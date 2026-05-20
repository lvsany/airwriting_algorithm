"""
几何工具模块（HAMCD 必需子集）
"""

import numpy as np


def get_landmark_3d(landmarks, index: int) -> np.ndarray:
    """
    获取单个地标的3D世界坐标。

    Args:
        landmarks: MediaPipe 手部地标
        index: 地标索引

    Returns:
        3D坐标 (3,)
    """
    lm = landmarks.landmark[index]
    return np.array([lm.x, lm.y, lm.z])
