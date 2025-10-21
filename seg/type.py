
from attr import dataclass
import numpy as np
from typing import List

@dataclass
class PcaPack:
    mean: np.ndarray             # (2,)
    pc1_axis: np.ndarray         # (2,) 单位向量
    pc1_normal: np.ndarray       # (2,) 单位向量（与 pc1_axis 垂直）
    density_bins: np.ndarray     # (B,) PC1 标量坐标（bin 中心）
    density_profile: np.ndarray  # (B,) 每个 bin 的采样密度
    red_lines: List[dict]     # 初步或精炼后的红线