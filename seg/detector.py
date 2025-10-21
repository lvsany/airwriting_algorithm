import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from seg.connect_scorer import ConnectSegmentScorer

@dataclass
class ConnSeg:
    i: int; j: int; score: float; geom: dict; gap_id: Tuple[int,int]

class ConnectStrokeDetector:
    def __init__(self, scorer: ConnectSegmentScorer, score_thr: float=0.3):
        self.scorer = scorer
        self.score_thr = score_thr

    def seeds_from_redlines(self, xy, redlines: List[dict]) -> List[Tuple[int,int]]:
        """
        返回所有“穿越红线”的点索引 (idx, lid)。若 redline.conf < τ_bdry，
        可不生成该线的seed，或后续降权。
        """
        seeds = []

        return seeds

    def grow_multiscale(self, xy, seed_idx, line_id, pca, redlines) -> List[ConnSeg]:
        """
        以 seed 为中心，双向多尺度生成候选段；对每个候选段：
        - 算几何指标（直线度/曲率/角度/长度）
        - 计算密度项（沿PC1的平均密度）
        - 融合边界conf
        得到 score。另记录所属“gap”（相邻红线对）。
        """

    def select_by_gap(self, cands: List[ConnSeg]) -> List[ConnSeg]:
        """
        同一gap内按score排序，做简单NMS，最多保留1个（或小DP）。
        """

    def detect(self, xy, pca: PcaPack, refined_redlines) -> List[ConnSeg]:
        """
        端到端：生成seeds -> 多尺度候选 -> gap内选择 -> 全局合并输出
        """
