import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from seg.trajectory_features import TrajectoryFeatures
from .type import PcaPack
from seg.connect_scorer import ConnectSegmentScorer

@dataclass
class ConnSeg:
    i: int; j: int; score: float; geom: dict; gap_id: Tuple[int,int]

class ConnectStrokeDetector:
    def __init__(self, scorer: ConnectSegmentScorer, score_thr: float=0.3, 
                 angle_max_deg: float=15, len_min_px: float=10.0, len_max_px: float=200.0):
        self.scorer = scorer
        self.score_thr = score_thr
        self.angle_max_deg = angle_max_deg
        self.len_min_px = len_min_px
        self.len_max_px = len_max_px

    # 点到直线的有向距离
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        # 计算直线的方向向量
        dir_vec = line_end - line_start
        
        # 计算单位法向量（逆时针旋转90°）
        n = np.array([-dir_vec[1], dir_vec[0]])
        n = n / np.linalg.norm(n)  # 归一化
        
        # 计算有向距离
        AP = point - line_start
        distance = AP @ n
        return distance


    def seeds_from_redlines(self, xy: np.ndarray, boundary_lines: List[dict]) -> List[Tuple[int,int]]:
        """
        返回所有“穿越红线”的点索引 (idx, lid)
        """
        seeds = []
        for line in boundary_lines:
            # Extract line properties
            line_start = np.array(line['start'])
            line_end = np.array(line['end'])
            
            s = np.array([self._point_to_line_distance(pt, line_start, line_end) for pt in xy])
            sign_changes = np.where(np.diff(np.sign(s)))[0]
            for idx in sign_changes.tolist():
                seeds.append((idx, line['index']))
        return seeds
    

    def _segment_ok(self, metrics: dict) -> bool:
        # 硬约束：角度要够“斜上”，长度在合理范围
        if metrics["angle"] > self.angle_max_deg:
            return False
        if not (self.len_min_px <= metrics["length"] <= self.len_max_px):
            return False
        return True

    def get_gap_id(self, line_id: int, boundary_lines: List[dict]) -> Tuple[int,int]:
        """
        根据 line_id 获取所属 gap_id（相邻红线对的索引）
        """
        line_indices = [line['index'] for line in boundary_lines]
        line_indices.sort()
        idx = line_indices.index(line_id)
        if idx == 0:
            return (-1, line_indices[0])
        elif idx == len(line_indices) - 1:
            return (line_indices[-1], -1)
        else:
            return (line_indices[idx - 1], line_indices[idx + 1])
    def grow_multiscale(self, xy: np.ndarray, seed_idx: int, line_id: int, pca: PcaPack, boundary_lines: List[dict]) -> List[ConnSeg]:
        """
        以 seed 为中心，双向多尺度生成候选段；对每个候选段：
        - 算几何指标（直线度/曲率/角度/长度）
        - 计算密度项（沿PC1的平均密度）
        得到 score。另记录所属“gap”（相邻红线对）。
        """
        conn_segs = []
        n_points = xy.shape[0]
        scales = [5, 10, 15, 20, 25]  # 多尺度长度（点数）
        
        for scale in scales:
            for direction in [-1, 1]:  # 向前和向后生长
                if direction == -1:
                    start_idx = max(0, seed_idx - scale)
                    end_idx = seed_idx
                else:
                    start_idx = seed_idx
                    end_idx = min(n_points - 1, seed_idx + scale)
                
                if end_idx - start_idx < 2:
                    continue  # 段太短，跳过
                
                seg_points = xy[start_idx:end_idx+1]
                
                trajectory_features = TrajectoryFeatures(trajectory=None, xy=seg_points)
                # 计算几何指标
                geom_metrics = trajectory_features.calculate_features()

                if not self._segment_ok(geom_metrics):
                    continue  # 不满足硬约束，跳过
                # 计算综合得分
                score = self.scorer.score_geom(
                    straight=geom_metrics['straightness'],
                    mean_curv=geom_metrics['mean_curvature'],
                    angle_deg=geom_metrics['angle']
                )
                
                # 获取所属gap_id
                gap_id = self.get_gap_id(line_id, boundary_lines)

                
                conn_seg = ConnSeg(
                    i=start_idx,
                    j=end_idx,
                    score=score,
                    geom=geom_metrics,
                    gap_id=gap_id
                )
                
                conn_segs.append(conn_seg)
        
        return conn_segs


    def select_by_gap(self, cands: List[ConnSeg]) -> List[ConnSeg]:
        """
        同一gap内按score排序，做简单NMS，最多保留1个（或小DP）。
        """
        groups: Dict[Tuple[int,int], List[ConnSeg]] = {}
        for c in cands:
            if c.score < self.score_thr:
                continue
            groups.setdefault(c.gap_id, []).append(c)

        picked: List[ConnSeg] = []
        for gid, arr in groups.items():
            arr.sort(key=lambda x: x.score, reverse=True)
            keep: List[ConnSeg] = []
            for c in arr:
                ok = True
                for k in keep:
                    if self._iou_1d((c.i,c.j), (k.i,k.j)) > 0.5:
                        ok = False; break
                if ok:
                    keep.append(c)
            # 每 gap 只取分数最高的一个
            if keep:
                picked.append(keep[0])
        return picked


    def detect(self, xy:np.ndarray, pca: PcaPack, boundary_lines: List[dict]) -> List[ConnSeg]:
        """
        端到端：生成seeds -> 多尺度候选 -> gap内选择 -> 全局合并输出
        """
        seeds = self.seeds_from_redlines(xy, boundary_lines)
        all_cands: List[ConnSeg] = []
        for (idx, lid) in seeds:
            cands = self.grow_multiscale(xy, idx, lid, pca, boundary_lines)
            all_cands.extend(cands)
        
        picked_segs = self.select_by_gap(all_cands)
        return picked_segs