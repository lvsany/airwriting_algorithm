import math
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
                 angle_max_deg: float=15, len_min_px: float=10.0,
                 score_tie_eps: float=0.05):
        self.scorer = scorer
        self.score_thr = score_thr
        self.angle_max_deg = angle_max_deg
        self.len_min_px = len_min_px
        # 当两个候选分数差距小于该阈值时，使用直线度作为决胜（越大优先）
        self.score_tie_eps = score_tie_eps

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
            print(f"Segment rejected due to angle: {metrics['angle']} > {self.angle_max_deg}")
            return False
        if not (self.len_min_px <= metrics["length"]):
            print(f"Segment rejected due to length: {metrics['length']} not <= [{self.len_min_px}]")
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
        得到 score。另记录所属“gap”（相邻红线对）。
        """
        conn_segs = []
        n_points = xy.shape[0]
        scales = [3,6,9,12,15,18,21,24,27,30,33,36,39]  # 多尺度长度（点数）
        
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
                
                trajectory_features = TrajectoryFeatures(trajectory=None, xy=seg_points, pca=pca)
                # 计算几何指标
                geom_metrics = trajectory_features.calculate_features()

                if not self._segment_ok(geom_metrics):
                    continue  # 不满足硬约束，跳过
                # 计算综合得分（使用数值稳定的 sigmoid 避免 math.exp 溢出）
                raw_score = float(self.scorer.score_segment(
                    metrics=geom_metrics,
                    seg_s=seg_points,
                    pca=pca
                ))

                # 调试输出：当 raw_score 非常大或非常小时打印详细信息，帮助定位为何 sigmoid 饱和
                try:
                    if abs(raw_score) > 5.0:  # 阈值可调
                        # 从 scorer 中单独计算几项以便定位（如果 scorer 有相应方法）
                        # 这里尽量打印可用信息：raw_score 与 geom 指标
                        print(f"[DEBUG SCORE] idx=[{start_idx},{end_idx}] raw_score={raw_score:.3f} straight={geom_metrics.get('straightness'):.3f} mean_curv={geom_metrics.get('mean_curvature'):.4f} angle={geom_metrics.get('angle'):.2f}")
                except Exception:
                    pass

                # 数值稳定的 sigmoid：对正负分支分别处理，避免对非常大/小的 x 计算 exp(大数)
                def _stable_sigmoid(x: float) -> float:
                    if x >= 0.0:
                        z = math.exp(-x)
                        return 1.0 / (1.0 + z)
                    else:
                        z = math.exp(x)
                        return z / (1.0 + z)

                score = _stable_sigmoid(raw_score)

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
    @staticmethod
    def _iou_1d(a: Tuple[int,int], b: Tuple[int,int]) -> float:
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = (a[1]-a[0]) + (b[1]-b[0]) - inter + 1e-9
        return float(inter / union)

    def select_by_gap(self, cands: List[ConnSeg], boundary_lines: List[dict]=None) -> List[ConnSeg]:
        """
        同一gap内按score排序，做简单NMS，最多保留1个（或小DP）。
        """
        groups: Dict[Tuple[int,int], List[ConnSeg]] = {}
        for c in cands:
            # print(c.geom['straightness']+", "+c.geom['mean_curvature']+", "+str(c.geom['angle'])+", "+str(c.score))
            if c.score < self.score_thr:
                continue
            groups.setdefault(c.gap_id, []).append(c)

        # If boundary_lines provided, compute expected gaps and print debug info for empty ones
        if boundary_lines is not None:
            try:
                line_indices = [line['index'] for line in boundary_lines]
                line_indices = sorted(line_indices)
                expected_gaps = []
                if len(line_indices) > 0:
                    # leftmost gap (before first)
                    expected_gaps.append((-1, line_indices[0]))
                    # inner adjacent gaps
                    for i in range(len(line_indices)-1):
                        expected_gaps.append((line_indices[i], line_indices[i+1]))
                    # rightmost gap (after last)
                    expected_gaps.append((line_indices[-1], -1))
                for eg in expected_gaps:
                    if eg not in groups:
                        # try to print more context if available
                        bd_info = None
                        # find centers for the two boundary lines if present
                        try:
                            if eg[0] != -1:
                                left = next((l for l in boundary_lines if l['index'] == eg[0]), None)
                            else:
                                left = None
                            if eg[1] != -1:
                                right = next((l for l in boundary_lines if l['index'] == eg[1]), None)
                            else:
                                right = None
                            bd_info = (left['center'] if left is not None else None,
                                       right['center'] if right is not None else None)
                        except Exception:
                            bd_info = None
                        print(f"[DEBUG] gap {eg} has no candidates. centers={bd_info}")
            except Exception:
                pass
        picked: List[ConnSeg] = []
        for gid, arr in groups.items():
            arr.sort(key=lambda x: x.score, reverse=True)
            keep: List[ConnSeg] = []
            for c in arr:
                ok = True
                # 遍历当前保留的候选（复制一份以便可能替换）
                for k in keep[:]:
                    if self._iou_1d((c.i,c.j), (k.i,k.j)) > 0.5:
                        # 若分数接近（平局），以直线度较大的为优
                        try:
                            score_diff = c.score - k.score
                            if abs(score_diff) <= self.score_tie_eps:
                                c_straight = float(c.geom.get('straightness', 0.0) or 0.0)
                                k_straight = float(k.geom.get('straightness', 0.0) or 0.0)
                                if c_straight > k_straight:
                                    # 替换掉已有的 k，允许继续检查是否与其他 kept 重叠
                                    keep.remove(k)
                                    # 不将 ok 设为 False，这样最后会把 c 加入
                                    continue
                                else:
                                    ok = False
                                    break
                            else:
                                # 分数差距较大，保留分数高的那一个（由于 arr 已按 score 降序，若到这里代表 c.score <= k.score）
                                ok = False
                                break
                        except Exception:
                            ok = False
                            break
                if ok:
                    keep.append(c)
                    # safe debug print: convert numeric values to string to avoid numpy ufunc errors
                    try:
                        print(f"cand gap={gid}: straight={c.geom.get('straightness')}, mean_curv={c.geom.get('mean_curvature')}, angle={c.geom.get('angle')}, score={c.score}")
                    except Exception:
                        pass
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
        
        picked_segs = self.select_by_gap(all_cands, boundary_lines=boundary_lines)
        return picked_segs