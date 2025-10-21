# segmentation/connect_scorer.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .type import PcaPack

# def _mu_sigma_from_ci(low: float, high: float) -> Tuple[float, float]:
#     """
#     Convert a two-sided 95% CI (low, high) into (mu, sigma) assuming normal approx.

#     If you want to make the prior wider to reflect greater uncertainty, change
#     the global CI_EXPAND_FACTOR below (e.g. 1.2 or 1.5) which inflates the half-width
#     of the interval before converting to sigma.
#     """
#     mu = 0.5 * (low + high)
#     half_width = 0.5 * (high - low)
#     # Apply global expansion factor to the half-width (useful when data is scarce)
#     half_width *= CI_EXPAND_FACTOR
#     # sigma for 95% CI: half_width ≈ 1.96 * sigma  => sigma = half_width / 1.96
#     sigma = half_width / (1.96 + 1e-12)
#     return mu, float(sigma)

# # Global factor to inflate the input CI width before converting to (mu,sigma).
# # Set >1.0 to widen the implied prior variance when your underlying statistics are scarce.
# CI_EXPAND_FACTOR = 1.0

def _mu_sigma_from_ci(low: float, high: float) -> Tuple[float, float]:
    mu = 0.5 * (low + high)
    sigma = (high - low) / (2 * 1.96 + 1e-9)
    return mu, sigma

# ===== 你的统计先验（由 95% CI 推出 μ,σ） =====
PRIORS = {
    "straight": dict(  # 连笔 vs 非连笔
        mc=_mu_sigma_from_ci(0.861296, 0.935421)[0],
        sc=_mu_sigma_from_ci(0.861296, 0.935421)[1],
        mn=_mu_sigma_from_ci(0.660111, 0.756844)[0],
        sn=_mu_sigma_from_ci(0.660111, 0.756844)[1],
    ),
    "curv": dict(
        mc=_mu_sigma_from_ci(0.012115, 0.016564)[0],
        sc=_mu_sigma_from_ci(0.012115, 0.016564)[1],
        mn=_mu_sigma_from_ci(0.029422, 0.048186)[0],
        sn=_mu_sigma_from_ci(0.029422, 0.048186)[1],
    ),
    "angle": dict(
        mc=_mu_sigma_from_ci(-39.542020, -25.433484)[0],
        sc=_mu_sigma_from_ci(-39.542020, -25.433484)[1],
        mn=_mu_sigma_from_ci(7.962786, 40.103116)[0],
        sn=_mu_sigma_from_ci(7.962786, 40.103116)[1],
    ),
}

@dataclass
class Weights:
    w_s: float = 0.40   # 直线度
    w_k: float = 0.35   # 曲率
    w_a: float = 0.25   # 角度
    w_dens: float = 0.25  # 密度项（鼓励连笔位于密度谷）
    w_bdry: float = 0.20  # 边界可靠度

class ConnectSegmentScorer:
    def __init__(self, priors: Dict = None, weights: Weights = Weights(),
                 dens_ref_quantile: float = 0.25):
        self.priors = priors or PRIORS
        self.w = weights
        self.dens_ref_q = dens_ref_quantile

    @staticmethod
    def _llr(x, mc, sc, mn, sn):
        # log N_c - log N_n
        return np.log((sn + 1e-9) / (sc + 1e-9)) + 0.5 * (
            ((x - mn) / (sn + 1e-9)) ** 2 - ((x - mc) / (sc + 1e-9)) ** 2
        )

    def score_geom(self, straight: float, mean_curv: float, angle_deg: float) -> float:
        ps = self.priors["straight"]; pk = self.priors["curv"]; pa = self.priors["angle"]
        s = ( self.w.w_s * self._llr(straight, **ps)
            + self.w.w_k * self._llr(mean_curv, **pk)
            + self.w.w_a * self._llr(angle_deg, **pa) )
        return float(s)

    def score_density(self, seg_s: np.ndarray, pca: PcaPack) -> float:
        """
        取段内 PC1 投影的平均密度；越低越“像连笔”。
        返回一个可与几何分相加的分数（线性缩放到 [-1,1] 左右）。
        """
        # 段内均值（或最大值）密度
        bins = pca.density_bins
        prof = pca.density_profile
        # 将 seg_s 映射到最近 bin
        idx = np.clip(np.searchsorted(bins, seg_s), 1, len(bins)-1)
        # 线性插值得到密度
        left = bins[idx-1]; right = bins[idx]
        t = np.clip((seg_s - left) / (right - left + 1e-9), 0.0, 1.0)
        dens = (1-t) * prof[idx-1] + t * prof[idx]
        mean_dens = float(np.mean(dens))
        # 参考分位
        ref = float(np.quantile(prof, self.dens_ref_q))
        # 分数：低于参考 → 正向加分；比例映射一下
        # s_dens ≈ (ref - mean)/ref，截断到 [-1,1]
        s_d = np.clip((ref - mean_dens) / (ref + 1e-9), -1.0, 1.0)
        return float(s_d)

    def fuse(self, geom: float, dens: float) -> float:
        return float(geom + self.w.w_dens * dens)

    def score_segment(self, metrics, seg_s: np.ndarray, pca: PcaPack) -> float:
        g = self.score_geom(metrics["straightness"], metrics["mean_curvature"], metrics["angle"])
        d = self.score_density(seg_s, pca)
        return self.fuse(g, d)
