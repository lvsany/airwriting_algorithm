"""
Hover-Anchored 在线接触检测模块

通过采集用户自然悬停姿态建立特征基线，
将接触检测建模为"偏离 hover 正常态的统计异常"。
无需任何接触标注数据，天然适应跨场景分布偏移。

两阶段流程：
  阶段一（Hover 校准）—— 稳定性等待 → 采集 150 帧 → 鲁棒基线 + 自适应阈值 τ
  阶段二（在线检测）  —— 归一化欧氏距离 D 与阈值 τ → raw_contact
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ── 默认超参数 ────────────────────────────────────────────────────────────────
_STABILITY_WIN = 10      # 稳定性检测窗口（帧）
_STABILITY_THR = 5.0     # dist2d_lm0 滑动标准差阈值（像素），低于此值视为稳定
_COLLECT_N     = 150     # hover 采集帧数（约 3 秒 @60fps）
_PERCENTILE    = 99.0    # 自适应阈值 τ 所用百分位  # FIX: 缺陷D - 由95改为99，使hover帧的5%误判降至1%
_MAD_SCALE     = 1.4826  # MAD → 高斯等效 σ 的一致性修正因子
_SIGMA_MIN     = 1e-6    # σ 下界，触发后置 1.0（防除零）

# 参与马氏距离计算的维度（f_0, f_7, f_8, f_9 —— contact时一致减小的特征）
_ACTIVE_DIMS = np.array([0, 7, 8, 9])

_ENTER_SCALE = 1
_EXIT_SCALE  = 1.05

class _Phase(Enum):
    WAITING    = "waiting"     # 等待 dist2d_lm0 稳定
    COLLECTING = "collecting"  # 采集 hover 帧，建立基线
    READY      = "ready"       # 校准完成，在线检测中


@dataclass
class HoverDetectResult:
    """
    每帧 HoverAnchorDetector.update() 的返回值。

    字段
    ----
    phase       : 当前阶段 'waiting' | 'collecting' | 'ready'
    progress    : 校准完成度 [0.0, 1.0]
    distance    : 归一化欧氏距离 D_t（phase != 'ready' 时为 NaN）
    threshold   : 自适应阈值 τ（phase != 'ready' 时为 NaN）
    raw_contact : 原始接触判定（phase != 'ready' 时为 False）
    z_vec       : 归一化特征向量 (10,)，NaN 维度已置 0；
                  phase != 'ready' 时为全零向量
    """
    phase:       str
    progress:    float
    distance:    float
    threshold:   float
    raw_contact: bool
    z_vec:       np.ndarray


class HoverAnchorDetector:
    """
    Hover-Anchored 在线接触检测器。

    用法
    ----
    det = HoverAnchorDetector()
    while True:
        feat = extractor.extract(...)       # shape (10,), 可含 NaN
        result = det.update(feat)
        if result.raw_contact:
            ...
    """

    def __init__(
        self,
        stability_win:    int   = _STABILITY_WIN,
        stability_thr:    float = _STABILITY_THR,
        collect_n:        int   = _COLLECT_N,
        percentile:       float = _PERCENTILE,
    ):
        self._stability_win   = stability_win
        self._stability_thr   = stability_thr
        self._collect_n       = collect_n
        self._percentile      = percentile

        self._phase     = _Phase.WAITING
        self._stab_buf: deque = deque(maxlen=stability_win)
        self._hover_buf: list = []

        self._mu:  Optional[np.ndarray] = None
        self._sig: Optional[np.ndarray] = None
        self._tau: Optional[float]      = None
        self._in_contact: bool          = False

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def update(self, feat: np.ndarray) -> HoverDetectResult:
        """
        喂入一帧 10 维特征向量，返回检测结果。

        Parameters
        ----------
        feat : np.ndarray, shape (10,)
            HandFeatureExtractor.extract() 的输出；可含 NaN。

        Returns
        -------
        HoverDetectResult
        """
        if self._phase == _Phase.WAITING:
            return self._do_waiting(feat)
        if self._phase == _Phase.COLLECTING:
            return self._do_collecting(feat)
        return self._do_detecting(feat)

    def get_baseline(self) -> dict:
        """
        返回当前基线参数，可用于持久化或调试。
        校准未完成时，mu / sigma / tau 为 None。
        """
        return {
            'phase': self._phase.value,
            'mu':    self._mu.tolist()  if self._mu  is not None else None,
            'sigma': self._sig.tolist() if self._sig is not None else None,
            'tau':   self._tau,
        }

    def reset(self):
        """重新开始校准流程（场景切换或角色变更时调用）。"""
        self._phase = _Phase.WAITING
        self._stab_buf.clear()
        self._hover_buf.clear()
        self._mu = self._sig = self._tau = None
        self._in_contact = False

    @property
    def is_calibrated(self) -> bool:
        return self._phase == _Phase.READY

    def get_debug_detail(self) -> dict:
        """返回内部诊断状态，供日志和 HUD 使用。"""
        stab_std = (float(np.std(list(self._stab_buf)))
                    if len(self._stab_buf) > 1 else None)
        return {
            'phase':         self._phase.value,
            'stab_buf_len':  len(self._stab_buf),
            'stab_buf_std':  stab_std,
            'collect_n':     len(self._hover_buf),
            'collect_total': self._collect_n,
            'tau':           self._tau,
        }

    # ── 阶段处理 ─────────────────────────────────────────────────────────────

    def _do_waiting(self, feat: np.ndarray) -> HoverDetectResult:
        """等待 dist2d_lm0（dim 0）连续稳定。"""
        d0 = feat[0]
        if not np.isnan(d0):
            self._stab_buf.append(float(d0))

        if (len(self._stab_buf) >= self._stability_win and
                float(np.std(list(self._stab_buf))) < self._stability_thr):
            self._phase = _Phase.COLLECTING

        return HoverDetectResult(
            phase='waiting', progress=0.0,
            distance=np.nan, threshold=np.nan,
            raw_contact=False,
            z_vec=np.zeros(len(feat)),
        )

    def _do_collecting(self, feat: np.ndarray) -> HoverDetectResult:
        """采集 hover 帧，帧满后建立基线。"""
        self._hover_buf.append(feat.copy())
        n        = len(self._hover_buf)
        progress = n / self._collect_n * 0.99   # 留最后 1% 给 READY 确认帧

        if n >= self._collect_n:
            self._build_baseline()
            self._phase = _Phase.READY

        return HoverDetectResult(
            phase='collecting', progress=progress,
            distance=np.nan, threshold=np.nan,
            raw_contact=False,
            z_vec=np.zeros(len(feat)),
        )

    def set_contact_state(self, is_contact: bool) -> None:
        """由 ContactStateMachine 的最终判定回写，保持 _in_contact 与平滑状态同步。"""
        self._in_contact = is_contact

    def _do_detecting(self, feat: np.ndarray) -> HoverDetectResult:
        """在线检测：以 4 维归一化距离 + 方向约束判定接触。"""
        z = self._normalize(feat)
        D = float(np.linalg.norm(z[_ACTIVE_DIMS]))  # 只用4维计算距离

        # 方向约束：z[0] < 0 表示 f_0 比 hover 更小（指尖更靠近手腕 = 接触）
        z0_ok = z[0] < 0.0 if np.isfinite(z[0]) and z[0] != 0.0 else False

        enter_tau = self._tau * _ENTER_SCALE
        exit_tau  = self._tau * _EXIT_SCALE

        if not self._in_contact:
            raw_contact = (D > enter_tau) and z0_ok
        else:
            raw_contact = D > exit_tau  # 退出时不检查方向，避免卡住

        # 不再用 raw_contact 自更新；由外部 set_contact_state() 在状态机输出后回写。

        return HoverDetectResult(
            phase='ready', progress=1.0,
            distance=D, threshold=float(self._tau),
            raw_contact=raw_contact,
            z_vec=z,
        )

    # ── 内部计算 ─────────────────────────────────────────────────────────────

    def _normalize(self, feat: np.ndarray) -> np.ndarray:
        """z = (f − μ) / σ；NaN 维度（含 μ/σ 为 NaN 的维度）置 0。"""
        z = (feat - self._mu) / self._sig
        z[np.isnan(z)] = 0.0
        return z

    def _build_baseline(self):
        """
        从采集的 hover 帧计算鲁棒基线 μ / σ 和自适应阈值 τ。

        μ   = nanmedian（逐维度）
        σ   = 1.4826 × MAD（逐维度），< 1e-6 时置 1.0
        τ   = 第 99 百分位数（hover 帧距离分布）
        """
        F   = np.array(self._hover_buf)            # (N, 10)
        mu  = np.nanmedian(F, axis=0)              # (10,)
        mad = np.nanmedian(np.abs(F - mu), axis=0) # (10,)
        sig = _MAD_SCALE * mad
        # NaN sig 保持 NaN（对应全 NaN 列），在 _normalize 里会被置 0
        sig[np.isfinite(sig) & (sig < _SIGMA_MIN)] = 1.0

        # 用 hover 帧本身计算距离分布以确定阈值
        Z   = (F - mu) / sig
        Z[np.isnan(Z)] = 0.0
        # D 为对角协方差近似的马氏距离：|| (f-μ)/σ ||_2（仅使用部分维度）
        D   = np.linalg.norm(Z[:, _ACTIVE_DIMS], axis=1)  # (N,)
        tau = float(np.percentile(D, self._percentile))

        self._mu  = mu
        self._sig = sig
        self._tau = tau


# ── 单元测试 ──────────────────────────────────────────────────────────────────

def _run_unit_test():
    """
    合成数据验证：
    - 150 帧 hover → 建立基线
    - 50 帧 contact（dims 0-4 显著减小，模拟指尖靠近掌面）→ 全部应被检出
    """
    print("Running HoverAnchorDetector unit test...")
    rng = np.random.default_rng(42)
    N   = 10

    # hover: 全维度均值 100，std 2（低噪声）
    hover = rng.normal(100.0, 2.0, (200, N))
    # 随机给约 40% 帧的 dim 6 设为 NaN，模拟真实的 local_n 缺失
    hover[rng.random(200) < 0.4, 6] = np.nan

    # contact: dims 0-4 降至 60（指尖靠近 → 距离显著减小），
    #          dim 6 降至 90（local_n 相对 hover 降低 → 通过方向门控）
    contact = rng.normal(100.0, 2.0, (50, N))
    contact[:, :5] = rng.normal(60.0, 2.0, (50, 5))
    contact[:, 6] = rng.normal(90.0, 2.0, 50)

    # stability_win=1, stability_thr=1e9 → 第一帧即触发稳定
    det = HoverAnchorDetector(stability_win=1, stability_thr=1e9, collect_n=150)

    # 阶段一：喂入 200 帧 hover（1 帧 waiting + 150 帧 collecting + 49 帧 detecting）
    last = None
    for feat in hover:
        last = det.update(feat)

    assert last.phase == 'ready', f"expected ready, got {last.phase}"
    assert last.distance < last.threshold, \
        f"hover 帧距离 {last.distance:.3f} 应 < τ={last.threshold:.3f}"

    baseline = det.get_baseline()
    assert baseline['tau'] is not None
    print(f"  μ[0]={baseline['mu'][0]:.1f}  σ[0]={baseline['sigma'][0]:.2f}  "
          f"τ={baseline['tau']:.3f}")

    # 阶段二：喂入 50 帧 contact，验证检测率 ≥ 96%
    n_detected = 0
    for feat in contact:
        r = det.update(feat)
        assert r.phase == 'ready'
        assert r.distance >= 0.0
        assert np.isfinite(r.distance)
        if r.raw_contact:
            n_detected += 1

    rate = n_detected / 50
    assert rate >= 0.96, f"检测率过低: {n_detected}/50 ({rate*100:.0f}%)"
    print(f"  contact 检测率: {n_detected}/50 ({rate*100:.0f}%)")

    # 验证 reset 后恢复 waiting 状态
    det.reset()
    r0 = det.update(hover[0])
    assert r0.phase == 'waiting'
    assert np.isnan(r0.distance)
    print("  reset 后恢复 waiting: OK")

    print("UNIT TEST PASSED\n")


if __name__ == "__main__":
    _run_unit_test()
