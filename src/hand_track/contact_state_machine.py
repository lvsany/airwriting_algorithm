"""
接触状态机
对 HoverAnchorDetector 的 raw_contact 输出去抖，并精确化接触 onset 帧号。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from enum import Enum


class ContactState(Enum):
    IDLE    = "idle"
    CONTACT = "contact"


# ── 默认超参数 ────────────────────────────────────────────────────────────────
_CONFIRM_FRAMES = 4   # 状态切换所需的连续确认帧数（≈50ms @60fps）
_ONSET_LOOKBACK = 8   # onset 精确化回溯窗口（帧）


@dataclass
class SmoothContactResult:
    """
    每帧 ContactStateMachine.update() 的返回值。

    state        : 平滑后的当前状态（ContactState.IDLE / CONTACT）
    changed      : 本帧是否发生状态切换
    onset_frame  : IDLE→CONTACT 切换时精确化的 onset 帧号；未切换时为 None
    offset_frame : CONTACT→IDLE 切换时的 offset 帧号；未切换时为 None
    """
    state:        ContactState
    changed:      bool
    onset_frame:  Optional[int]
    offset_frame: Optional[int]


class ContactStateMachine:
    """
    轻量级防抖状态机：IDLE ↔ CONTACT。

    转移规则
    --------
    IDLE    → CONTACT : 连续 confirm_frames 帧 raw_contact=True
    CONTACT → IDLE    : 连续 confirm_frames 帧 raw_contact=False

    任意一帧方向与目标相反，待确认计数立即归零，必须重新积累。

    Contact Onset 精确化
    --------------------
    刚切入 CONTACT 时，回溯最近 onset_lookback 帧的距离序列，
    取 argmax(D[t] - D[t-1]) 对应的帧号作为精确 onset。
    D（Mahalanobis 距离）在接触发生时因特征向量快速偏离 hover 基线而急剧增大，
    差分最大处即为真正的接触起始时刻，可减少约 50ms 的检测延迟。
    """

    def __init__(
        self,
        confirm_frames: int = _CONFIRM_FRAMES,
        onset_lookback: int = _ONSET_LOOKBACK,
    ):
        self._confirm  = confirm_frames
        self._lookback = onset_lookback

        self._state   = ContactState.IDLE
        self._pending = 0                                # 反向连续帧计数
        self._buf: deque = deque(maxlen=onset_lookback)  # (frame_id, dist, raw)

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def update(
        self,
        raw_contact: bool,
        distance:    float,
        frame_id:    int,
    ) -> SmoothContactResult:
        """
        每帧调用一次，推进状态机。

        Parameters
        ----------
        raw_contact : HoverDetectResult.raw_contact（校准阶段传 False）
        distance    : HoverDetectResult.distance（可为 NaN）
        frame_id    : 单调递增的帧序号，由调用方维护

        Returns
        -------
        SmoothContactResult
        """
        self._buf.append((frame_id, distance, raw_contact))
        target = ContactState.CONTACT if raw_contact else ContactState.IDLE

        # 方向与当前状态一致：重置待确认计数，状态稳定
        if target == self._state:
            self._pending = 0
            return SmoothContactResult(
                state=self._state, changed=False,
                onset_frame=None, offset_frame=None,
            )

        # 方向相反：积累待确认帧
        self._pending += 1
        if self._pending < self._confirm:
            return SmoothContactResult(
                state=self._state, changed=False,
                onset_frame=None, offset_frame=None,
            )

        # 连续确认帧达到阈值，执行状态切换
        self._pending = 0
        self._state   = target

        onset_frame  = None
        offset_frame = None
        if self._state == ContactState.CONTACT:
            onset_frame = self._refine_onset(frame_id)
        else:
            offset_frame = frame_id

        return SmoothContactResult(
            state=self._state, changed=True,
            onset_frame=onset_frame,
            offset_frame=offset_frame,
        )

    @property
    def is_contact(self) -> bool:
        return self._state == ContactState.CONTACT

    @property
    def state(self) -> ContactState:
        return self._state

    def reset(self):
        """重置到初始 IDLE 状态。"""
        self._state   = ContactState.IDLE
        self._pending = 0
        self._buf.clear()

    def get_debug_info(self) -> dict:
        return {
            'state':   self._state.value,
            'pending': self._pending,
            'buf_len': len(self._buf),
        }

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _refine_onset(self, current_frame_id: int) -> int:
        """
        回溯缓冲区，以 argmax(D[t] - D[t-1]) 精确化 onset 帧号。
        若有效帧不足 2 帧或距离全为 NaN，退回 current_frame_id。
        """
        buf = list(self._buf)
        if len(buf) < 2:
            return current_frame_id

        ids   = [e[0] for e in buf]
        dists = [e[1] for e in buf]

        best_idx  = None
        best_diff = float('-inf')
        for i in range(1, len(dists)):
            d_prev, d_curr = dists[i - 1], dists[i]
            if math.isnan(d_prev) or math.isnan(d_curr):
                continue
            diff = d_curr - d_prev
            if diff > best_diff:
                best_diff = diff
                best_idx  = i

        return ids[best_idx] if best_idx is not None else current_frame_id


# ── 自测 ─────────────────────────────────────────────────────────────────────

def _run_unit_test():
    """验证防抖逻辑、onset 精确化和 reset。"""
    print("Running ContactStateMachine unit test...")

    sm = ContactStateMachine(confirm_frames=3, onset_lookback=8)

    # ── 防抖：需要连续 3 帧才能切换 ─────────────────────────────────────────
    r = sm.update(True, 1.0, frame_id=0)
    assert r.state.value == 'idle' and not r.changed, "1 帧不应切换"
    r = sm.update(True, 2.0, frame_id=1)
    assert r.state.value == 'idle' and not r.changed, "2 帧不应切换"
    r = sm.update(True, 5.0, frame_id=2)
    assert r.state.value == 'contact' and r.changed,  "3 帧应切换到 CONTACT"
    assert r.onset_frame is not None, "onset_frame 应有值"
    assert r.offset_frame is None

    # onset 精确化：距离序列 [1.0, 2.0, 5.0]，最大差分在 frame_id=2（5-2=3）
    assert r.onset_frame == 2, f"onset 应为 2，得到 {r.onset_frame}"
    print(f"  onset_frame={r.onset_frame}  (expected 2) — OK")

    # ── 方向中断：连续计数归零 ────────────────────────────────────────────────
    r = sm.update(False, 4.5, frame_id=3)  # 1 帧 False，pending=1
    assert r.state.value == 'contact' and not r.changed
    r = sm.update(True, 5.5, frame_id=4)   # True 打断，pending 归零
    assert r.state.value == 'contact' and not r.changed
    r = sm.update(False, 4.0, frame_id=5)  # 重新积累 1 帧
    r = sm.update(False, 3.0, frame_id=6)  # 2 帧
    assert not r.changed, "2 帧 False 不应切换"
    r = sm.update(False, 2.0, frame_id=7)  # 3 帧 → 切换到 IDLE
    assert r.state.value == 'idle' and r.changed, "连续 3 帧 False 应切换到 IDLE"
    assert r.offset_frame == 7 and r.onset_frame is None
    print(f"  offset_frame={r.offset_frame}  (expected 7) — OK")

    # ── onset 精确化：NaN 距离被跳过 ─────────────────────────────────────────
    sm.reset()
    assert sm.state.value == 'idle'
    import math as _math
    sm.update(True, float('nan'), frame_id=10)
    sm.update(True, float('nan'), frame_id=11)
    r = sm.update(True, 3.0,     frame_id=12)
    assert r.changed and r.onset_frame == 12, \
        f"NaN 帧全跳过，应退回 current_frame_id=12，得 {r.onset_frame}"
    print(f"  NaN 回退 onset_frame={r.onset_frame}  (expected 12) — OK")

    # ── is_contact 属性 ───────────────────────────────────────────────────────
    sm.reset()
    assert not sm.is_contact
    for i, raw in enumerate([True, True, True]):
        sm.update(raw, float(i + 1), frame_id=i)
    assert sm.is_contact
    print("  is_contact property — OK")

    print("UNIT TEST PASSED\n")


if __name__ == "__main__":
    _run_unit_test()
