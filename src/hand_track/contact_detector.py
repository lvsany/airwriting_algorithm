"""
多特征手部接触检测器

基于 Exp-A 调研结论：
  主方案（运动学）: dist_raw + v_n + sigma_d  → AUC ≥ 0.85
  辅助特征:         dist2d_palm_{0,5,17}, brightness_contact

两种工作模式:
  'rule'  — 内置加权规则，无需训练数据，开箱即用
  'model' — 加载 sklearn LogisticRegression (.pkl)，离线训练后部署

接口与旧 ContactStateMachine 兼容:
  update(dist_raw, v_n, sigma_d, ...)  → ContactState
  is_contact() / just_started() / just_stopped() / reset()
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np


class ContactState(Enum):
    IDLE    = "idle"
    CONTACT = "contact"


# ── 内置规则参数（来自 Exp-A 调研） ──────────────────────────────────────────
# 每个特征对"接触"的贡献分（越高越接近接触）
_RULE = dict(
    # 一级门控：距离必须在掌面边界内
    dist_gate_mm      = 30.0,   # dist_raw 超过此值直接判 IDLE

    # 主特征权重（归一化评分后的线性组合）
    w_dist            = 0.45,   # 距离越小分越高
    dist_contact_mm   = 5.0,    # ≤ 此距离得满分
    dist_idle_mm      = 20.0,   # ≥ 此距离得0分

    w_sigma           = 0.25,   # sigma_d 越小越稳定 → 分越高
    sigma_contact     = 1.0,    # ≤ 此std得满分
    sigma_idle        = 6.0,    # ≥ 此std得0分

    w_vn              = 0.20,   # v_n 趋0（停驻）或负（靠近）→ 分高
    vn_approach       = -0.5,   # ≤ 此速度（靠近）得满分
    vn_leaving        = 1.5,    # ≥ 此速度（远离）得0分

    w_dist2d          = 0.10,   # dist2d_palm_{0,5,17} 平均值，越小越近

    # 决策门槛（带滞回防抖）
    thresh_enter      = 0.55,   # 分数超过此值 IDLE → CONTACT
    thresh_exit       = 0.42,   # 分数低于此值 CONTACT → IDLE

    # 时序平滑与防抖
    smooth_win        = 5,      # 评分滑动均值窗口（帧）
    confirm_frames    = 3,      # 状态切换需连续确认帧数（≈100ms @30fps）
    min_contact_frames = 8,     # 进入 CONTACT 后至少保持帧数，防止短暂抖出
    min_idle_frames   = 5,      # 进入 IDLE 后至少保持帧数，防止误触反弹
)

# dist2d_palm_{0,5,17} 在接触/空闲时的参考值（来自 Exp-A 箱线图）
_DIST2D_CONTACT_PX = 80.0
_DIST2D_IDLE_PX    = 180.0


class MultiFeatureContactDetector:
    """
    多特征手部接触检测器。

    Parameters
    ----------
    mode : 'rule' | 'model'
        'rule'  — 内置加权规则，无需任何训练。
        'model' — 从 model_path 加载 sklearn Pipeline (含 StandardScaler + LogisticRegression)。
    model_path : str, optional
        mode='model' 时需提供 .pkl 路径（由 exp_a3_roc.py 训练导出）。
    rule_params : dict, optional
        覆盖 _RULE 中任意参数，便于针对特定被试微调。
    """

    def __init__(
        self,
        mode: str = "rule",
        model_path: Optional[str] = None,
        rule_params: Optional[dict] = None,
    ):
        assert mode in ("rule", "model"), "mode 必须为 'rule' 或 'model'"
        self.mode = mode
        self._model = None

        self._p = dict(_RULE)
        if rule_params:
            self._p.update(rule_params)

        if mode == "model":
            if model_path is None:
                raise ValueError("mode='model' 时必须提供 model_path")
            self.load_model(model_path)

        self._state   = ContactState.IDLE
        self._prev    = ContactState.IDLE
        self._frames  = 0           # 当前状态已持续帧数
        self._pending = None        # 待确认的目标状态
        self._pending_frames = 0
        self._locked  = False       # 最短持续帧锁定中

        self._score_buf = deque(maxlen=self._p["smooth_win"])

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def update(
        self,
        dist_raw: Optional[float],
        v_n: float = 0.0,
        sigma_d: float = 0.0,
        dist2d_palm: Optional[dict] = None,   # {0: px, 5: px, 17: px, ...}
        brightness: Optional[float] = None,
        *,
        return_score: bool = False,
    ) -> ContactState:
        """
        每帧调用一次，更新接触状态。

        Parameters
        ----------
        dist_raw    : 指尖到掌面的3D距离（mm）；None 表示手不在掌面范围内。
        v_n         : 距离变化速率（mm/frame），负值=靠近，正值=远离。
        sigma_d     : 最近 N 帧距离的标准差（mm）。
        dist2d_palm : dict，指尖到各掌心关键点的2D像素距离。
        brightness  : 指尖附近区域亮度均值（0-255）。
        return_score: True 时同时返回 (state, score)。

        Returns
        -------
        ContactState  (或 (ContactState, float) when return_score=True)
        """
        # 一级门控：超出掌面边界 → 直接 IDLE
        if dist_raw is None or dist_raw > self._p["dist_gate_mm"]:
            self._score_buf.clear()
            self._transition(ContactState.IDLE, force=True)
            score = 0.0
            return (self._state, score) if return_score else self._state

        score = self._compute_score(dist_raw, v_n, sigma_d, dist2d_palm, brightness)
        self._score_buf.append(score)
        smooth_score = float(np.mean(self._score_buf))

        target = self._threshold(smooth_score)
        self._transition(target)
        self._frames += 1

        return (self._state, smooth_score) if return_score else self._state

    def is_contact(self) -> bool:
        return self._state == ContactState.CONTACT

    def just_started(self) -> bool:
        """本帧刚进入接触状态"""
        return self._prev == ContactState.IDLE and self._state == ContactState.CONTACT

    def just_stopped(self) -> bool:
        """本帧刚离开接触状态"""
        return self._prev == ContactState.CONTACT and self._state == ContactState.IDLE

    def get_state(self) -> ContactState:
        return self._state

    def get_score(self) -> float:
        """返回最近平滑后的评分（0~1）"""
        return float(np.mean(self._score_buf)) if self._score_buf else 0.0

    def load_model(self, path: str):
        """加载 sklearn Pipeline (.pkl)，切换到 model 模式"""
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self.mode = "model"

    def reset(self):
        self._state = self._prev = ContactState.IDLE
        self._frames = self._pending_frames = 0
        self._pending = None
        self._locked  = False
        self._score_buf.clear()

    def get_debug_info(self) -> dict:
        return {
            "state":        self._state.value,
            "prev_state":   self._prev.value,
            "state_frames": self._frames,
            "score":        self.get_score(),
            "mode":         self.mode,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _compute_score(
        self,
        dist_raw: float,
        v_n: float,
        sigma_d: float,
        dist2d_palm: Optional[dict],
        brightness: Optional[float],
    ) -> float:
        if self.mode == "model" and self._model is not None:
            return self._model_score(dist_raw, v_n, sigma_d, dist2d_palm, brightness)
        return self._rule_score(dist_raw, v_n, sigma_d, dist2d_palm)

    def _rule_score(
        self,
        dist_raw: float,
        v_n: float,
        sigma_d: float,
        dist2d_palm: Optional[dict],
    ) -> float:
        p = self._p

        # --- 距离分 ---
        s_dist = _linear_score(dist_raw, p["dist_contact_mm"], p["dist_idle_mm"])

        # --- sigma_d 分（接触时更小）---
        s_sigma = _linear_score(sigma_d, p["sigma_contact"], p["sigma_idle"])

        # --- v_n 分（靠近=负=高分；远离=正=低分） ---
        s_vn = _linear_score(v_n, p["vn_approach"], p["vn_leaving"])

        # --- dist2d 辅助分 ---
        s_dist2d = 0.5  # 无数据时中性
        if dist2d_palm:
            vals = [dist2d_palm[k] for k in (0, 5, 17) if k in dist2d_palm]
            if vals:
                mean_px = float(np.mean(vals))
                s_dist2d = _linear_score(mean_px, _DIST2D_CONTACT_PX, _DIST2D_IDLE_PX)

        score = (
            p["w_dist"]  * s_dist  +
            p["w_sigma"] * s_sigma +
            p["w_vn"]    * s_vn    +
            p["w_dist2d"] * s_dist2d
        )
        return float(np.clip(score, 0.0, 1.0))

    def _model_score(
        self,
        dist_raw: float,
        v_n: float,
        sigma_d: float,
        dist2d_palm: Optional[dict],
        brightness: Optional[float],
    ) -> float:
        """用加载的 sklearn Pipeline 返回接触概率"""
        d0 = dist2d_palm.get(0,  150.0) if dist2d_palm else 150.0
        d5 = dist2d_palm.get(5,  150.0) if dist2d_palm else 150.0
        d17= dist2d_palm.get(17, 150.0) if dist2d_palm else 150.0
        br = brightness if brightness is not None else 128.0
        X  = np.array([[dist_raw, v_n, sigma_d, d0, d5, d17, br]])
        return float(self._model.predict_proba(X)[0, 1])

    def _threshold(self, score: float) -> ContactState:
        """带滞回的阈值决策"""
        p = self._p
        if self._state == ContactState.IDLE:
            return ContactState.CONTACT if score >= p["thresh_enter"] else ContactState.IDLE
        else:
            return ContactState.IDLE if score < p["thresh_exit"] else ContactState.CONTACT

    def _transition(self, target: ContactState, force: bool = False):
        """带确认帧数 + 最短持续帧的状态切换（防抖）"""
        p = self._p

        if force:
            if self._state != target:
                self._prev, self._state = self._state, target
                self._frames = 0
                self._locked = True
            self._pending = None
            self._pending_frames = 0
            return

        # 最短持续帧锁定：当前状态未达到最小帧数时，忽略反向切换请求
        if self._locked:
            min_f = (p["min_contact_frames"] if self._state == ContactState.CONTACT
                     else p["min_idle_frames"])
            if self._frames < min_f:
                # 不允许离开当前状态，但仍累计同向 pending
                if target != self._state:
                    return
            else:
                self._locked = False

        if target == self._state:
            self._pending = None
            self._pending_frames = 0
        else:
            if self._pending == target:
                self._pending_frames += 1
            else:
                self._pending = target
                self._pending_frames = 1

            if self._pending_frames >= p["confirm_frames"]:
                self._prev, self._state = self._state, target
                self._frames = 0
                self._locked = True
                self._pending = None
                self._pending_frames = 0


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _linear_score(val: float, best: float, worst: float) -> float:
    """
    将 val 线性归一化到 [0, 1]。
    best  → 1.0（代表"接触"方向的极值）
    worst → 0.0（代表"空闲"方向的极值）
    """
    if abs(worst - best) < 1e-9:
        return 1.0 if val <= best else 0.0
    score = (worst - val) / (worst - best)
    return float(np.clip(score, 0.0, 1.0))


# ── 离线训练辅助：导出 sklearn 模型 ──────────────────────────────────────────

def train_and_save_model(csv_path: str, model_path: str):
    """
    从 exp_a1 CSV 训练逻辑回归模型并保存为 .pkl，
    供 MultiFeatureContactDetector(mode='model', model_path=...) 加载。

    特征列: dist_raw, v_n, sigma_d, dist2d_palm_0, dist2d_palm_5, dist2d_palm_17,
            brightness_contact
    标签列: contact_label
    """
    import pandas as pd
    from sklearn.linear_model  import LogisticRegression
    from sklearn.pipeline      import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute        import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    FEATURES = [
        "dist_raw", "v_n", "sigma_d",
        "dist2d_palm_0", "dist2d_palm_5", "dist2d_palm_17",
        "brightness_contact",
    ]

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["dist_raw"]).reset_index(drop=True)
    for f in FEATURES:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    avail = [f for f in FEATURES if f in df.columns and df[f].notna().sum() >= 20]
    X = df[avail].values
    y = df["contact_label"].astype(int).values

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(max_iter=500, C=1.0, random_state=42)),
    ])

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs  = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"  交叉验证 AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"  使用特征: {avail}")

    pipe.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  模型已保存: {model_path}")
    return pipe, avail


def _run_camera_demo():
    """
    实时摄像头接触检测 Demo（纯 MediaPipe + OpenCV，无项目依赖）

    角色分配：左手 = 画布（palm），右手 = 书写（writing）。
    按键: ESC/Q 退出  R 重置
    """
    import time
    import cv2
    import mediapipe as mp

    CAM_ID = "test1.mp4"      # 摄像头编号，改为 1/2 可切换
    FW, FH = 1280, 720

    # ── MediaPipe ─────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    # ── 掌面拟合（SVD，world landmarks，单位：米） ────────────────────
    _PALM_IDX = [0, 1, 5, 9, 13, 17]   # 腕 + 五指根
    _BOUND_IDX = list(range(21))        # 边界用全部关键点

    def fit_palm_plane(world_lms):
        """返回 (origin, normal)，world 坐标系，米"""
        pts = np.array([[world_lms[i].x, world_lms[i].y, world_lms[i].z]
                        for i in _PALM_IDX])
        origin = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - origin)
        normal = vt[-1]
        return origin, normal

    def tip_to_plane_mm(world_lms, origin, normal, tip_idx=8):
        tip = np.array([world_lms[tip_idx].x,
                        world_lms[tip_idx].y,
                        world_lms[tip_idx].z])
        dist_m = abs(float(np.dot(tip - origin, normal)))
        return dist_m * 1000.0   # → mm

    def tip_in_palm_region(norm_lms_write, norm_lms_palm, shape, margin=0.08):
        """指尖投影是否在掌心凸包内（带 margin）"""
        h, w = shape[:2]
        palm_pts = np.array([[norm_lms_palm[i].x * w, norm_lms_palm[i].y * h]
                             for i in _BOUND_IDX], dtype=np.float32)
        hull = cv2.convexHull(palm_pts)
        tip = norm_lms_write[8]
        tx, ty = tip.x * w, tip.y * h
        return cv2.pointPolygonTest(hull, (tx, ty), False) >= -margin * w

    # ── 特征缓冲区 ────────────────────────────────────────────────────
    DIST_WIN, SIGMA_WIN = 5, 8
    _dbuf = deque(maxlen=DIST_WIN + 2)
    _sbuf = deque(maxlen=SIGMA_WIN)

    def _push(d):  _dbuf.append(d); _sbuf.append(d)
    def _vn():
        return float(np.mean(np.diff(list(_dbuf)[-DIST_WIN:]))) if len(_dbuf)>=2 else 0.0
    def _sd():
        return float(np.std(list(_sbuf))) if len(_sbuf)>1 else 0.0
    def _clear(): _dbuf.clear(); _sbuf.clear()

    # ── 辅助特征 ──────────────────────────────────────────────────────
    def brightness(gray, cx, cy, r=18):
        roi = gray[max(cy-r,0):cy+r, max(cx-r,0):cx+r]
        return float(np.mean(roi)) if roi.size else 128.0

    def dist2d_palm(norm_w, norm_p, shape):
        h, w = shape[:2]
        tx = norm_w[8].x * w; ty = norm_w[8].y * h
        return {i: float(np.hypot(norm_p[i].x*w - tx, norm_p[i].y*h - ty))
                for i in (0, 5, 9, 13, 17)}

    # ── 检测器 ────────────────────────────────────────────────────────
    det = MultiFeatureContactDetector(mode="rule")

    # ── 主循环 ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f"[ERR] 无法打开摄像头 id={CAM_ID}"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FH)

    fps_cnt, fps_t0, fps_val = 0, time.time(), 0.0
    flash_n = 0
    print(f"摄像头 {CAM_ID}  {FW}×{FH}   ESC/Q 退出  R 重置\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = hands.process(rgb)

        write_lm = write_wlm = palm_lm = palm_wlm = None
        dist = vn = sd = 0.0

        if res.multi_hand_landmarks and res.multi_handedness:
            lms_list  = res.multi_hand_landmarks
            wlms_list = res.multi_hand_world_landmarks
            hd_list   = res.multi_handedness

            # 角色分配（非镜像摄像头）：
            # MediaPipe "Left"  = 画面左侧 = 真实右手 → writing（书写手）
            # MediaPipe "Right" = 画面右侧 = 真实左手 → palm（画板手）
            for lm, wlm, hd in zip(lms_list, wlms_list, hd_list):
                label = hd.classification[0].label
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                if label == "Right":
                    palm_lm, palm_wlm = lm, wlm
                else:
                    write_lm, write_wlm = lm, wlm

            if palm_wlm and write_wlm:
                origin, normal = fit_palm_plane(palm_wlm.landmark)

                # 只在指尖投影落在掌心区域内时才计算距离
                if tip_in_palm_region(write_lm.landmark, palm_lm.landmark, frame.shape):
                    dist = tip_to_plane_mm(write_wlm.landmark, origin, normal)
                    _push(dist); vn = _vn(); sd = _sd()
                else:
                    dist = None; _clear()

                d2d = dist2d_palm(write_lm.landmark, palm_lm.landmark, frame.shape)
                tip_px = (int(write_lm.landmark[8].x * frame.shape[1]),
                          int(write_lm.landmark[8].y * frame.shape[0]))
                br = brightness(gray, *tip_px)
            else:
                dist = None; _clear()
                d2d = {}; br = None; tip_px = None
        else:
            dist = None; _clear()
            d2d = {}; br = None; tip_px = None

        det.update(dist_raw=dist, v_n=vn, sigma_d=sd, dist2d_palm=d2d, brightness=br)

        # ── 接触事件日志 ──────────────────────────────────────────────
        if det.just_started():
            flash_n = 6
            print(f"  ▶ CONTACT  dist={dist:.1f}mm  vn={vn:.2f}  σ={sd:.2f}  "
                  f"score={det.get_score():.3f}")
        if det.just_stopped():
            flash_n = 4
            print(f"  ■ IDLE     score={det.get_score():.3f}")

        # ── 可视化 ────────────────────────────────────────────────────
        if flash_n > 0:
            col = (60, 60, 220) if det.is_contact() else (60, 140, 220)
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (frame.shape[1], frame.shape[0]), col, 20)
            cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)
            flash_n -= 1

        # 指尖光标
        if tip_px:
            col = (60,60,220) if det.is_contact() else (60,160,220)
            cv2.circle(frame, tip_px, 14, col, 2, cv2.LINE_AA)
            cv2.circle(frame, tip_px, 4,  col,-1, cv2.LINE_AA)

        # 文字 HUD（极简）
        state_str = "CONTACT" if det.is_contact() else "IDLE"
        score     = det.get_score()
        col_s = (60,60,220) if det.is_contact() else (100,100,110)
        cv2.putText(frame, state_str, (16, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, col_s, 2, cv2.LINE_AA)
        cv2.putText(frame, f"score {score:.2f}", (16, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,190), 1, cv2.LINE_AA)
        dist_str = f"dist  {dist:.1f} mm" if dist is not None else "dist  --"
        cv2.putText(frame, dist_str, (16, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"vn {vn:+.2f}  sd {sd:.2f}", (16, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140,140,150), 1, cv2.LINE_AA)

        fps_cnt += 1
        if fps_cnt >= 20:
            fps_val = fps_cnt / (time.time() - fps_t0)
            fps_t0 = time.time(); fps_cnt = 0
        cv2.putText(frame, f"{fps_val:.0f} fps", (FW - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,110), 1, cv2.LINE_AA)

        cv2.imshow("Contact Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        elif key == ord('r'):
            det.reset(); _clear(); print("  [RESET]")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Demo 结束")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        train_and_save_model(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        _run_camera_demo()
    else:
        print("用法:")
        print("  python contact_detector.py                        # 实时摄像头 Demo")
        print("  python contact_detector.py <csv> <output.pkl>     # 离线训练模型")
