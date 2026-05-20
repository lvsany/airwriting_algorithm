"""
实时摄像头测试 - Block A 核心功能验证（含详细诊断日志）
"""

import csv
import cv2
import numpy as np
import sys
import os
import time
import yaml
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_track.dual_hand_detector import DualHandDetector

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_cfg_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

# ─── 配色方案 ───
C = {
    'bg':         (20, 20, 25),
    'panel':      (30, 30, 38),
    'panel_edge': (55, 55, 68),
    'text':       (210, 210, 215),
    'text_dim':   (120, 120, 130),
    'accent':     (230, 165, 60),
    'accent2':    (90, 200, 250),
    'palm':       (80, 220, 200),
    'palm_fill':  (60, 160, 150),
    'writing':    (80, 230, 120),
    'idle':       (90, 90, 100),
    'hover':      (200, 180, 80),
    'touch':      (80, 180, 240),
    'lifted':     (180, 100, 220),
    'warn':       (60, 60, 240),
    'axis_x':     (80, 80, 255),
    'axis_y':     (80, 220, 80),
    'axis_z':     (255, 160, 80),
    'traj':       (45, 45, 50),
    'traj_dot':   (230, 165, 60),
    'canvas_bg':  (245, 245, 248),
    'canvas_grid':(225, 225, 230),
    'canvas_line':(40, 40, 50),
    'canvas_border': (180, 180, 190),
}

STATE_STYLE = {
    'idle':    {'color': C['idle'],    'label': 'IDLE'},
    'contact': {'color': C['writing'], 'label': 'CONTACT'},
}


# ─── CSV 日志 ───
_CSV_COLS = (
    ['frame_id', 'timestamp', 'phase', 'distance', 'threshold',
     'raw_contact', 'dir_ok', 'final_state', 'sm_pending',
     'left_det', 'right_det'] +
    [f'f_{i}' for i in range(10)] +
    [f'z_{i}' for i in range(10)]
)

class _CsvLogger:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(output_dir, f"session_{ts}.csv")
        self._fh   = open(self._path, 'w', newline='', encoding='utf-8')
        self._w    = csv.DictWriter(self._fh, fieldnames=_CSV_COLS)
        self._w.writeheader()
        self._count = 0
        print(f"[LOG] CSV → {self._path}")

    def write(self, frame_id: int, timestamp: float, detector: 'DualHandDetector'):
        hr   = detector.hover_result
        cr   = detector.contact_result
        feat = detector.last_feat
        z    = hr.z_vec if (hr and hr.z_vec is not None) else np.zeros(10)

        dbg  = detector._contact_sm.get_debug_info()
        dir_ok = int(float(np.mean(z[:5])) < 0.0) if hr and hr.phase == 'ready' else ''

        row = {
            'frame_id':   frame_id,
            'timestamp':  f"{timestamp:.4f}",
            'phase':      hr.phase if hr else 'waiting',
            'distance':   f"{hr.distance:.4f}"  if (hr and np.isfinite(hr.distance))  else '',
            'threshold':  f"{hr.threshold:.4f}" if (hr and np.isfinite(hr.threshold)) else '',
            'raw_contact': int(hr.raw_contact) if hr else 0,
            'dir_ok':     dir_ok,
            'final_state': cr.state.value if cr else 'idle',
            'sm_pending': dbg['pending'],
            'left_det':   int(detector.left_lm  is not None),
            'right_det':  int(detector.right_lm is not None),
        }
        for i in range(10):
            row[f'f_{i}'] = f"{feat[i]:.4f}" if np.isfinite(feat[i]) else ''
        for i in range(10):
            row[f'z_{i}'] = f"{z[i]:.4f}" if np.isfinite(z[i]) else ''

        self._w.writerow(row)
        self._count += 1
        # Flush every 60 frames so data survives crashes
        if self._count % 60 == 0:
            self._fh.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()
        print(f"[LOG] Saved {self._count} rows → {self._path}")


# ─── 延迟统计 ───
class _LatencyAccum:
    def __init__(self, window: int = 100):
        self._det = deque(maxlen=window)
        self._vis = deque(maxlen=window)
        self._tot = deque(maxlen=window)

    def record(self, t_det_ms: float, t_vis_ms: float, t_tot_ms: float):
        self._det.append(t_det_ms)
        self._vis.append(t_vis_ms)
        self._tot.append(t_tot_ms)

    def print_report(self, frame_id: int):
        if not self._det:
            return
        print(f"[LAT f={frame_id:5d}]  "
              f"detect={np.mean(self._det):.1f}ms  "
              f"vis={np.mean(self._vis):.1f}ms  "
              f"total={np.mean(self._tot):.1f}ms")

    def get(self) -> dict:
        return {
            'det_ms': np.mean(self._det) if self._det else 0.0,
            'vis_ms': np.mean(self._vis) if self._vis else 0.0,
            'tot_ms': np.mean(self._tot) if self._tot else 0.0,
        }


# ─── 绘图辅助 ───
def rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    x1, y1 = pt1; x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1+r, y1), (x2-r, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1+r), (x2, y2-r), 255, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    if thickness == -1:
        img[mask > 0] = color
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness, cv2.LINE_AA)


def draw_pill(img, center, w, h, color, alpha=1.0):
    x, y = center; r = h // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x - w//2 + r, y - h//2), (x + w//2 - r, y + h//2), color, -1)
    cv2.circle(overlay, (x - w//2 + r, y), r, color, -1)
    cv2.circle(overlay, (x + w//2 - r, y), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img, text, pos, scale=0.45, color=None, thickness=1,
             font=cv2.FONT_HERSHEY_SIMPLEX):
    if color is None:
        color = C['text']
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_hbar(img, x, y, w, h, ratio, fg_color, bg_color=(50, 50, 58)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = max(1, int(w * np.clip(ratio, 0, 1)))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), fg_color, -1)


# ─── 手掌平面可视化 ───
def draw_palm_plane_visualization(frame, detector):
    plf = detector.palm_local_frame
    if not plf.is_valid or detector.palm_lm is None:
        return
    h, w = frame.shape[:2]
    lm = detector.palm_lm.landmark

    def p2d(p3):
        return (int(p3[0] * w), int(p3[1] * h))

    all_pts = np.array([p2d([lm[i].x, lm[i].y, lm[i].z]) for i in range(21)])
    hull = cv2.convexHull(all_pts)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, hull, C['palm_fill'])
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [hull], True, C['palm'], 1, cv2.LINE_AA)
    for idx in [0, 5, 9, 13, 17]:
        pt = p2d([lm[idx].x, lm[idx].y, lm[idx].z])
        cv2.circle(frame, pt, 3, C['palm'], -1, cv2.LINE_AA)

    origin   = plf.origin
    o2d      = p2d(origin)
    axis_len = 0.08
    for axis, color, label in [
        (plf.u_axis, C['axis_x'], 'u'),
        (plf.v_axis, C['axis_y'], 'v'),
        (plf.n_axis, C['axis_z'], 'n'),
    ]:
        if axis is None:
            continue
        end = origin + axis * axis_len
        e2d = p2d(end)
        cv2.arrowedLine(frame, o2d, e2d, color, 2, cv2.LINE_AA, tipLength=0.22)
        put_text(frame, label, (e2d[0] + 5, e2d[1] - 5), 0.38, color, 1)
    cv2.circle(frame, o2d, 4, C['accent'], -1, cv2.LINE_AA)


# ─── 右手指尖高亮 ───
def draw_writing_cursor(frame, pos, is_writing):
    if pos is None or pos == (0, 0):
        return
    x, y = pos
    if is_writing:
        cv2.circle(frame, (x, y), 14, C['writing'], 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y),  5, C['traj_dot'], -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), 10, C['hover'], 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y),  2, C['hover'], -1, cv2.LINE_AA)


# ─── 当前笔画叠加到相机帧 ───
def draw_traj_on_frame(frame, screen_traj):
    """将正在书写的轨迹（屏幕坐标列表）叠加绘制到相机帧。"""
    valid = [p for p in screen_traj if p is not None]
    if len(valid) < 2:
        return
    for i in range(1, len(valid)):
        cv2.line(frame, valid[i-1], valid[i], C['traj_dot'], 2, cv2.LINE_AA)


# ─── HUD 信息面板（详细诊断模式） ───
def draw_debug_info(frame, detector, is_writing, fps, lat: dict):
    h, w = frame.shape[:2]
    pw = 272
    panel_h = 360

    panel = frame.copy()
    rounded_rect(panel, (8, 8), (8 + pw, panel_h), C['panel'], 10)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    rounded_rect(frame, (8, 8), (8 + pw, panel_h), C['panel_edge'], 10, 1)

    x0 = 20
    y  = 30

    # ── 标题 ──
    put_text(frame, "PalmWrite", (x0, y), 0.52, C['accent'], 1)
    put_text(frame, "BLOCK A", (x0 + 128, y), 0.33, C['text_dim'])
    y += 10
    cv2.line(frame, (x0, y), (x0 + pw - 20, y), C['panel_edge'], 1, cv2.LINE_AA)
    y += 18

    # ── FPS ──
    fps_c = C['writing'] if fps >= 25 else C['hover'] if fps >= 15 else C['warn']
    put_text(frame, "FPS", (x0, y), 0.36, C['text_dim'])
    put_text(frame, f"{fps:.0f}", (x0 + 44, y), 0.40, fps_c, 1)

    # ── 状态 pill ──
    hr = detector.hover_result
    cr = detector.contact_result
    state_val = cr.state.value if cr else 'idle'
    ss = STATE_STYLE.get(state_val, STATE_STYLE['idle'])
    draw_pill(frame, (x0 + 185, y - 5), 78, 20, ss['color'], 0.7)
    put_text(frame, ss['label'], (x0 + 157, y), 0.36, (255, 255, 255), 1)
    y += 24

    cv2.line(frame, (x0, y), (x0 + pw - 20, y), C['panel_edge'], 1, cv2.LINE_AA)
    y += 14

    # ── 手部检测状态 ──
    ld = detector.left_lm  is not None
    rd = detector.right_lm is not None
    lc = C['writing'] if ld else C['warn']
    rc = C['writing'] if rd else C['warn']
    lr = detector.left_role.value  if ld else "–"
    rr = detector.right_role.value if rd else "–"
    put_text(frame, "L hand", (x0, y), 0.33, C['text_dim'])
    put_text(frame, (lr.upper() if ld else "LOST"), (x0 + 55, y), 0.33, lc)
    put_text(frame, "R hand", (x0 + 135, y), 0.33, C['text_dim'])
    put_text(frame, (rr.upper() if rd else "LOST"), (x0 + 190, y), 0.33, rc)
    y += 22

    # ── Hover 校准阶段 ──
    phase    = hr.phase    if hr else 'waiting'
    progress = hr.progress if hr else 0.0
    detail   = detector._hover_det.get_debug_detail()

    if phase == 'waiting':
        std_val  = detail['stab_buf_std']
        buf_len  = detail['stab_buf_len']
        std_str  = f"{std_val:.1f}px" if std_val is not None else "–"
        p_color  = C['warn']
        put_text(frame, "Hover", (x0, y), 0.33, C['text_dim'])
        put_text(frame, "WAITING", (x0 + 50, y), 0.33, p_color)
        put_text(frame, f"std={std_str} buf={buf_len}/10", (x0 + 118, y), 0.29, C['text_dim'])
        y += 20
        put_text(frame, "  → hold both hands still to calibrate", (x0, y), 0.29, C['text_dim'])

    elif phase == 'collecting':
        n        = detail['collect_n']
        total    = detail['collect_total']
        p_color  = C['hover']
        put_text(frame, "Hover", (x0, y), 0.33, C['text_dim'])
        put_text(frame, f"CALIB {n}/{total}", (x0 + 50, y), 0.33, p_color)
        draw_hbar(frame, x0 + 148, y - 10, 100, 12, progress, p_color)

    else:  # ready
        tau = detail['tau']
        put_text(frame, "Hover", (x0, y), 0.33, C['text_dim'])
        put_text(frame, "READY", (x0 + 50, y), 0.33, C['writing'])
        put_text(frame, f"τ={tau:.3f}" if tau else "τ=?", (x0 + 110, y), 0.30, C['text_dim'])

    y += 24

    # ── Mahalanobis 距离 (红/绿色条) ──
    put_text(frame, "D (Mahal.)", (x0, y), 0.33, C['text_dim'])
    if hr and hr.phase == 'ready':
        D   = hr.distance
        tau = hr.threshold
        raw = hr.raw_contact
        z   = hr.z_vec
        dir_ok = float(np.mean(z[:5])) < 0.0

        ratio     = D / (tau * 2.0) if tau > 0 else 0.0
        bar_color = C['writing'] if D <= tau else C['warn']
        draw_hbar(frame, x0 + 80, y - 10, 145, 12, ratio, bar_color)

        d_str = f"D={D:.3f}"
        t_str = f"τ={tau:.3f}"
        put_text(frame, d_str, (x0 + 82, y), 0.30, (255, 255, 255), 1)
        put_text(frame, t_str, (x0 + 158, y), 0.30, C['text_dim'])
        y += 20

        # raw_contact + direction
        raw_c = C['writing'] if raw else C['idle']
        dir_c = C['writing'] if dir_ok else C['warn']
        put_text(frame, "raw_contact", (x0, y), 0.30, C['text_dim'])
        put_text(frame, str(int(raw)), (x0 + 84, y), 0.30, raw_c)
        put_text(frame, "dir_ok", (x0 + 115, y), 0.30, C['text_dim'])
        put_text(frame, str(int(dir_ok)), (x0 + 165, y), 0.30, dir_c)
        put_text(frame, f"pend={detector._contact_sm.get_debug_info()['pending']}",
                 (x0 + 192, y), 0.30, C['text_dim'])
    else:
        put_text(frame, "–", (x0 + 80, y), 0.33, C['text_dim'])
        y += 20
        put_text(frame, "raw_contact –  dir_ok –", (x0, y), 0.30, C['text_dim'])

    y += 24

    # ── 掌面 UV + n ──
    put_text(frame, "Palm UV", (x0, y), 0.33, C['text_dim'])
    if detector.write_pos_palm:
        pu, pv = detector.write_pos_palm
        nc = detector.dist_palm
        put_text(frame, f"u={pu:.3f} v={pv:.3f} n={nc:.4f}" if nc is not None
                 else f"u={pu:.3f} v={pv:.3f}",
                 (x0 + 57, y), 0.28, C['text'])
    else:
        put_text(frame, "–", (x0 + 57, y), 0.33, C['text_dim'])
    y += 22

    # ── 掌面坐标系 ──
    put_text(frame, "Palm frame", (x0, y), 0.33, C['text_dim'])
    if detector.palm_local_frame.is_valid:
        put_text(frame, "OK", (x0 + 80, y), 0.33, C['writing'])
    else:
        put_text(frame, "NO FIT", (x0 + 80, y), 0.33, C['warn'])
    y += 22

    cv2.line(frame, (x0, y), (x0 + pw - 20, y), C['panel_edge'], 1, cv2.LINE_AA)
    y += 14

    # ── 延迟统计 ──
    put_text(frame, "Latency", (x0, y), 0.33, C['text_dim'])
    put_text(frame, f"det={lat['det_ms']:.1f}ms  vis={lat['vis_ms']:.1f}ms  tot={lat['tot_ms']:.1f}ms",
             (x0 + 58, y), 0.28, C['text_dim'])
    y += 20

    # ── 键位提示 ──
    put_text(frame, "r=recalib  s=save  q=quit", (x0, y), 0.30, C['text_dim'])

    # ── 底部条 ──
    bar_h = 26
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - bar_h), (w, h), C['panel'], -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), C['panel_edge'], 1)
    put_text(frame, "R  Recalibrate    S  Save    C  Clear canvas    Q / ESC  Quit",
             (w // 2 - 210, h - 8), 0.34, C['text_dim'], 1)


# ─── 轨迹画布 ───
def create_canvas(h, w):
    canvas = np.full((h, w, 3), C['canvas_bg'], dtype=np.uint8)
    step = 40
    for gx in range(0, w, step):
        cv2.line(canvas, (gx, 0), (gx, h), C['canvas_grid'], 1)
    for gy in range(0, h, step):
        cv2.line(canvas, (0, gy), (w, gy), C['canvas_grid'], 1)
    return canvas


def draw_canvas_overlay(frame, canvas, scale=0.28):
    fh, fw = frame.shape[:2]
    ch, cw = int(fh * scale), int(fw * scale)
    mini = cv2.resize(canvas, (cw, ch), interpolation=cv2.INTER_AREA)
    margin = 12
    x1, y1 = fw - cw - margin, margin
    x2, y2 = x1 + cw, y1 + ch
    shadow = frame.copy()
    cv2.rectangle(shadow, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), -1)
    cv2.addWeighted(shadow, 0.3, frame, 0.7, 0, frame)
    frame[y1:y2, x1:x2] = mini
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), C['canvas_border'], 1, cv2.LINE_AA)
    put_text(frame, "CANVAS", (x1+4, y1+14), 0.30, C['text_dim'], 1)


# ─── 主循环 ───
def main():
    print("=" * 52)
    print("  Block A - Palm Writing Real-time Test")
    print("=" * 52)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERR] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG['video']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['video']['height'])
    cap.set(cv2.CAP_PROP_FPS,          CONFIG['video']['fps'])

    palm_mode = CONFIG.get('palm_writing', {}).get('enabled', True)

    output_dir = "./results/test"

    if not palm_mode:
        print("[ERR] palm_writing.enabled is false; HAMCD requires palm mode")
        return

    detector = DualHandDetector()
    logger   = _CsvLogger(output_dir)
    print("[OK] Dual-hand palm writing mode (hover-anchored)")
    print("[OK] Heuristic handedness correction ON")
    print("[TIP] Hold both hands steady in hover position to calibrate")

    palm_traces    = []    # 已完成的 (u,v) 轨迹列表
    current_trace  = []    # 当前笔画的 (u,v) 轨迹
    screen_traj    = []    # 屏幕像素位置，仅用于帧叠加可视化
    prev_pos       = None  # canvas 绘制上一像素位置
    prev_writing   = False

    vh, vw = CONFIG['video']['height'], CONFIG['video']['width']
    canvas = create_canvas(vh, vw)

    lat    = _LatencyAccum(window=100)
    fps_t0 = time.time()
    fps_cnt = 0
    fps = 0.0
    frame_id = 0
    t_session_start = time.time()

    os.makedirs(output_dir, exist_ok=True)
    print("\n  R  Recalibrate  |  S  Save  |  C  Clear  |  Q/ESC  Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_frame_start = time.perf_counter()
            frame_id += 1
            ts = time.time() - t_session_start

            # ── 1. 检测（计时） ──────────────────────────────────────────
            t0 = time.perf_counter()
            is_writing = detector.process(frame)
            cur_pos    = detector.get_screen_position()
            t_det_ms = (time.perf_counter() - t0) * 1e3

            hr = getattr(detector, 'hover_result', None)
            cr = getattr(detector, 'contact_result', None)

            # ── 2. 校准完成时打印基线摘要 ────────────────────────────────
            if hr and hr.phase == 'ready':
                if not getattr(main, '_calib_printed', False):
                    baseline = detector._hover_det.get_baseline()
                    tau = baseline.get('tau')
                    print(f"[CALIB DONE] τ={tau:.4f}"
                          f"  μ[0]={baseline['mu'][0]:.1f}"
                          f"  σ[0]={baseline['sigma'][0]:.2f}")
                    main._calib_printed = True
            elif hr and hr.phase != 'ready':
                main._calib_printed = False

            # ── 3. 接触事件日志 ──────────────────────────────────────────
            if cr:
                if cr.changed and cr.state.value == 'contact':
                    D   = hr.distance if hr else float('nan')
                    tau = hr.threshold if hr else float('nan')
                    print(f"[CONTACT ↓  f={frame_id}]  D={D:.3f}  τ={tau:.3f}  "
                          f"onset={cr.onset_frame}")
                elif cr.changed and cr.state.value == 'idle':
                    print(f"[IDLE    ↑  f={frame_id}]  offset={cr.offset_frame}")

            # WAITING 阶段每 150 帧打印一次卡住原因
            if hr and hr.phase == 'waiting' and frame_id % 150 == 0:
                dd = detector._hover_det.get_debug_detail()
                std_val = dd['stab_buf_std']
                std_str = f"{std_val:.2f}px" if std_val is not None else "–"
                print(f"[WAITING   f={frame_id}]  "
                      f"stab_std={std_str}  "
                      f"buf={dd['stab_buf_len']}/10  "
                      f"L={detector.left_lm is not None}  "
                      f"R={detector.right_lm is not None}")

            # ── 4. CSV 日志（在可视化前完成） ────────────────────────────
            if logger:
                logger.write(frame_id, ts, detector)

            # ── 5. 轨迹记录 ────────────────────────────────────────────
            palm_uv = detector.get_writing_position()

            if not prev_writing and is_writing:
                current_trace = []
                screen_traj.clear()

            if prev_writing and not is_writing:
                if current_trace:
                    palm_traces.append(current_trace)
                current_trace = []
                screen_traj.clear()
                prev_pos = None

            prev_writing = is_writing

            if is_writing and cur_pos != (0, 0):
                if palm_uv is not None:
                    current_trace.append(palm_uv)
                if prev_pos is not None:
                    cv2.line(canvas, prev_pos, cur_pos, C['canvas_line'], 2, cv2.LINE_AA)
                prev_pos = cur_pos
                screen_traj.append(cur_pos)
            else:
                prev_pos = None

            # ── 6. FPS ───────────────────────────────────────────────────
            fps_cnt += 1
            if fps_cnt >= 30:
                fps = fps_cnt / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_cnt = 0

            # ── 7. 可视化（计时，与计算分离） ────────────────────────────
            t_vis0 = time.perf_counter()

            draw_palm_plane_visualization(frame, detector)
            draw_debug_info(frame, detector, is_writing, fps, lat.get())

            draw_traj_on_frame(frame, screen_traj)
            draw_canvas_overlay(frame, canvas)
            draw_writing_cursor(frame, cur_pos if cur_pos != (0, 0) else None, is_writing)

            t_vis_ms  = (time.perf_counter() - t_vis0) * 1e3
            t_tot_ms  = (time.perf_counter() - t_frame_start) * 1e3
            lat.record(t_det_ms, t_vis_ms, t_tot_ms)

            if frame_id % 100 == 0:
                lat.print_report(frame_id)

            cv2.imshow('PalmWrite - Block A', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                detector.reset()
                palm_traces.clear()
                current_trace = []
                screen_traj.clear()
                prev_pos = None
                prev_writing = False
                main._calib_printed = False
                print(f"[RESET f={frame_id}] Calibration reset")
            elif key == ord('s'):
                ts_str = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(output_dir, f"traj_{ts_str}.png")
                cv2.imwrite(img_path, canvas)
                print(f"[SAVE] {img_path}")
                if logger:
                    logger.close()
                    logger = _CsvLogger(output_dir)
            elif key == ord('c'):
                canvas = create_canvas(vh, vw)
                palm_traces.clear()
                current_trace = []
                screen_traj.clear()
                prev_pos = None
                print("[CLEAR] Canvas reset")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logger:
            logger.close()
        if current_trace:
            palm_traces.append(current_trace)
        total = sum(len(t) for t in palm_traces)
        print(f"\n  Strokes: {len(palm_traces)}  Points: {total}")
        print(f"  Output:  {output_dir}")


if __name__ == "__main__":
    main._calib_printed = False
    main()
