"""
Pre-study · Task B：连续书写场景采集
标注方式：受试者同步按住空格键记录接触状态（实时，无需后处理）

  - 按住空格 → contact_label = 1（接触中）
  - 松开空格 → contact_label = 0（悬空）

特征提取：通过 DualHandDetector.last_feat（HandFeatureExtractor）与 src 完全一致。
flow_mag 额外从指尖 ROI 单独计算（HandFeatureExtractor 中无此特征）。

输出（datasets/prestudy/data/write/）：
  prestudy_write_{subject}_{lighting}_{speed}.csv
  prestudy_write_{subject}_{lighting}_{speed}_raw.mp4   — 原始视频
  prestudy_write_{subject}_{lighting}_{speed}_ui.mp4    — UI 视频
  prestudy_write_{subject}_{lighting}_{speed}_meta.json
"""

import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.hand_track.dual_hand_detector import DualHandDetector
from src.hand_track.feature_extractor import FEATURE_NAMES

OUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'write')

# 与 HandFeatureExtractor._ROI_HALF 保持一致
_ROI_HALF    = 16
SIGMA_WINDOW = 7
DIST_WINDOW  = 5

LIGHTING_OPTIONS = {"normal", "low", "side"}
SPEED_OPTIONS    = {"slow", "normal", "fast"}

CSV_FIELDS = [
    "frame_id", "timestamp", "contact_label",
    "dist_raw", "dist_local", "v_n", "a_n", "sigma_d", "v_t", "approach_theta",
    "shadow_score", "flow_mag", "brightness_contact",
    "dist2d_palm_0", "dist2d_palm_5", "dist2d_palm_9", "dist2d_palm_13", "dist2d_palm_17",
    "hull_overlap_ratio",
] + [f"lm_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]

# ── 空格键实时检测 ────────────────────────────────────────────────────────────
try:
    import keyboard as _kb
    _HAS_KEYBOARD_LIB = True
except ImportError:
    _HAS_KEYBOARD_LIB = False


def _space_is_pressed(toggle_state: list) -> int:
    if _HAS_KEYBOARD_LIB:
        return 1 if _kb.is_pressed('space') else 0
    return toggle_state[0]


def _fmt_hms(total_sec: float) -> str:
    sec = max(0, int(total_sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def _draw_text(frame, text, org, color=(255, 255, 255), scale=0.72, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _flow_mag_roi(frame_gray, prev_gray, cx, cy):
    """指尖 ROI 光流幅值（ROI 尺寸与 HandFeatureExtractor 一致：32×32）。"""
    h, w = frame_gray.shape
    x1, x2 = max(cx - _ROI_HALF, 0), min(cx + _ROI_HALF, w)
    y1, y2 = max(cy - _ROI_HALF, 0), min(cy + _ROI_HALF, h)
    roi = frame_gray[y1:y2, x1:x2]
    if prev_gray is None or roi.shape[0] < 4 or roi.shape[1] < 4:
        return 0.0
    prev_roi = prev_gray[y1:y2, x1:x2]
    if prev_roi.shape != roi.shape:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_roi, roi, None,
        pyr_scale=0.5, levels=2, winsize=9,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
    return float(np.mean(np.linalg.norm(flow, axis=2)))


class _KinematicBuffer:
    def __init__(self):
        self.dist_buf = deque(maxlen=DIST_WINDOW + 2)
        self.sigma_buf = deque(maxlen=SIGMA_WINDOW)
        self.pos_buf   = deque(maxlen=5)
        self._prev_vn  = None

    def push(self, dist, pos_uv, ts):
        self.dist_buf.append(dist)
        self.sigma_buf.append(dist)
        self.pos_buf.append((*pos_uv, ts))

    def v_n(self):
        if len(self.dist_buf) < 2:
            return 0.0
        return float(np.mean(np.diff(list(self.dist_buf)[-DIST_WINDOW:])))

    def a_n(self, vn_now):
        a = float(vn_now - self._prev_vn) if self._prev_vn is not None else 0.0
        self._prev_vn = vn_now
        return a

    def sigma_d(self):
        return float(np.std(list(self.sigma_buf))) if len(self.sigma_buf) >= 2 else 0.0

    def v_t(self):
        if len(self.pos_buf) < 2:
            return 0.0
        pts = list(self.pos_buf)
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        dt = pts[-1][2] - pts[0][2]
        return float(np.sqrt(dx**2 + dy**2) / dt * 1000.0) if dt > 1e-6 else 0.0

    def reset(self):
        self.dist_buf.clear()
        self.sigma_buf.clear()
        self.pos_buf.clear()
        self._prev_vn = None


# ── 主采集函数 ────────────────────────────────────────────────────────────────

def run_collect(subject: str, lighting: str, speed: str, cam_id: int = 0):
    os.makedirs(OUT_DIR, exist_ok=True)
    base     = f"prestudy_write_{subject}_{lighting}_{speed}"
    out_csv  = os.path.join(OUT_DIR, f"{base}.csv")
    out_raw  = os.path.join(OUT_DIR, f"{base}_raw.mp4")
    out_ui   = os.path.join(OUT_DIR, f"{base}_ui.mp4")
    out_meta = os.path.join(OUT_DIR, f"{base}_meta.json")

    detector = DualHandDetector()
    kbuf     = _KinematicBuffer()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {cam_id}")

    # 暖机
    first_frame = None
    for _ in range(45):
        ok, probe = cap.read()
        if ok and probe is not None and probe.size > 0:
            first_frame = probe
            break
        time.sleep(0.03)
    if first_frame is None:
        cap.release()
        raise RuntimeError("摄像头首帧读取失败")

    fps_cam = float(cap.get(cv2.CAP_PROP_FPS))
    if fps_cam <= 1e-3:
        fps_cam = 30.0

    if not _HAS_KEYBOARD_LIB:
        print("  [提示] keyboard 库未安装，将使用空格键 toggle 模式")
        print("         可用 `pip install keyboard` 安装持续按住模式")

    print("=" * 55)
    print(f"  Pre-study WRITE  受试者: {subject}")
    print(f"  光照: {lighting}  速度: {speed}")
    print(f"  特征: HandFeatureExtractor（与 src 一致）")
    print(f"  输出: raw.mp4 + ui.mp4 + .csv")
    if _HAS_KEYBOARD_LIB:
        print("  标注: 按住空格 = 接触 (contact=1)")
    else:
        print("  标注: 空格键 toggle（按下切换 0/1）")
    print("  按 Q 结束采集")
    print("=" * 55)

    cv2.namedWindow("Pre-study WRITE", cv2.WINDOW_NORMAL)

    raw_writer = ui_writer = None
    prev_gray    = None
    toggle_state = [0]
    total_frames = 0
    start_time   = time.time()
    pending      = first_frame

    with open(out_csv, "w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(CSV_FIELDS)

        while True:
            if pending is not None:
                frame, pending = pending, None
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            h, w = frame.shape[:2]
            if raw_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                raw_writer = cv2.VideoWriter(out_raw, fourcc, fps_cam, (w, h))
                ui_writer  = cv2.VideoWriter(out_ui,  fourcc, fps_cam, (w, h))

            # ── 保存原始帧（无任何标注）──────────────────────────────────
            raw_writer.write(frame)

            ts = time.time()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── DualHandDetector（与 src/main.py 一致，内含 HandFeatureExtractor）
            detector.process(frame)  # 在 frame 上绘制关键点

            feat = detector.last_feat  # shape (10,) from HandFeatureExtractor
            # [0..4]=dist2d_lm{0,5,9,13,17}, [5]=hull_iou, [6]=local_n,
            # [7]=roi_brightness, [8]=roi_shadow, [9]=approach_theta

            dist_local = detector.dist_palm
            dist_raw   = dist_local

            v_n = a_n = sigma_d = v_t = 0.0
            if dist_raw is not None:
                pos_uv = detector.write_pos_palm or (0.0, 0.0)
                kbuf.push(dist_raw, pos_uv, ts)
                v_n     = kbuf.v_n()
                sigma_d = kbuf.sigma_d()
                v_t     = kbuf.v_t()
                a_n     = kbuf.a_n(v_n)
            else:
                kbuf.reset()

            # flow_mag：HandFeatureExtractor 未包含，单独计算
            flow_mag = 0.0
            if detector.write_lm:
                px, py = detector.write_pos
                flow_mag = _flow_mag_roi(frame_gray, prev_gray, px, py)
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)

            # ── 空格键标注 ────────────────────────────────────────────────
            contact_label = _space_is_pressed(toggle_state)

            # ── landmark 坐标（书写手） ───────────────────────────────────
            lm_row = [""] * (21 * 3)
            if detector.write_lm:
                for i in range(21):
                    lm = detector.write_lm.landmark[i]
                    lm_row[i*3]   = f"{lm.x:.6f}"
                    lm_row[i*3+1] = f"{lm.y:.6f}"
                    lm_row[i*3+2] = f"{lm.z:.6f}"

            def _f(v):
                return f'{v:.4f}' if v is not None and np.isfinite(float(v)) else ''

            row = [
                total_frames, f"{ts:.6f}", contact_label,
                _f(dist_raw), _f(dist_local),
                f"{v_n:.4f}", f"{a_n:.4f}", f"{sigma_d:.4f}", f"{v_t:.4f}",
                _f(feat[9]),   # approach_theta
                _f(feat[8]),   # shadow_score
                f"{flow_mag:.4f}" if detector.write_lm else '',  # flow_mag
                _f(feat[7]),   # brightness_contact
                _f(feat[0]), _f(feat[1]), _f(feat[2]), _f(feat[3]), _f(feat[4]),  # dist2d
                _f(feat[5]),   # hull_overlap_ratio
            ] + lm_row
            csv_writer.writerow(row)
            f_csv.flush()

            # ── 绘制 UI 覆盖信息（在已有关键点的 frame 上叠加）───────────
            elapsed = time.time() - start_time
            contact_color = (0, 255, 0) if contact_label else (0, 0, 255)
            contact_text  = "CONTACT (space)" if contact_label else "IDLE"
            _draw_text(frame, f"REC  {_fmt_hms(elapsed)}  Frame:{total_frames}",
                       (20, 40), (255, 255, 255))
            _draw_text(frame, f"{subject} | {lighting} | {speed}",
                       (20, 74), (255, 255, 255))
            _draw_text(frame, contact_text, (20, 108), contact_color)
            if not detector.write_lm:
                _draw_text(frame, "No dual hand detected", (20, 142), (0, 140, 255))
            if not _HAS_KEYBOARD_LIB:
                _draw_text(frame, "toggle mode  [space]=switch",
                           (20, 176), (180, 180, 100), scale=0.6, thickness=1)
            _draw_text(frame, "Press Q to stop",
                       (20, h - 20), (150, 150, 150), scale=0.6, thickness=1)

            # ── 保存 UI 帧、显示 ──────────────────────────────────────────
            ui_writer.write(frame)
            cv2.imshow("Pre-study WRITE", frame)

            prev_gray    = frame_gray.copy()
            total_frames += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not _HAS_KEYBOARD_LIB:
                toggle_state[0] = 1 - toggle_state[0]

    duration_sec = max(0.0, time.time() - start_time)
    fps_actual   = total_frames / duration_sec if duration_sec > 1e-6 else 0.0

    meta = {
        "subject": subject, "lighting": lighting, "speed": speed,
        "total_frames": total_frames,
        "fps_actual": round(fps_actual, 1),
        "duration_sec": int(round(duration_sec)),
        "label_mode": "keyboard_hold" if _HAS_KEYBOARD_LIB else "keyboard_toggle",
        "collect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    cap.release()
    if raw_writer is not None:
        raw_writer.release()
    if ui_writer is not None:
        ui_writer.release()
    detector.reset()
    cv2.destroyAllWindows()

    mm, ss = int(duration_sec) // 60, int(duration_sec) % 60
    print(f"\n========== 采集完成 ==========")
    print(f"受试者: {subject}  光照: {lighting}  速度: {speed}")
    print(f"总帧数: {total_frames}  时长: {mm}分{ss:02d}秒  帧率: {fps_actual:.1f}fps")
    print(f"CSV: {out_csv}")
    print(f"原始视频: {out_raw}")
    print(f"UI 视频:  {out_ui}")
    print("==============================")
