"""
Pre-study · Task A：受控点触场景采集
标注方式：荧光贴纸 HSV 面积自动检测（同 exp_a/a1_collect.py sticker 模式）

特征提取：通过 DualHandDetector.last_feat（HandFeatureExtractor）与 src 完全一致。
flow_mag 额外从指尖 ROI 单独计算（HandFeatureExtractor 中无此特征）。

输出（datasets/data_prestudy/tap/）：
  prestudy_tap_{subject}.csv          — 每帧特征 + contact_label
  prestudy_tap_{subject}_raw.mp4      — 原始视频（无任何标注）
  prestudy_tap_{subject}_ui.mp4       — UI 视频（含关键点/标注叠加）
"""

import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from src.hand_track.dual_hand_detector import DualHandDetector
from src.hand_track.feature_extractor import FEATURE_NAMES

# ── 输出目录 ────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_prestudy', 'tap')

# ── 贴纸 HSV 阈值 ────────────────────────────────────────────────────────────
STICKER_HSV = {
    'green':  ((35,  40,  30),  (85,  255, 255)),
    'yellow': ((20,  100, 100), (35,  255, 255)),
    'pink':   ((140, 80,  80),  (170, 255, 255)),
    'blue':   ((85,  15,  120), (225, 255, 255)),
    'black':  ((0,   0,   0),   (180, 80,  90)),
}
STICKER_OCCLUDE_RATIO = 0.4
STICKER_CALIB_FRAMES  = 30

# 与 HandFeatureExtractor._ROI_HALF 保持一致
_ROI_HALF = 16

DIST_WINDOW  = 5
SIGMA_WINDOW = 5

CSV_FIELDS = (
    ['frame_id', 'timestamp', 'contact_label',
     'dist_raw', 'dist_local', 'v_n', 'a_n', 'sigma_d', 'v_t', 'approach_theta',
     'shadow_score', 'flow_mag', 'brightness_contact',
     'dist2d_palm_0', 'dist2d_palm_5', 'dist2d_palm_9', 'dist2d_palm_13', 'dist2d_palm_17',
     'hull_overlap_ratio']
    + [f'lm_{i}_{ax}' for i in range(21) for ax in ('x', 'y', 'z')]
)


# ── 贴纸检测（与 a1_collect.py 保持一致） ────────────────────────────────────

def _detect_sticker_area(frame_hsv, lower, upper, palm_lm=None):
    mask = cv2.inRange(frame_hsv,
                       np.array(lower, dtype=np.uint8),
                       np.array(upper, dtype=np.uint8))
    has_palm = False
    if palm_lm is not None:
        h, w = frame_hsv.shape[:2]
        palm_core_idx = (0, 1, 2, 5, 9, 13, 17)
        palm_pts = np.array(
            [[int(palm_lm.landmark[i].x * w), int(palm_lm.landmark[i].y * h)]
             for i in palm_core_idx], dtype=np.int32)
        if palm_pts.shape[0] >= 3:
            has_palm = True
            palm_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(palm_mask, cv2.convexHull(palm_pts), 255)
            palm_mask = cv2.erode(
                palm_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1)
            mask = cv2.bitwise_and(mask, palm_mask)
    if not has_palm:
        mask[:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    return int(np.sum(mask > 0)), contour, has_palm


def _flow_mag_roi(frame_gray, prev_gray, cx, cy):
    """指尖 ROI 光流幅值（与 HandFeatureExtractor ROI 尺寸一致：32×32）。"""
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
    """计算 v_n / a_n / sigma_d / v_t，逻辑与 a1_collect.py 保持一致。"""
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

def run_collect(subject: str, sticker_color: str = 'green', cam_id: int = 0):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv    = os.path.join(OUT_DIR, f'prestudy_tap_{subject}.csv')
    out_raw    = os.path.join(OUT_DIR, f'prestudy_tap_{subject}_raw.mp4')
    out_ui     = os.path.join(OUT_DIR, f'prestudy_tap_{subject}_ui.mp4')

    hsv_lower, hsv_upper = STICKER_HSV.get(sticker_color, STICKER_HSV['green'])

    detector = DualHandDetector()
    kbuf     = _KinematicBuffer()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {cam_id}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0

    sticker_ref_area = None
    calib_areas      = []
    raw_writer = ui_writer = None
    prev_gray  = None
    frame_id   = 0

    print("=" * 55)
    print(f"  Pre-study TAP  受试者: {subject}")
    print(f"  贴纸颜色: {sticker_color}  标注: 自动（HSV 面积）")
    print(f"  特征: HandFeatureExtractor（与 src 一致）")
    print(f"  输出: raw.mp4 + ui.mp4 + .csv")
    print(f"  请将手掌平放，贴纸朝上，等待标定（约2秒）...")
    print(f"  按 Q 或 ESC 结束采集")
    print("=" * 55)

    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if raw_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                raw_writer = cv2.VideoWriter(out_raw, fourcc, fps, (w, h))
                ui_writer  = cv2.VideoWriter(out_ui,  fourcc, fps, (w, h))

            # ── 保存原始帧（无任何标注）──────────────────────────────────────
            raw_writer.write(frame)

            ts = time.time()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # ── DualHandDetector（与 src/main.py 一致，内含 HandFeatureExtractor）
            detector.process(frame)   # 会在 frame 上绘制手部关键点

            feat = detector.last_feat  # shape (10,) from HandFeatureExtractor
            # feat 索引见 src/hand_track/feature_extractor.py FEATURE_NAMES:
            # [0..4]=dist2d_lm{0,5,9,13,17}, [5]=hull_iou, [6]=local_n,
            # [7]=roi_brightness, [8]=roi_shadow, [9]=approach_theta

            dist_local = detector.dist_palm
            dist_raw   = dist_local   # PalmLocalFrame n 分量

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

            # ── 贴纸标注 ───────────────────────────────────────────────────
            contact_label = 0
            palm_ref = detector.palm_lm or detector.left_lm or detector.right_lm
            area, sticker_contour, has_palm = _detect_sticker_area(
                frame_hsv, hsv_lower, hsv_upper, palm_ref)
            if sticker_contour is not None:
                cv2.drawContours(frame, [sticker_contour], -1, (255, 0, 255), 2)

            if sticker_ref_area is None:
                if has_palm:
                    calib_areas.append(area)
                    if len(calib_areas) >= STICKER_CALIB_FRAMES:
                        sticker_ref_area = float(np.median(calib_areas))
                        print(f"  [标定完成] 贴纸参考面积: {sticker_ref_area:.0f} px²")
                        if sticker_ref_area < 50:
                            print("  [警告] 参考面积过小，贴纸可能未被检测到")
                    cv2.putText(frame,
                                f"Calibrating... {len(calib_areas)}/{STICKER_CALIB_FRAMES}",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    cv2.putText(frame, "Palm not detected",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            else:
                if has_palm:
                    ratio = area / sticker_ref_area if sticker_ref_area > 0 else 1.0
                    contact_label = 1 if ratio < STICKER_OCCLUDE_RATIO else 0
                    color = (0, 255, 0) if contact_label else (0, 0, 255)
                    cv2.putText(frame,
                                f"Sticker {area}px ({ratio:.2f})  "
                                f"{'CONTACT' if contact_label else 'IDLE'}",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(frame, "Palm not detected",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

            if dist_raw is not None:
                cv2.putText(frame, f"d={dist_raw:.1f}  vn={v_n:.2f}  sd={sigma_d:.2f}",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)
            cv2.putText(frame, f"Frame {frame_id}  Subject: {subject}",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            # ── 保存 UI 帧、显示 ──────────────────────────────────────────
            ui_writer.write(frame)
            cv2.imshow("Pre-study TAP", frame)

            # ── landmark 坐标（书写手） ───────────────────────────────────
            lm_flat = [0.0] * (21 * 3)
            if detector.write_lm:
                for i, lm_pt in enumerate(detector.write_lm.landmark):
                    lm_flat[i*3], lm_flat[i*3+1], lm_flat[i*3+2] = lm_pt.x, lm_pt.y, lm_pt.z

            def _f(v):
                return f'{v:.4f}' if v is not None and np.isfinite(v) else ''

            row = {
                'frame_id':      frame_id,
                'timestamp':     f'{ts:.6f}',
                'contact_label': contact_label,
                'dist_raw':      _f(dist_raw),
                'dist_local':    _f(dist_local),
                'v_n':           f'{v_n:.4f}',
                'a_n':           f'{a_n:.4f}',
                'sigma_d':       f'{sigma_d:.4f}',
                'v_t':           f'{v_t:.4f}',
                # from HandFeatureExtractor.last_feat
                'approach_theta':    _f(feat[9]),
                'shadow_score':      _f(feat[8]),
                'flow_mag':          f'{flow_mag:.4f}' if detector.write_lm else '',
                'brightness_contact': _f(feat[7]),
                'dist2d_palm_0':     _f(feat[0]),
                'dist2d_palm_5':     _f(feat[1]),
                'dist2d_palm_9':     _f(feat[2]),
                'dist2d_palm_13':    _f(feat[3]),
                'dist2d_palm_17':    _f(feat[4]),
                'hull_overlap_ratio': _f(feat[5]),
            }
            for i in range(21):
                row[f'lm_{i}_x'] = f'{lm_flat[i*3]:.6f}'
                row[f'lm_{i}_y'] = f'{lm_flat[i*3+1]:.6f}'
                row[f'lm_{i}_z'] = f'{lm_flat[i*3+2]:.6f}'
            writer.writerow(row)

            prev_gray = frame_gray.copy()
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    if raw_writer is not None:
        raw_writer.release()
    if ui_writer is not None:
        ui_writer.release()
    cv2.destroyAllWindows()
    detector.reset()
    print(f"  数据已保存: {out_csv}  ({frame_id} 帧)")
    print(f"  原始视频: {out_raw}")
    print(f"  UI 视频:  {out_ui}")
