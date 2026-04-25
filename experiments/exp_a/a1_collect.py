"""
Exp-A1: 纯 tap 数据采集与标注

标注模式：
  sticker  — 荧光贴纸 HSV 颜色面积检测，全自动，误差 < 1 帧

输出 CSV 格式（每帧一行）：
  frame_id, timestamp, contact_label,
  dist_raw, v_n, sigma_d, v_t,
  shadow_score, flow_mag,
  lm_{i}_x/y/z (i=0..20)
"""

import cv2
import numpy as np
import csv
import time
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.hand_track.dual_hand_detector import DualHandDetector

# ── HSV 颜色范围（荧光贴纸） ────────────────────────────────────────────────
STICKER_HSV = {
    'green':  ((35,  40,  30),  (85,  255, 255)),
    'yellow': ((20,  100, 100), (35,  255, 255)),
    'pink':   ((140, 80,  80),  (170, 255, 255)),
    'blue':   ((100, 100, 80),  (130, 255, 255)),
    'black':  ((0,   0,   0),   (180, 80,  90)),
}
STICKER_OCCLUDE_RATIO = 0.4   # 面积低于参考值此比例时判定为遮挡
STICKER_CALIB_FRAMES  = 30    # 标定所需帧数

DIST_WINDOW  = 5
SIGMA_WINDOW = 5


def _detect_sticker_area(frame_hsv, lower, upper):
    mask = cv2.inRange(frame_hsv,
                       np.array(lower, dtype=np.uint8),
                       np.array(upper, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return int(np.sum(mask > 0))


def _extract_appearance(frame_gray, prev_gray, cx, cy, radius=18):
    h, w = frame_gray.shape
    x1, x2 = max(cx - radius, 0), min(cx + radius, w)
    y1, y2 = max(cy - radius, 0), min(cy + radius, h)
    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0

    shadow = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    flow_mag = 0.0
    if prev_gray is not None:
        prev_roi = prev_gray[y1:y2, x1:x2]
        if prev_roi.shape == roi.shape and roi.shape[0] > 4 and roi.shape[1] > 4:
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, roi, None,
                pyr_scale=0.5, levels=2, winsize=9,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            flow_mag = float(np.mean(np.linalg.norm(flow, axis=2)))

    return shadow, flow_mag


class FeatureBuffer:
    def __init__(self):
        self.dist_buf  = deque(maxlen=DIST_WINDOW + 2)
        self.sigma_buf = deque(maxlen=SIGMA_WINDOW)
        self.pos_buf   = deque(maxlen=5)

    def push(self, dist, pos_xy, ts):
        self.dist_buf.append(dist)
        self.sigma_buf.append(dist)
        self.pos_buf.append((*pos_xy, ts))

    def v_n(self):
        if len(self.dist_buf) < 2:
            return 0.0
        return float(np.mean(np.diff(list(self.dist_buf)[-DIST_WINDOW:])))

    def sigma_d(self):
        if len(self.sigma_buf) < 2:
            return 0.0
        return float(np.std(list(self.sigma_buf)))

    def v_t(self):
        if len(self.pos_buf) < 2:
            return 0.0
        pts = list(self.pos_buf)
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        dt = pts[-1][2] - pts[0][2]
        if dt < 1e-6:
            return 0.0
        return float(np.sqrt(dx**2 + dy**2) / dt * 1000.0)


def run_collect(subject: str, data_dir: str,
                label_mode: str = 'sticker',
                sticker_color: str = 'green'):
    if label_mode != 'sticker':
        raise ValueError("Exp-A1 仅支持 sticker 标注模式")

    os.makedirs(data_dir, exist_ok=True)

    detector  = DualHandDetector()
    feat_buf  = FeatureBuffer()

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    out_csv = os.path.join(data_dir, f'exp_a1_{subject}.csv')
    out_video = os.path.join(data_dir, f'exp_a1_{subject}_raw.mp4')
    fieldnames = (
        ['frame_id', 'timestamp', 'contact_label',
         'dist_raw', 'v_n', 'sigma_d', 'v_t',
         'shadow_score', 'flow_mag'] +
        [f'lm_{i}_{ax}' for i in range(21) for ax in ('x', 'y', 'z')]
    )

    hsv_lower, hsv_upper = STICKER_HSV.get(sticker_color, STICKER_HSV['green'])
    sticker_ref_area = None
    calib_areas      = []
    video_writer     = None
    video_fps        = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 1e-3:
        video_fps = 30.0
    prev_gray        = None
    frame_id         = 0

    print("=" * 55)
    print(f"  标注模式: 荧光贴纸（{sticker_color}）自动检测")
    print("  请将手掌平放，贴纸朝上，等待标定（约2秒）...")
    print("  按 Q 或 ESC 结束采集")
    print("=" * 55)

    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    out_video,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    video_fps,
                    (w, h),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"无法创建视频文件: {out_video}")
            raw_frame = frame.copy()

            ts = time.time()
            detector.process(frame, ts)

            frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 从 DualHandDetector 读取已计算好的特征 ────────────────────
            dist_raw = detector.dist_palm
            pos_xy   = detector.write_pos_palm if detector.write_pos_palm else (0.0, 0.0)

            v_n = sigma_d = v_t = 0.0
            shadow_score = flow_mag = 0.0
            lm_flat = [0.0] * (21 * 3)

            if dist_raw is not None:
                feat_buf.push(dist_raw, pos_xy, ts)
                v_n     = feat_buf.v_n()
                sigma_d = feat_buf.sigma_d()
                v_t     = feat_buf.v_t()

                # 接触投影点 → 像素坐标，提取外观特征
                px, py = detector.write_pos
                shadow_score, flow_mag = _extract_appearance(
                    frame_gray, prev_gray, px, py)
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)

            # 关键点坐标（书写手）
            if detector.write_lm:
                for i, lm_pt in enumerate(detector.write_lm.landmark):
                    lm_flat[i*3+0] = lm_pt.x
                    lm_flat[i*3+1] = lm_pt.y
                    lm_flat[i*3+2] = lm_pt.z

            # ── 标注判断 ──────────────────────────────────────────────────
            contact_label = 0

            area = _detect_sticker_area(frame_hsv, hsv_lower, hsv_upper)
            if sticker_ref_area is None:
                calib_areas.append(area)
                if len(calib_areas) >= STICKER_CALIB_FRAMES:
                    sticker_ref_area = float(np.median(calib_areas))
                    print(f"  [标定完成] 贴纸参考面积: {sticker_ref_area:.0f} px²")
                    if sticker_ref_area < 50:
                        print(f"  [警告] 参考面积过小（{sticker_ref_area:.0f}px²），贴纸可能未被检测到")
                cv2.putText(frame,
                            f"Calibrating... {len(calib_areas)}/{STICKER_CALIB_FRAMES}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                ratio = area / sticker_ref_area if sticker_ref_area > 0 else 1.0
                contact_label = 1 if ratio < STICKER_OCCLUDE_RATIO else 0
                color = (0, 255, 0) if contact_label else (0, 0, 255)
                cv2.putText(frame,
                            f"Sticker {area}px ({ratio:.2f})  "
                            f"{'CONTACT' if contact_label else 'IDLE'}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ── 调试信息 ──────────────────────────────────────────────────
            if dist_raw is not None:
                cv2.putText(frame,
                            f"d={dist_raw:.1f}mm  vn={v_n:.2f}  sd={sigma_d:.2f}",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)
                cv2.putText(frame,
                            f"shadow={shadow_score:.1f}  flow={flow_mag:.2f}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)

            cv2.putText(frame, f"Frame {frame_id}  Subject: {subject}",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.imshow("Exp-A1 Collect", frame)

            # ── 写入 CSV ──────────────────────────────────────────────────
            row = {
                'frame_id':     frame_id,
                'timestamp':    f'{ts:.6f}',
                'contact_label': contact_label,
                'dist_raw':     f'{dist_raw:.4f}' if dist_raw is not None else '',
                'v_n':          f'{v_n:.4f}',
                'sigma_d':      f'{sigma_d:.4f}',
                'v_t':          f'{v_t:.4f}',
                'shadow_score': f'{shadow_score:.4f}',
                'flow_mag':     f'{flow_mag:.4f}',
            }
            for i in range(21):
                row[f'lm_{i}_x'] = f'{lm_flat[i*3]:.6f}'
                row[f'lm_{i}_y'] = f'{lm_flat[i*3+1]:.6f}'
                row[f'lm_{i}_z'] = f'{lm_flat[i*3+2]:.6f}'
            writer.writerow(row)
            video_writer.write(raw_frame)

            prev_gray = frame_gray.copy()
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    detector.reset()
    print(f"  数据已保存: {out_csv}  ({frame_id} 帧)")
    print(f"  原视频已保存: {out_video}")
