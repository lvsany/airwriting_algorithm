"""
Exp-A1: 纯 tap 数据采集与标注

双标注模式：
  sticker  — 荧光贴纸 HSV 颜色面积检测，全自动，误差 < 1 帧
  keyboard — 空格键按下=接触，松开=抬起，误差 < 2 帧

输出 CSV 格式（每帧一行）：
  frame_id, timestamp, contact_label,
  dist_raw,
  v_n, sigma_d, v_t,
  shadow_score, flow_mag,
  lm_{i}_x/y/z (i=0..20)
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os
import yaml
import sys
from collections import deque
from threading import Event

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.hand_track.palm_coordinate_system import PalmPlaneTracker
from src.utils.geometry_utils import get_landmark_3d

# ── HSV 颜色范围（荧光贴纸） ────────────────────────────────────────────────
STICKER_HSV = {
    'green':  ((40,  80, 80),  (80,  255, 255)),
    'yellow': ((20,  100, 100), (35, 255, 255)),
    'pink':   ((140, 80, 80),  (170, 255, 255)),
    'blue':   ((100, 100, 80), (130, 255, 255)),
}
# 贴纸被遮挡时面积骤降的阈值比例（相对于标定面积）
STICKER_OCCLUDE_RATIO = 0.4
# 贴纸标定帧数（采集开始前静止等待）
STICKER_CALIB_FRAMES = 30
# 滑窗参数
DIST_WINDOW = 5   # 距离差分窗口
SIGMA_WINDOW = 5  # 方差滑窗


def _load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _detect_sticker_area(frame_hsv, lower, upper):
    mask = cv2.inRange(frame_hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return int(np.sum(mask > 0))


def _extract_appearance(frame_gray, prev_gray, cx, cy, radius=18):
    """在接触投影点附近提取阴影分数和光学流幅值"""
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
    """维护距离历史，计算 v_n / σ_d / v_t"""

    def __init__(self):
        self.dist_buf = deque(maxlen=DIST_WINDOW + 2)
        self.sigma_buf = deque(maxlen=SIGMA_WINDOW)
        self.pos_buf = deque(maxlen=5)   # (x, y, t)

    def push(self, dist, pos_xy, ts):
        self.dist_buf.append(dist)
        self.sigma_buf.append(dist)
        self.pos_buf.append((*pos_xy, ts))

    def v_n(self):
        if len(self.dist_buf) < 2:
            return 0.0
        diffs = np.diff(list(self.dist_buf)[-DIST_WINDOW:])
        return float(np.mean(diffs))

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

    config = _load_config()
    palm_tracker = PalmPlaneTracker(config)
    feat_buf = FeatureBuffer()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    out_csv = os.path.join(data_dir, f'exp_a1_{subject}.csv')
    fieldnames = (
        ['frame_id', 'timestamp', 'contact_label',
         'dist_raw', 'v_n', 'sigma_d', 'v_t',
         'shadow_score', 'flow_mag'] +
        [f'lm_{i}_{ax}' for i in range(21) for ax in ('x', 'y', 'z')]
    )

    # 贴纸标定
    sticker_ref_area = None
    calib_areas = []
    hsv_lower, hsv_upper = STICKER_HSV.get(sticker_color, STICKER_HSV['green'])

    # 键盘标注状态
    key_contact = False

    prev_gray = None
    frame_id = 0

    print("=" * 55)
    if label_mode == 'sticker':
        print(f"  标注模式: 荧光贴纸（{sticker_color}）自动检测")
        print("  请将手掌平放，贴纸朝上，等待标定（约2秒）...")
    else:
        print("  标注模式: 空格键  [按住=接触 / 松开=抬起]")
    print("  按 Q 结束采集")
    print("=" * 55)

    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = hands.process(frame_rgb)

            # ── 双手分配（简化：右手=书写手，左手=手掌） ─────────────────
            write_lm = palm_lm = None
            if results.multi_hand_landmarks and results.multi_handedness:
                for lm, hd in zip(results.multi_hand_landmarks,
                                  results.multi_handedness):
                    label = hd.classification[0].label
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    if label == 'Left':    # MediaPipe 镜像：Left=右手
                        write_lm = lm
                    else:
                        palm_lm = lm

            # ── 手掌平面更新 ──────────────────────────────────────────────
            palm_sys = palm_tracker.update(palm_lm)

            dist_raw = None
            v_n = sigma_d = v_t = 0.0
            shadow_score = flow_mag = 0.0
            contact_proj = None
            lm_flat = [0.0] * (21 * 3)

            if write_lm and palm_sys:
                tip_3d = get_landmark_3d(write_lm, 8)
                in_bound = palm_sys.is_within_palm_boundary(tip_3d)
                x_palm, y_palm, z_mm = palm_sys.get_2d_coordinates(tip_3d)
                dist_raw = z_mm if in_bound else None

                if dist_raw is not None:
                    feat_buf.push(dist_raw, (x_palm, y_palm), ts)
                    v_n = feat_buf.v_n()
                    sigma_d = feat_buf.sigma_d()
                    v_t = feat_buf.v_t()

                    # 接触投影点（像素坐标）
                    proj = palm_sys.project_to_plane(tip_3d)
                    h, w = frame.shape[:2]
                    cx = int(proj[0] * w)
                    cy = int(proj[1] * h)
                    contact_proj = (cx, cy)
                    shadow_score, flow_mag = _extract_appearance(
                        frame_gray, prev_gray, cx, cy)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                # 提取关键点坐标
                for i, lm_pt in enumerate(write_lm.landmark):
                    lm_flat[i*3+0] = lm_pt.x
                    lm_flat[i*3+1] = lm_pt.y
                    lm_flat[i*3+2] = lm_pt.z

            # ── 贴纸标定 & 标注 ───────────────────────────────────────────
            contact_label = 0

            if label_mode == 'sticker':
                area = _detect_sticker_area(frame_hsv, hsv_lower, hsv_upper)

                if sticker_ref_area is None:
                    calib_areas.append(area)
                    if len(calib_areas) >= STICKER_CALIB_FRAMES:
                        sticker_ref_area = float(np.median(calib_areas))
                        print(f"  [标定完成] 贴纸参考面积: {sticker_ref_area:.0f} px²")
                    cv2.putText(frame, f"Calibrating... {len(calib_areas)}/{STICKER_CALIB_FRAMES}",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    ratio = area / sticker_ref_area if sticker_ref_area > 0 else 1.0
                    contact_label = 1 if ratio < STICKER_OCCLUDE_RATIO else 0
                    color = (0, 255, 0) if contact_label else (0, 0, 255)
                    cv2.putText(frame,
                                f"Sticker: {area}px ({ratio:.2f})  {'CONTACT' if contact_label else 'IDLE'}",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            else:  # keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    key_contact = True
                elif key == 255:
                    pass
                else:
                    key_contact = False
                contact_label = 1 if key_contact else 0
                color = (0, 255, 0) if contact_label else (100, 100, 100)
                cv2.putText(frame,
                            f"[SPACE] {'CONTACT' if contact_label else 'IDLE'}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # ── 调试信息显示 ───────────────────────────────────────────────
            if dist_raw is not None:
                cv2.putText(frame, f"d={dist_raw:.1f}mm  vn={v_n:.2f}  sd={sigma_d:.2f}",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)
                cv2.putText(frame, f"shadow={shadow_score:.1f}  flow={flow_mag:.2f}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)

            cv2.putText(frame, f"Frame {frame_id}  Subject: {subject}",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.imshow("Exp-A1 Collect", frame)

            # ── 写入 CSV ──────────────────────────────────────────────────
            row = {
                'frame_id': frame_id,
                'timestamp': f'{ts:.6f}',
                'contact_label': contact_label,
                'dist_raw': f'{dist_raw:.4f}' if dist_raw is not None else '',
                'v_n': f'{v_n:.4f}',
                'sigma_d': f'{sigma_d:.4f}',
                'v_t': f'{v_t:.4f}',
                'shadow_score': f'{shadow_score:.4f}',
                'flow_mag': f'{flow_mag:.4f}',
            }
            for i in range(21):
                row[f'lm_{i}_x'] = f'{lm_flat[i*3]:.6f}'
                row[f'lm_{i}_y'] = f'{lm_flat[i*3+1]:.6f}'
                row[f'lm_{i}_z'] = f'{lm_flat[i*3+2]:.6f}'
            writer.writerow(row)

            prev_gray = frame_gray.copy()
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"  数据已保存: {out_csv}  ({frame_id} 帧)")
