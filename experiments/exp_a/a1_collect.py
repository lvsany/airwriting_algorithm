"""
Exp-A1: 纯 tap 数据采集与标注

标注模式：
  sticker  — 荧光贴纸 HSV 颜色面积检测，全自动，误差 < 1 帧
  palmpad  — PalmPad 数据集，读取 .txt 标签文件（每帧 0/1）

输出 CSV 格式（每帧一行）：
  frame_id, timestamp, contact_label,
  dist_raw, dist_local, v_n, a_n, sigma_d, v_t, approach_theta,
  shadow_score, flow_mag, brightness_contact,
  dist2d_palm_{0,5,9,13,17}, hull_overlap_ratio,
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
from src.utils.geometry_utils import get_landmark_3d

# ── HSV 颜色范围（荧光贴纸） ────────────────────────────────────────────────
STICKER_HSV = {
    'green':  ((35,  40,  30),  (85,  255, 255)),
    'yellow': ((20,  100, 100), (35,  255, 255)),
    'pink':   ((140, 80,  80),  (170, 255, 255)),
    'blue':   ((85, 15, 120),   (225, 255, 255)),
    'black':  ((0,   0,   0),   (180, 80,  90)),
}
STICKER_OCCLUDE_RATIO = 0.4   # 面积低于参考值此比例时判定为遮挡
STICKER_CALIB_FRAMES = 30     # 标定所需帧数

DIST_WINDOW  = 5
SIGMA_WINDOW = 5


def _detect_sticker_area(frame_hsv, lower, upper, palm_lm=None):
    mask = cv2.inRange(frame_hsv,
                       np.array(lower, dtype=np.uint8),
                       np.array(upper, dtype=np.uint8))
    has_palm = False
    if palm_lm is not None:
        h, w = frame_hsv.shape[:2]
        # 仅使用掌心/掌根关键点，避免手指区域进入检测
        palm_core_idx = (0, 1, 2, 5, 9, 13, 17)
        palm_pts = np.array(
            [[int(palm_lm.landmark[i].x * w), int(palm_lm.landmark[i].y * h)] for i in palm_core_idx],
            dtype=np.int32
        )
        if palm_pts.shape[0] >= 3:
            has_palm = True
            palm_mask = np.zeros((h, w), dtype=np.uint8)
            palm_hull = cv2.convexHull(palm_pts)
            cv2.fillConvexPoly(palm_mask, palm_hull, 255)
            # 收缩一点边界，进一步抑制指根附近误检
            palm_mask = cv2.erode(
                palm_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
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


def _extract_appearance(frame_gray, prev_gray, cx, cy, radius=18):
    h, w = frame_gray.shape
    x1, x2 = max(cx - radius, 0), min(cx + radius, w)
    y1, y2 = max(cy - radius, 0), min(cy + radius, h)
    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0, 0.0

    shadow = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    brightness = float(np.mean(roi))

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

    return shadow, flow_mag, brightness


def _compute_approach_theta(prev_tip, curr_tip, palm_normal):
    if prev_tip is None or curr_tip is None or palm_normal is None:
        return None
    motion = curr_tip - prev_tip
    m_norm = float(np.linalg.norm(motion))
    n_norm = float(np.linalg.norm(palm_normal))
    if m_norm < 1e-9 or n_norm < 1e-9:
        return None
    cos_n = float(np.dot(motion, palm_normal) / (m_norm * n_norm))
    cos_n = max(-1.0, min(1.0, cos_n))
    # 与掌面夹角：0°=平行滑动，90°=垂直压入
    return float(np.degrees(np.arcsin(abs(cos_n))))


def _compute_tip_to_palm_dist_2d(write_lm, palm_lm, frame_shape):
    if write_lm is None or palm_lm is None:
        return {}
    h, w = frame_shape[:2]
    tip = write_lm.landmark[8]
    tx, ty = float(tip.x * w), float(tip.y * h)
    out = {}
    for idx in (0, 5, 9, 13, 17):
        pt = palm_lm.landmark[idx]
        px, py = float(pt.x * w), float(pt.y * h)
        out[idx] = float(np.hypot(tx - px, ty - py))
    return out


def _compute_hull_overlap_ratio(write_lm, palm_lm, frame_shape):
    if write_lm is None or palm_lm is None:
        return None
    h, w = frame_shape[:2]
    pts_w = np.array([[lm.x * w, lm.y * h] for lm in write_lm.landmark], dtype=np.float32)
    pts_p = np.array([[lm.x * w, lm.y * h] for lm in palm_lm.landmark], dtype=np.float32)
    if pts_w.shape[0] < 3 or pts_p.shape[0] < 3:
        return None

    hull_w = cv2.convexHull(pts_w)
    hull_p = cv2.convexHull(pts_p)
    area_w = float(cv2.contourArea(hull_w))
    area_p = float(cv2.contourArea(hull_p))
    if area_w < 1e-6 or area_p < 1e-6:
        return None

    inter_area, _ = cv2.intersectConvexConvex(hull_w, hull_p)
    inter_area = float(inter_area)
    union_area = area_w + area_p - inter_area
    if union_area < 1e-6:
        return None
    return inter_area / union_area


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


def _normalize_timestamp(raw_ts):
    ts = float(raw_ts)
    if ts > 1e12:
        return ts / 1e6
    if ts > 1e10:
        return ts / 1e3
    return ts


def _load_palmpad_labels(label_file):
    labels = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ts = _normalize_timestamp(parts[0])
            label = int(float(parts[1]))
            labels.append((ts, label))
    return labels


def run_collect(subject: str, data_dir: str,
                label_mode: str = 'sticker',
                sticker_color: str = 'green',
                video_path: str = None,
                label_file: str = None,
                no_vis: bool = False,
                save_raw_video: bool = True):
    if label_mode not in ('sticker', 'palmpad'):
        raise ValueError("Exp-A1 仅支持 sticker / palmpad 标注模式")
    if label_mode == 'palmpad' and not label_file:
        raise ValueError("palmpad 模式必须提供 label_file")

    os.makedirs(data_dir, exist_ok=True)

    detector  = DualHandDetector()
    feat_buf  = FeatureBuffer()

    cap_source = 1 if not video_path or video_path in ('1', 'camera') else video_path
    cap = cv2.VideoCapture(cap_source)
    if isinstance(cap_source, str) and not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {cap_source}")

    out_csv = os.path.join(data_dir, f'exp_a1_{subject}.csv')
    out_video = os.path.join(data_dir, f'exp_a1_{subject}_raw.mp4')
    fieldnames = (
        ['frame_id', 'timestamp', 'contact_label',
         'dist_raw', 'dist_local', 'v_n', 'a_n', 'sigma_d', 'v_t', 'approach_theta',
         'shadow_score', 'flow_mag', 'brightness_contact',
         'dist2d_palm_0', 'dist2d_palm_5', 'dist2d_palm_9', 'dist2d_palm_13', 'dist2d_palm_17',
         'hull_overlap_ratio'] +
        [f'lm_{i}_{ax}' for i in range(21) for ax in ('x', 'y', 'z')]
    )

    hsv_lower, hsv_upper = STICKER_HSV.get(sticker_color, STICKER_HSV['green'])
    label_list = _load_palmpad_labels(label_file) if label_mode == 'palmpad' else []
    sticker_ref_area = None
    calib_areas      = []
    video_writer     = None
    video_fps        = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 1e-3:
        video_fps = 30.0
    prev_gray        = None
    prev_tip_3d      = None
    prev_v_n         = None
    frame_id         = 0

    print("=" * 55)
    if label_mode == 'sticker':
        print(f"  标注模式: 荧光贴纸（{sticker_color}）自动检测")
        print("  请将手掌平放，贴纸朝上，等待标定（约2秒）...")
        print("  按 Q 或 ESC 结束采集")
    else:
        print("  标注模式: PalmPad 标签文件")
        print(f"  标签文件: {label_file}")
        print(f"  视频源: {cap_source}")
        if no_vis:
            print("  可视化: 关闭")
    print("=" * 55)

    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if save_raw_video and video_writer is None:
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

            label_ts = None
            label_val = 0
            if label_mode == 'palmpad':
                if frame_id < len(label_list):
                    label_ts, label_val = label_list[frame_id]
                elif frame_id == len(label_list):
                    print(f"[警告] 标签帧数不足，剩余帧将使用 0 标签")

            ts = label_ts if label_ts is not None else time.time()
            detector.process(frame, ts)

            frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 从 DualHandDetector 读取已计算好的特征 ────────────────────
            palm_sys = detector.palm_tracker.get_current_system()
            tip_3d = get_landmark_3d(detector.write_lm, 8) if detector.write_lm else None
            dist_local = None
            if palm_sys is not None and tip_3d is not None and palm_sys.is_within_palm_boundary(tip_3d):
                dist_local = float(palm_sys.get_distance_to_plane(tip_3d))

            dist_raw = dist_local
            pos_xy   = detector.write_pos_palm if detector.write_pos_palm else (0.0, 0.0)

            v_n = a_n = sigma_d = v_t = 0.0
            approach_theta = None
            shadow_score = flow_mag = brightness_contact = 0.0
            dist2d_group = {}
            hull_overlap_ratio = None
            lm_flat = [0.0] * (21 * 3)

            if dist_raw is not None:
                feat_buf.push(dist_raw, pos_xy, ts)
                v_n     = feat_buf.v_n()
                sigma_d = feat_buf.sigma_d()
                v_t     = feat_buf.v_t()
                if prev_v_n is not None:
                    a_n = float(v_n - prev_v_n)
                prev_v_n = v_n
            else:
                prev_v_n = None

            if palm_sys is not None and tip_3d is not None:
                approach_theta = _compute_approach_theta(prev_tip_3d, tip_3d, palm_sys.plane_normal)

            if detector.write_lm and detector.palm_lm:
                dist2d_group = _compute_tip_to_palm_dist_2d(
                    detector.write_lm, detector.palm_lm, frame.shape
                )
                hull_overlap_ratio = _compute_hull_overlap_ratio(
                    detector.write_lm, detector.palm_lm, frame.shape
                )

            if detector.write_lm:
                # 接触投影点 → 像素坐标，提取外观特征
                px, py = detector.write_pos
                shadow_score, flow_mag, brightness_contact = _extract_appearance(
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
            if label_mode == 'sticker':
                palm_lm_for_sticker = detector.palm_lm or detector.left_lm or detector.right_lm
                area, sticker_contour, has_palm = _detect_sticker_area(
                    frame_hsv, hsv_lower, hsv_upper, palm_lm_for_sticker
                )
                if sticker_contour is not None:
                    cv2.drawContours(frame, [sticker_contour], -1, (255, 0, 255), 2)

                if sticker_ref_area is None:
                    if has_palm:
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
                        contact_label = 0
                        cv2.putText(frame, "Palm not detected",
                                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            else:
                contact_label = 1 if label_val else 0
                if not no_vis:
                    color = (0, 255, 0) if contact_label else (0, 0, 255)
                    cv2.putText(frame,
                                f"PalmPad label: {'CONTACT' if contact_label else 'IDLE'}",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ── 调试信息 ──────────────────────────────────────────────────
            if dist_raw is not None:
                cv2.putText(frame,
                            f"d={dist_raw:.1f}mm  vn={v_n:.2f}  sd={sigma_d:.2f}",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)
                cv2.putText(frame,
                            f"shadow={shadow_score:.1f}  flow={flow_mag:.2f}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)

            if not no_vis:
                # ── 绘制手部骨架 ────────────────────────────────
                h, w = frame.shape[:2]
                
                # 绘制关键点和连接线
                if detector.write_lm or detector.palm_lm:
                    hand_lm = detector.write_lm or detector.palm_lm
                    lm_points = [(lm.x * w, lm.y * h) for lm in hand_lm.landmark]
                    
                    # 连接线（MediaPipe 标准）
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (0, 9), (9, 10), (10, 11), (11, 12),
                        (0, 13), (13, 14), (14, 15), (15, 16),
                        (0, 17), (17, 18), (18, 19), (19, 20)
                    ]
                    
                    for i, j in connections:
                        pt1 = (int(lm_points[i][0]), int(lm_points[i][1]))
                        pt2 = (int(lm_points[j][0]), int(lm_points[j][1]))
                        cv2.line(frame, pt1, pt2, (100, 150, 200), 2)
                    
                    # 关键点圆圈
                    for i, (x, y) in enumerate(lm_points):
                        pt = (int(x), int(y))
                        if i == 8:  # 食指指尖（接触检测点）
                            cv2.circle(frame, pt, 6, (0, 255, 0), -1)
                        else:
                            cv2.circle(frame, pt, 3, (255, 100, 0), -1)
                
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
                'dist_local':   f'{dist_local:.4f}' if dist_local is not None else '',
                'v_n':          f'{v_n:.4f}',
                'a_n':          f'{a_n:.4f}',
                'sigma_d':      f'{sigma_d:.4f}',
                'v_t':          f'{v_t:.4f}',
                'approach_theta': f'{approach_theta:.4f}' if approach_theta is not None else '',
                'shadow_score': f'{shadow_score:.4f}' if detector.write_lm else '',
                'flow_mag':     f'{flow_mag:.4f}'     if detector.write_lm else '',
                'brightness_contact': f'{brightness_contact:.4f}' if detector.write_lm else '',
                'dist2d_palm_0':  f"{dist2d_group[0]:.4f}" if 0 in dist2d_group else '',
                'dist2d_palm_5':  f"{dist2d_group[5]:.4f}" if 5 in dist2d_group else '',
                'dist2d_palm_9':  f"{dist2d_group[9]:.4f}" if 9 in dist2d_group else '',
                'dist2d_palm_13': f"{dist2d_group[13]:.4f}" if 13 in dist2d_group else '',
                'dist2d_palm_17': f"{dist2d_group[17]:.4f}" if 17 in dist2d_group else '',
                'hull_overlap_ratio': f'{hull_overlap_ratio:.6f}' if hull_overlap_ratio is not None else '',
            }
            for i in range(21):
                row[f'lm_{i}_x'] = f'{lm_flat[i*3]:.6f}'
                row[f'lm_{i}_y'] = f'{lm_flat[i*3+1]:.6f}'
                row[f'lm_{i}_z'] = f'{lm_flat[i*3+2]:.6f}'
            writer.writerow(row)
            if video_writer is not None:
                video_writer.write(raw_frame)

            prev_gray = frame_gray.copy()
            prev_tip_3d = tip_3d
            frame_id += 1

            if not no_vis:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

    if label_mode == 'palmpad' and len(label_list) != frame_id:
        print(f"[警告] 标签帧数({len(label_list)})与视频帧数({frame_id})不一致")

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if not no_vis:
        cv2.destroyAllWindows()
    detector.reset()
    print(f"  数据已保存: {out_csv}  ({frame_id} 帧)")
    if save_raw_video:
        print(f"  原视频已保存: {out_video}")
