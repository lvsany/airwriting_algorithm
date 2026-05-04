"""
Exp-B1 第一阶段采集脚本：
仅采集视频 + 提取特征，contact_label 全部留空，供后续标注脚本填充。
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "experiments" / "data_b"
LAST_SID_FILE = OUT_DIR / ".last_sid"
WINDOW_NAME = "Exp-B1 Collect"

LIGHTING_OPTIONS = {"normal", "low", "side"}
SPEED_OPTIONS = {"slow", "normal", "fast"}
DEFAULT_PROMPT_WORDS = [
    "the", "and", "you", "for", "that", "with", "this", "have", "from", "not",
    "hello", "world", "apple", "water", "music", "happy", "school", "friend", "green", "light",
]

ROI_HALF_SIZE = 18  # 36x36 ROI
SIGMA_WINDOW = 7
PALM_WIDTH_MM = 80.0  # 用于像素到毫米近似换算

CSV_FIELDS = [
    "frame_id",
    "timestamp",
    "contact_label",
    "dist_raw",
    "dist_local",
    "v_n",
    "a_n",
    "sigma_d",
    "v_t",
    "approach_theta",
    "shadow_score",
    "flow_mag",
    "brightness_contact",
    "dist2d_palm_0",
    "dist2d_palm_5",
    "dist2d_palm_9",
    "dist2d_palm_13",
    "dist2d_palm_17",
    "hull_overlap_ratio",
] + [f"lm_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]

FEATURE_KEYS = [
    "dist_raw",
    "dist_local",
    "v_n",
    "a_n",
    "sigma_d",
    "v_t",
    "approach_theta",
    "shadow_score",
    "flow_mag",
    "brightness_contact",
    "dist2d_palm_0",
    "dist2d_palm_5",
    "dist2d_palm_9",
    "dist2d_palm_13",
    "dist2d_palm_17",
    "hull_overlap_ratio",
]


def _fmt_hms(total_sec: float) -> str:
    sec = max(0, int(total_sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_text(frame: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.75,
               thickness: int = 2) -> None:
    # 黑描边 + 白/红主体，提升强光与暗光场景可读性
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _blank_feature_dict(reset_motion: bool = False) -> Dict[str, object]:
    out: Dict[str, object] = {k: "" for k in FEATURE_KEYS}
    out["_roi_gray"] = None
    out["_dist_raw_num"] = None
    out["_v_n_num"] = None
    if reset_motion:
        extract_features._prev_tip_mm = None
        extract_features._prev_tip_px = None
        extract_features._prev_ts = None
    return out


def _is_zero_landmark(lm) -> bool:
    return abs(lm.x) < 1e-9 and abs(lm.y) < 1e-9 and abs(lm.z) < 1e-9


def _rel_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_prompt_words(raw: str) -> list:
    words = [w.strip() for w in raw.replace("，", ",").replace(";", ",").replace("|", ",").split(",")]
    if len(words) == 1:
        words = [w.strip() for w in raw.split()]
    words = [w for w in words if w]
    return words


def parse_meta() -> dict:
    """终端交互，返回 {sid, lighting, speed, cam_id}。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    last_sid = ""
    if LAST_SID_FILE.exists():
        last_sid = LAST_SID_FILE.read_text(encoding="utf-8").strip()

    while True:
        sid_raw = input("受试者编号（如 s02，直接回车使用上次记录）: ").strip()
        sid = sid_raw or last_sid
        if sid:
            break
        print("受试者编号不能为空，请重新输入。")

    while True:
        lighting = input("光照条件 [normal / low / side]: ").strip().lower()
        if lighting in LIGHTING_OPTIONS:
            break
        print("光照条件仅支持 normal / low / side，请重新输入。")

    while True:
        speed = input("书写速度 [slow / normal / fast]: ").strip().lower()
        if speed in SPEED_OPTIONS:
            break
        print("书写速度仅支持 slow / normal / fast，请重新输入。")

    while True:
        cam_raw = input("摄像头编号（默认 0，直接回车跳过）: ").strip()
        if not cam_raw:
            cam_id = 0
            break
        try:
            cam_id = int(cam_raw)
            break
        except ValueError:
            print("摄像头编号必须是整数，请重新输入。")

    words_raw = input("提示词（空格/逗号分隔，直接回车使用默认常用词）: ").strip()
    prompt_words = _parse_prompt_words(words_raw) if words_raw else list(DEFAULT_PROMPT_WORDS)

    video_path = OUT_DIR / f"exp_b1_{sid}_{lighting}_{speed}.mp4"
    csv_path = OUT_DIR / f"exp_b1_{sid}_{lighting}_{speed}_features.csv"

    print("\n---")
    print(f"即将开始采集：{sid} · {lighting} · {speed}")
    print("输出文件：")
    print(f"  视频：{_rel_path(video_path)}")
    print(f"  特征：{_rel_path(csv_path)}")
    print(f"  提示词数量：{len(prompt_words)}（当前：{prompt_words[0]}）")
    print("---")
    start_cmd = input("按 Enter 开始，输入 q 退出: ").strip().lower()
    if start_cmd == "q":
        raise SystemExit("已退出采集。")

    # 记住上次 sid 仅用于便捷输入，不应影响主流程
    LAST_SID_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        LAST_SID_FILE.write_text(sid, encoding="utf-8")
    except OSError as exc:
        print(f"[提示] 保存上次受试者编号失败（不影响采集）：{exc}")
    return {
        "sid": sid,
        "lighting": lighting,
        "speed": speed,
        "cam_id": cam_id,
        "prompt_words": prompt_words,
    }


def init_mediapipe():
    """初始化 MediaPipe Hands。"""
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def identify_hands(results) -> tuple:
    """
    从 MediaPipe 结果中区分 write_hand 和 palm_hand：
    lm8 z 更小（更靠近摄像头）的手视为 write_hand。
    """
    if results is None or not results.multi_hand_landmarks:
        return None, None
    if len(results.multi_hand_landmarks) < 2:
        return None, None

    candidates = list(results.multi_hand_landmarks)[:2]
    candidates.sort(key=lambda lms: float(lms.landmark[8].z))
    write_lms, palm_lms = candidates[0], candidates[1]

    if _is_zero_landmark(write_lms.landmark[8]):
        return None, None
    return write_lms, palm_lms


def extract_features(write_lms, palm_lms, prev_roi_gray,
                     frame_gray, frame_w, frame_h,
                     prev_dist_raw, prev_v_n,
                     dist_raw_buffer: deque) -> dict:
    """
    提取所有特征。
    检测丢失时返回全空字符串字段；v_n/a_n/sigma_d/v_t 在不可计算时同样留空。
    """
    if not hasattr(extract_features, "_prev_tip_mm"):
        extract_features._prev_tip_mm = None
        extract_features._prev_tip_px = None
        extract_features._prev_ts = None

    if write_lms is None or palm_lms is None:
        dist_raw_buffer.clear()
        return _blank_feature_dict(reset_motion=True)

    tip_lm = write_lms.landmark[8]
    if _is_zero_landmark(tip_lm):
        dist_raw_buffer.clear()
        return _blank_feature_dict(reset_motion=True)

    # 使用掌宽估计 px->mm 比例，保证不同距离下量纲稳定
    p5 = palm_lms.landmark[5]
    p17 = palm_lms.landmark[17]
    p5_px = np.array([p5.x * frame_w, p5.y * frame_h], dtype=np.float64)
    p17_px = np.array([p17.x * frame_w, p17.y * frame_h], dtype=np.float64)
    palm_width_px = float(np.linalg.norm(p5_px - p17_px))
    if palm_width_px < 1e-6:
        dist_raw_buffer.clear()
        return _blank_feature_dict(reset_motion=True)
    mm_per_px = float(np.clip(PALM_WIDTH_MM / palm_width_px, 0.05, 5.0))

    def lm_to_mm(lm) -> np.ndarray:
        # MediaPipe z 与 x 同尺度，按图像宽度换算到像素后再转 mm
        return np.array(
            [lm.x * frame_w * mm_per_px, lm.y * frame_h * mm_per_px, lm.z * frame_w * mm_per_px],
            dtype=np.float64,
        )

    tip_mm = lm_to_mm(tip_lm)
    p0_mm = lm_to_mm(palm_lms.landmark[0])
    p5_mm = lm_to_mm(p5)
    p17_mm = lm_to_mm(p17)

    palm_normal = np.cross(p5_mm - p0_mm, p17_mm - p0_mm)
    n_norm = float(np.linalg.norm(palm_normal))
    if n_norm < 1e-8:
        dist_raw_buffer.clear()
        return _blank_feature_dict(reset_motion=True)
    palm_normal /= n_norm

    dist_raw = float(tip_mm[2] - p0_mm[2])
    dist_local = float(np.dot(tip_mm - p0_mm, palm_normal))

    v_n_num: Optional[float] = None
    v_n_str = ""
    a_n_str = ""
    if prev_dist_raw is not None:
        v_n_num = float(dist_raw - prev_dist_raw)
        v_n_str = f"{v_n_num:.4f}"
        if prev_v_n is not None:
            a_n_str = f"{(v_n_num - prev_v_n):.4f}"

    dist_raw_buffer.append(dist_raw)
    sigma_d_str = f"{float(np.std(dist_raw_buffer)):.4f}" if len(dist_raw_buffer) >= 2 else ""

    tip_px = np.array([tip_lm.x * frame_w, tip_lm.y * frame_h], dtype=np.float64)

    approach_theta_str = ""
    prev_tip_mm = extract_features._prev_tip_mm
    if prev_tip_mm is not None:
        motion = tip_mm - prev_tip_mm
        motion_norm = float(np.linalg.norm(motion))
        if motion_norm > 1e-8:
            cos_theta = float(np.dot(motion, palm_normal) / motion_norm)
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            approach_theta_str = f"{np.degrees(np.arccos(cos_theta)):.4f}"

    v_t_str = ""
    now_ts = time.time()
    prev_tip_px = extract_features._prev_tip_px
    prev_ts = extract_features._prev_ts
    if prev_tip_px is not None and prev_ts is not None:
        dt = now_ts - prev_ts
        if dt > 1e-6:
            dp = tip_px - prev_tip_px
            # Tangential projection on palm plane directions (2D approximation)
            t1 = p5_mm - p0_mm
            t1_norm = float(np.linalg.norm(t1))
            tangential_px = float(np.linalg.norm(dp))
            if t1_norm > 1e-8:
                t1 = t1 / t1_norm
                t2 = np.cross(palm_normal, t1)
                t1_xy, t2_xy = t1[:2], t2[:2]
                n1, n2 = float(np.linalg.norm(t1_xy)), float(np.linalg.norm(t2_xy))
                if n1 > 1e-8 and n2 > 1e-8:
                    u1, u2 = t1_xy / n1, t2_xy / n2
                    tangential_px = float(np.hypot(np.dot(dp, u1), np.dot(dp, u2)))
            v_t = tangential_px * mm_per_px / dt
            v_t_str = f"{v_t:.4f}"

    tip_x, tip_y = float(tip_px[0]), float(tip_px[1])
    dist2d = {}
    for idx in (0, 5, 9, 13, 17):
        p = palm_lms.landmark[idx]
        px = float(p.x * frame_w)
        py = float(p.y * frame_h)
        dist2d[idx] = float(np.hypot(tip_x - px, tip_y - py))

    hull_ratio_str = ""
    pts_w = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in write_lms.landmark], dtype=np.float32)
    pts_p = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in palm_lms.landmark], dtype=np.float32)
    if pts_w.shape[0] >= 3 and pts_p.shape[0] >= 3:
        hull_w = cv2.convexHull(pts_w)
        hull_p = cv2.convexHull(pts_p)
        area_w = float(cv2.contourArea(hull_w))
        area_p = float(cv2.contourArea(hull_p))
        if area_w > 1e-6 and area_p > 1e-6:
            inter_area, _ = cv2.intersectConvexConvex(hull_w, hull_p)
            denom = area_w + area_p
            if denom > 1e-6:
                hull_ratio_str = f"{(float(inter_area) / denom):.6f}"

    cx = int(round(tip_x))
    cy = int(round(tip_y))
    x1 = max(cx - ROI_HALF_SIZE, 0)
    x2 = min(cx + ROI_HALF_SIZE, frame_w)
    y1 = max(cy - ROI_HALF_SIZE, 0)
    y2 = min(cy + ROI_HALF_SIZE, frame_h)
    roi = frame_gray[y1:y2, x1:x2]

    shadow_str = ""
    flow_mag_str = ""
    brightness_str = ""
    roi_to_store = None
    if roi.size > 0:
        shadow_str = f"{float(cv2.Laplacian(roi, cv2.CV_64F).var()):.4f}"
        brightness_str = f"{float(np.mean(roi)):.4f}"
        roi_to_store = roi.copy()
        if prev_roi_gray is not None and prev_roi_gray.shape == roi.shape and roi.shape[0] > 4 and roi.shape[1] > 4:
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi_gray,
                roi,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            flow_mag_str = f"{float(np.mean(np.linalg.norm(flow, axis=2))):.4f}"

    extract_features._prev_tip_mm = tip_mm
    extract_features._prev_tip_px = tip_px
    extract_features._prev_ts = now_ts

    return {
        "dist_raw": f"{dist_raw:.4f}",
        "dist_local": f"{dist_local:.4f}",
        "v_n": v_n_str,
        "a_n": a_n_str,
        "sigma_d": sigma_d_str,
        "v_t": v_t_str,
        "approach_theta": approach_theta_str,
        "shadow_score": shadow_str,
        "flow_mag": flow_mag_str,
        "brightness_contact": brightness_str,
        "dist2d_palm_0": f"{dist2d[0]:.4f}",
        "dist2d_palm_5": f"{dist2d[5]:.4f}",
        "dist2d_palm_9": f"{dist2d[9]:.4f}",
        "dist2d_palm_13": f"{dist2d[13]:.4f}",
        "dist2d_palm_17": f"{dist2d[17]:.4f}",
        "hull_overlap_ratio": hull_ratio_str,
        "_roi_gray": roi_to_store,
        "_dist_raw_num": dist_raw,
        "_v_n_num": v_n_num,
    }


def draw_overlay(frame, meta, frame_id, elapsed_sec,
                 detected, feature_dict, current_word: str, word_pos: Tuple[int, int]):
    """在 frame 上绘制采集状态文字叠加（风格参考 Exp-A）。"""
    white = (255, 255, 255)
    red = (0, 0, 255)
    info = (200, 200, 50)
    gray = (150, 150, 150)

    def _f(key: str) -> Optional[float]:
        v = feature_dict.get(key, "")
        if v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    y0, dy = 40, 34
    line1 = f"REC    {_fmt_hms(elapsed_sec)}  |  Frame: {frame_id}"
    _draw_text(frame, line1, (20, y0), white, scale=0.78, thickness=2)
    cv2.circle(frame, (89, y0 - 8), 7, red, -1)

    line2 = f"{meta['sid']} | {meta['lighting']} | {meta['speed']}"
    _draw_text(frame, line2, (20, y0 + dy), white, scale=0.72, thickness=2)

    line3 = "MediaPipe: OK (both hands)" if detected else "MediaPipe: FAIL (both hands)"
    _draw_text(frame, line3, (20, y0 + 2 * dy), white if detected else red, scale=0.72, thickness=2)

    dist_raw = _f("dist_raw")
    v_n = _f("v_n")
    sigma_d = _f("sigma_d")
    if dist_raw is not None:
        vn_text = f"{v_n:.2f}" if v_n is not None else "--"
        sd_text = f"{sigma_d:.2f}" if sigma_d is not None else "--"
        _draw_text(
            frame,
            f"d={dist_raw:.1f}mm  vn={vn_text}  sd={sd_text}",
            (20, y0 + 3 * dy),
            info,
            scale=0.66,
            thickness=2,
        )

    shadow = _f("shadow_score")
    flow_mag = _f("flow_mag")
    if shadow is not None or flow_mag is not None:
        shadow_text = f"{shadow:.1f}" if shadow is not None else "--"
        flow_text = f"{flow_mag:.2f}" if flow_mag is not None else "--"
        _draw_text(
            frame,
            f"shadow={shadow_text}  flow={flow_text}",
            (20, y0 + 4 * dy),
            info,
            scale=0.66,
            thickness=2,
        )

    d0 = feature_dict.get("dist2d_palm_0", "")
    theta = feature_dict.get("approach_theta", "")
    if detected and d0 and theta:
        line4 = f"dist2d_palm_0: {int(round(float(d0)))}px  approach_theta: {float(theta):.1f}deg"
        _draw_text(frame, line4, (20, y0 + 5 * dy), white, scale=0.66, thickness=2)

    _draw_text(
        frame,
        f"Target [{word_pos[0]}/{word_pos[1]}]: {current_word}",
        (20, y0 + 6 * dy),
        (120, 255, 180),
        scale=0.72,
        thickness=2,
    )
    _draw_text(
        frame,
        "Keys: n-next  p-prev  q-stop",
        (20, y0 + 7 * dy),
        gray,
        scale=0.60,
        thickness=1,
    )

    _draw_text(
        frame,
        f"Frame {frame_id}  Subject: {meta['sid']}",
        (20, frame.shape[0] - 20),
        gray,
        scale=0.62,
        thickness=1,
    )

    tip = "Press q to stop and save"
    (tw, th), _ = cv2.getTextSize(tip, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
    x = (frame.shape[1] - tw) // 2
    y = frame.shape[0] - max(24, th + 20)
    _draw_text(frame, tip, (x, y), white, scale=0.78, thickness=2)
    return frame


def main():
    """主流程：交互→初始化→采集循环→保存。"""
    meta = parse_meta()
    sid = meta["sid"]
    lighting = meta["lighting"]
    speed = meta["speed"]
    prompt_words = meta["prompt_words"]
    word_idx = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"exp_b1_{sid}_{lighting}_{speed}"
    video_path = OUT_DIR / f"{base_name}.mp4"
    csv_path = OUT_DIR / f"{base_name}_features.csv"
    meta_path = OUT_DIR / f"{base_name}_meta.json"

    hands = init_mediapipe()
    cap = cv2.VideoCapture(meta["cam_id"])
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {meta['cam_id']}")

    # 某些设备（尤其是 macOS 连续互通相机）会出现“能打开但首帧读不到”的情况
    # Warm-up several reads before entering capture loop.
    first_frame = None
    for _ in range(45):
        ok, probe = cap.read()
        if ok and probe is not None and probe.size > 0:
            first_frame = probe
            break
        time.sleep(0.03)
    if first_frame is None:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        raise RuntimeError(
            f"摄像头 {meta['cam_id']} 已打开，但连续读取首帧失败。"
            "请尝试摄像头编号 0/1/2，或关闭占用摄像头的软件后重试。"
        )

    fps_cam = float(cap.get(cv2.CAP_PROP_FPS))
    if fps_cam <= 1e-3:
        fps_cam = 30.0

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    video_writer = None
    prev_roi_gray = None
    prev_dist_raw = None
    prev_v_n = None
    dist_raw_buffer = deque(maxlen=SIGMA_WINDOW)

    total_frames = 0
    dual_detect_frames = 0
    feature_success_frames = 0
    start_time = time.time()
    pending_frame = first_frame

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(CSV_FIELDS)

        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                ret = True
            else:
                ret, frame = cap.read()
            if not ret or frame is None:
                break

            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_cam,
                    (w, h),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"无法创建视频文件: {video_path}")

            ts = time.time()
            frame_h, frame_w = frame.shape[:2]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            write_lms, palm_lms = identify_hands(results)
            detected = write_lms is not None and palm_lms is not None
            if detected:
                dual_detect_frames += 1

            feature_dict = extract_features(
                write_lms=write_lms,
                palm_lms=palm_lms,
                prev_roi_gray=prev_roi_gray,
                frame_gray=frame_gray,
                frame_w=frame_w,
                frame_h=frame_h,
                prev_dist_raw=prev_dist_raw,
                prev_v_n=prev_v_n,
                dist_raw_buffer=dist_raw_buffer,
            )

            if feature_dict["_dist_raw_num"] is not None:
                feature_success_frames += 1

            lm_row = []
            if write_lms is not None and not _is_zero_landmark(write_lms.landmark[8]):
                for i in range(21):
                    lm = write_lms.landmark[i]
                    lm_row.extend([f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"])
            else:
                lm_row = [""] * (21 * 3)

            row = [
                total_frames,
                f"{ts:.6f}",
                "",  # contact_label：第一阶段全部留空
                feature_dict["dist_raw"],
                feature_dict["dist_local"],
                feature_dict["v_n"],
                feature_dict["a_n"],
                feature_dict["sigma_d"],
                feature_dict["v_t"],
                feature_dict["approach_theta"],
                feature_dict["shadow_score"],
                feature_dict["flow_mag"],
                feature_dict["brightness_contact"],
                feature_dict["dist2d_palm_0"],
                feature_dict["dist2d_palm_5"],
                feature_dict["dist2d_palm_9"],
                feature_dict["dist2d_palm_13"],
                feature_dict["dist2d_palm_17"],
                feature_dict["hull_overlap_ratio"],
            ] + lm_row

            writer.writerow(row)
            f_csv.flush()
            video_writer.write(frame)

            prev_roi_gray = feature_dict["_roi_gray"]
            prev_dist_raw = feature_dict["_dist_raw_num"]
            prev_v_n = feature_dict["_v_n_num"]

            elapsed_sec = time.time() - start_time
            frame_show = frame.copy()
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_show,
                        hand_lms,
                        hand_connections,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            if write_lms is not None and not _is_zero_landmark(write_lms.landmark[8]):
                tip_x = int(round(write_lms.landmark[8].x * frame_w))
                tip_y = int(round(write_lms.landmark[8].y * frame_h))
                cv2.circle(frame_show, (tip_x, tip_y), 6, (0, 0, 255), -1)

            frame_show = draw_overlay(
                frame_show,
                meta,
                total_frames,
                elapsed_sec,
                detected,
                feature_dict,
                current_word=prompt_words[word_idx],
                word_pos=(word_idx + 1, len(prompt_words)),
            )
            cv2.imshow(WINDOW_NAME, frame_show)

            total_frames += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                word_idx = (word_idx + 1) % len(prompt_words)
            if key == ord("p"):
                word_idx = (word_idx - 1) % len(prompt_words)

    duration_sec = max(0.0, time.time() - start_time)
    fps_actual = (total_frames / duration_sec) if duration_sec > 1e-6 else 0.0
    detect_rate = (dual_detect_frames / total_frames) if total_frames > 0 else 0.0
    lost_frames = total_frames - feature_success_frames

    meta_json = {
        "sid": sid,
        "lighting": lighting,
        "speed": speed,
        "total_frames": int(total_frames),
        "fps_actual": round(float(fps_actual), 1),
        "duration_sec": int(round(duration_sec)),
        "detect_rate": round(float(detect_rate), 3),
        "prompt_word_count": len(prompt_words),
        "prompt_words": prompt_words,
        "collect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta_json, ensure_ascii=False, indent=2), encoding="utf-8")

    cap.release()
    if video_writer is not None:
        video_writer.release()
    hands.close()
    cv2.destroyAllWindows()

    dur_int = int(round(duration_sec))
    mm = dur_int // 60
    ss = dur_int % 60

    print("\n========== 采集完成 ==========")
    print(f"受试者：{sid}  光照：{lighting}  速度：{speed}")
    print(f"总帧数：{total_frames}  时长：{mm}分{ss:02d}秒  实际帧率：{fps_actual:.1f} fps")
    print(f"MediaPipe 双手检测率：{detect_rate * 100:.1f}%（{dual_detect_frames}帧）")
    print(f"特征提取成功帧数：{feature_success_frames}帧")
    print(f"检测丢失帧数：{lost_frames}帧（全部写为空字符串，非0）")
    print("--------------------------------")
    print("文件已保存：")
    print(f"  视频：{_rel_path(video_path)}")
    print(f"  特征：{_rel_path(csv_path)}")
    print(f"  元数据：{_rel_path(meta_path)}")
    print("==============================")


if __name__ == "__main__":
    main()
