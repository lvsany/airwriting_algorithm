#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-air handwriting data collection for mobile/emergency annotation scenarios.

Updates:
- Stimulus text is shown on screen but NOT saved into the video file.
- During recording, we write the CLEAN raw frame (no stimulus, no labels, no trajectory).
- Window overlays (stimulus, countdown, block/mode) remain for guidance.
- Filename = safe(stimulus text). CSV log preserved. All on-screen prompts in ENGLISH.

Blocks (set BLOCK=1..4):
1) No time pressure (15s) + Continuous pinch (no release)
2) Time pressure (7s)     + Continuous pinch (no release)
3) No time pressure (15s) + Pinch with releases (paper-like)
4) Time pressure (7s)     + Pinch with releases (paper-like)
"""

import os
import sys
import math
import time
import random
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import cv2

# =========================
# ===== Global Config =====
# =========================
BLOCK = 1                 # 1..4
CAMERA_INDEX = 1
OUT_DIR       = "data/test_videos"
FPS           = 60.0
CODEC         = "mp4v"
PREVIEW_SEC   = 2.0
COUNTDOWN_SEC = 3
WINDOW_NAME   = "data_collect"

# Optional external phrase list (one per line, 2–5 words)
PHRASES_FILE  = "extern/phrases_emergency_en.txt"
# =========================


def _block_cfg(block_id: int) -> Tuple[float, str, str]:
    if block_id == 1:
        return 15.0, "No time pressure (15s)", "Continuous pinch (no release)"
    if block_id == 2:
        return 7.0,  "Time pressure (7s)",     "Continuous pinch (no release)"
    if block_id == 3:
        return 15.0, "No time pressure (15s)", "Pinch with releases (paper-like)"
    if block_id == 4:
        return 7.0,  "Time pressure (7s)",     "Pinch with releases (paper-like)"
    return 15.0, "No time pressure (15s)", "Continuous pinch (no release)"


# ===== Emergency/mobile English phrases (2–5 words) =====
_BUILTIN_PHRASES = [
    "report incident location",
    "mark blocked sidewalk",
    "note slippery floor",
    "record license plate",
    "share live situation",
    "need medical help",
    "crowd moving fast",
    "exit route here",
    "avoid this area",
    "request backup now",
    "call emergency line",
    "suspect heading north",
    "road closed ahead",
    "temporary safe zone",
    "hazard on road",
    "meeting point moved",
    "short power outage",
    "signal is weak",
    "send quick update",
    "mark first aid",
    "stairs not safe",
    "use side entrance",
    "traffic standstill",
    "reroute immediately",
    "check camera feed",
    "bridge is crowded",
    "keep distance please",
    "watch falling objects",
    "wind gusts rising",
    "flash flood risk",
    "report new symptom",
    "mask required here",
    "keep right side",
    "left lane blocked",
    "elevator out service",
    "line forms here",
    "need translation help",
    "lost child here",
    "return item found",
    "shelter in place",
    "evacuate level two",
    "stairs to lobby",
    "gate changed now",
    "train delayed again",
    "bus rerouted west",
    "use alternate path",
    "check oxygen level",
    "low battery warning",
    "send proof photo",
    "upload short clip",
    "mute background noise",
    "note time and place",
    "mark entrance A",
    "meet at corner",
    "arriving in five",
    "two minutes late",
    "on foot now",
    "crossing the street",
    "at next stop"
]

def _load_phrases_from_file(path: str, min_len=2, max_len=5) -> List[str]:
    f = Path(path)
    if not f.exists():
        return []
    phrases = []
    with f.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if 2 <= len(s.split()) <= max_len:
                phrases.append(s.lower())
    return phrases

def build_stimuli_for_block() -> List[str]:
    pool = _load_phrases_from_file(PHRASES_FILE)
    if not pool:
        pool = _BUILTIN_PHRASES[:]
    k = min(6, len(pool))
    phrases = random.sample(pool, k=k) if k > 0 else []

    digit_lens = [3, 4]
    digits = [f"{random.randint(0, 10**random.choice(digit_lens)-1):0{random.choice(digit_lens)}d}" for _ in range(6)]

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{};:,.?/\\|"
    rndchars = ["".join(random.choice(alphabet) for _ in range(random.randint(3,5))) for _ in range(3)]

    stimuli = phrases + digits + rndchars
    random.shuffle(stimuli)
    while len(stimuli) < 15:
        stimuli.append(random.choice(_BUILTIN_PHRASES))
    return stimuli[:15]


# ===== Utilities =====
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def unique_filename(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def safe_filename_from_text(text: str, max_len: int = 80) -> str:
    """
    Convert stimulus text to a safe ASCII filename stem:
    - normalize to NFKD, strip accents
    - lowercase
    - spaces -> underscores
    - keep [a-z0-9._-], remove others
    - collapse multiple underscores/dots/dashes
    - trim to max_len
    """
    norm = unicodedata.normalize("NFKD", text)
    ascii_str = norm.encode("ascii", "ignore").decode("ascii")
    s = ascii_str.lower().strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9._-]", "", s)
    s = re.sub(r"[_\-\.]{2,}", lambda m: m.group(0)[0], s)
    s = s.strip("._-")
    if not s:
        s = "untitled"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s


def overlay_center_text(frame, text, font_scale=2.2, thickness=5, color=(255,255,255)):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(12, (w - tw) // 2)
    y = (h + th) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, (0,0,0), thickness + 6, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def overlay_label(frame, text, x=12, y=32):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)


# ===== One trial =====
def run_one_trial(cap: cv2.VideoCapture,
                  out_dir: Path,
                  block_id: int,
                  trial_idx: int,
                  stimulus: str,
                  duration: float,
                  preview: float,
                  fps: float,
                  codec: str,
                  countdown: int,
                  pressure_label: str,
                  pinch_label: str) -> Path:
    """
    Preview -> countdown -> record.
    - Window shows overlays (stimulus/labels/countdown/trajectory).
    - Video file stores CLEAN frames only (no overlays at all).
    - Filename equals safe(stimulus).
    """
    ensure_dir(out_dir)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    H, W = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    stem = safe_filename_from_text(stimulus)
    save_path = unique_filename(out_dir / f"{stem}.mp4")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (W, H))

    # Optional detector (for on-screen trajectory ONLY)
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from hand_track.finger_tracking import HandWritingDetector, smart_smooth
        detector = HandWritingDetector()
    except Exception:
        detector, smart_smooth = None, None

    trajectory = []
    history_points = []
    prev_index_pos = None
    MAX_TRAJECTORY_POINTS = 2048

    def process_and_draw(process_frame, draw_frame):
        """Run detector on clean frame; draw trajectory ONLY on draw_frame."""
        nonlocal prev_index_pos, trajectory, history_points
        if detector is None:
            return
        is_writing = detector.process(process_frame)
        current_pos = detector.index_tip_position
        if is_writing and current_pos != (0, 0):
            if prev_index_pos is not None:
                smoothed = smart_smooth(current_pos, prev_index_pos, history_points)
                prev_index_pos = smoothed
                trajectory.append(smoothed)
            else:
                trajectory.append(None)
                smoothed = current_pos
                prev_index_pos = current_pos
                trajectory.append(smoothed)
            cv2.circle(draw_frame, smoothed, 12, (0, 0, 255), -1)
        else:
            prev_index_pos = None
            history_points.clear()
        for i in range(1, len(trajectory)):
            if trajectory[i-1] is not None and trajectory[i] is not None:
                cv2.line(draw_frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
        if len(trajectory) > MAX_TRAJECTORY_POINTS:
            trajectory = trajectory[-MAX_TRAJECTORY_POINTS:]

    # ===== Preview (not recorded) =====
    t0 = time.time()
    while time.time() - t0 < preview:
        ok, frame = cap.read()
        if not ok:
            break
        clean = frame  # keep clean for potential processing
        disp = frame.copy()
        overlay_center_text(disp, stimulus, font_scale=2.0)
        overlay_label(disp, f"Block: {pressure_label} | Mode: {pinch_label}", 12, 32)
        remain = int(preview - (time.time() - t0)) + 1
        overlay_label(disp, f"Preview: {remain}s", 12, 64)
        process_and_draw(clean, disp)
        cv2.imshow(WINDOW_NAME, disp)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            writer.release()
            raise KeyboardInterrupt("User aborted")

    # ===== Countdown (not recorded) =====
    if countdown and countdown > 0:
        cd_start = time.time()
        while True:
            elapsed = time.time() - cd_start
            if elapsed >= countdown:
                break
            ok, frame = cap.read()
            if not ok:
                break
            clean = frame
            disp = frame.copy()
            overlay_center_text(disp, stimulus, font_scale=1.9)
            rem = int(math.ceil(countdown - elapsed))
            (tw, th), _ = cv2.getTextSize(str(rem), cv2.FONT_HERSHEY_SIMPLEX, 6.0, 6)
            cx, cy = (disp.shape[1] - tw) // 2, (disp.shape[0] + th) // 2
            cv2.putText(disp, str(rem), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 255), 6, cv2.LINE_AA)
            overlay_label(disp, f"Block: {pressure_label} | Mode: {pinch_label}", 12, 32)
            process_and_draw(clean, disp)
            cv2.imshow(WINDOW_NAME, disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                writer.release()
                raise KeyboardInterrupt("User aborted")

    # ===== Recording (SAVE CLEAN FRAMES) =====
    rec_start = time.time()
    while time.time() - rec_start < duration:
        ok, frame = cap.read()
        if not ok:
            break
        clean = frame                    # what we save
        disp = frame.copy()              # what we show (with overlays)
        overlay_center_text(disp, stimulus, font_scale=1.9)
        remain = max(0.0, duration - (time.time() - rec_start))
        overlay_label(disp, f"REC: {remain:.1f}s", 12, 64)
        overlay_label(disp, f"Block: {pressure_label} | Mode: {pinch_label}", 12, 32)
        process_and_draw(clean, disp)

        writer.write(clean)              # <<< save CLEAN frame only
        cv2.imshow(WINDOW_NAME, disp)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            writer.release()
            raise KeyboardInterrupt("User aborted")

    writer.release()
    return save_path


def main():
    duration, pressure_txt, pinch_txt = _block_cfg(BLOCK)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    out_path = Path(OUT_DIR)
    ensure_dir(out_path)

    stimuli = build_stimuli_for_block()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_path / f"session_{ts}_b{BLOCK}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filename", "block", "pressure", "pinch_mode", "stimulus", "duration_sec", "preview_sec", "fps"])
        try:
            for idx, stim in enumerate(stimuli, start=1):
                saved = run_one_trial(
                    cap=cap,
                    out_dir=out_path,
                    block_id=BLOCK,
                    trial_idx=idx,
                    stimulus=stim,
                    duration=duration,
                    preview=PREVIEW_SEC,
                    fps=FPS,
                    codec=CODEC,
                    countdown=COUNTDOWN_SEC,
                    pressure_label=pressure_txt,
                    pinch_label=pinch_txt
                )
                print(f"[OK] Saved: {saved.name}")
                writer.writerow([saved.name, BLOCK, pressure_txt, pinch_txt, stim, duration, PREVIEW_SEC, FPS])
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("Aborted by user. Partial block saved.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
