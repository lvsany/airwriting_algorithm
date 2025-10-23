#!/usr/bin/env python3
"""
Simple data collection helper.

Opens the default camera, shows a random 4-digit code in the preview window,
waits a short preview period (default 2s) and then records for a fixed duration
(default 5s). The recorded video is saved to the output directory with the
filename equal to the 4-digit code (e.g. "0123.mp4").

Usage examples:
  python extern/data_collect.py
  python extern/data_collect.py --duration 5 --preview 1 --out data/test_videos

Requirements:
  - OpenCV (cv2).

This file was added to provide a reproducible data capture tool for
labelling short video clips by a displayed code.
"""
import argparse
import math
import os
import random
import time
from pathlib import Path

import cv2


def make_code() -> str:
    return f"{random.randint(0, 9999):04d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_filename(path: Path) -> Path:
    """If path exists, append a suffix to avoid overwrite."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def overlay_text(frame, text, font_scale=3.0, thickness=4, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    # draw outline for readability
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def run_one_capture(cap: cv2.VideoCapture, out_path: Path, duration: float, preview: float, fps: float, auto_start: bool, codec: str, countdown: int = 3) -> bool:
    """Run one cycle of preview+record. Returns True to continue, False to stop (user pressed 'q')."""
    ensure_dir(out_path)

    code = make_code()
    filename = f"cly_{code}.mp4"
    filepath = unique_filename(out_path / filename)

    # Try to read a frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print('Failed to read from camera during initialization for this capture')
        return False

    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    window = 'data_collect'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        # Preview stage
        preview_start = time.time()
        while time.time() - preview_start < preview:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            overlay_text(disp, code, font_scale=3.0)
            remaining = int(preview - (time.time() - preview_start)) + 1
            cv2.putText(disp, f"Preview: {remaining}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('User requested stop (q) during preview')
                return False

        if not auto_start:
            print(f"Ready to record for {duration} seconds. Press SPACE to start, or 'q' to cancel.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                disp = frame.copy()
                overlay_text(disp, code, font_scale=3.0)
                cv2.imshow(window, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    print('User requested stop (q) before recording')
                    return False

        # Countdown before starting recording (big centered numbers)
        if countdown and countdown > 0:
            cd_start = time.time()
            while True:
                elapsed = time.time() - cd_start
                if elapsed >= countdown:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                disp = frame.copy()
                # show the code in smaller text and the countdown big in center
                overlay_text(disp, code, font_scale=2.0)
                rem = int(math.ceil(countdown - elapsed))
                # large red countdown
                overlay_text(disp, str(rem), font_scale=6.0, color=(0, 0, 255), thickness=6)
                cv2.imshow(window, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('User requested stop (q) during countdown')
                    return False

        # Recording stage
        print(f"Recording to {filepath} ...")
        record_start = time.time()
        while time.time() - record_start < duration:
            ret, frame = cap.read()
            if not ret:
                print('Frame read failed during recording')
                break
            disp = frame.copy()
            overlay_text(disp, code, font_scale=3.0)
            # draw countdown
            remaining = max(0, duration - (time.time() - record_start))
            cv2.putText(disp, f"Rec: {remaining:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(frame)
            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('User requested stop (q) during recording')
                return False

        print(f"Saved: {filepath}")
        return True

    finally:
        writer.release()


def parse_args():
    p = argparse.ArgumentParser(description='Collect short labelled video clips by showing a 4-digit code on the preview.')
    p.add_argument('--camera', '-c', type=int, default=1, help='Camera index for cv2.VideoCapture')
    p.add_argument('--out', '-o', type=str, default='data/test_videos', help='Output directory for saved videos')
    p.add_argument('--duration', '-d', type=float, default=7.0, help='Recording duration in seconds')
    p.add_argument('--preview', type=float, default=2.0, help='Preview time before start (seconds)')
    p.add_argument('--fps', type=float, default=60.0, help='Frames per second for output video')
    p.add_argument('--auto', action='store_true', default=True, help='Automatically start recording after preview without waiting for SPACE (default ON). Use --no-auto to disable.')
    p.add_argument('--no-auto', dest='auto', action='store_false', help='Disable automatic start; wait for SPACE to begin each recording')
    p.add_argument('--once', action='store_true', help='Record only a single clip then exit')
    p.add_argument('--countdown', type=int, default=3, help='Show a centered countdown (seconds) before each recording (default 3)')
    p.add_argument('--codec', type=str, default='mp4v', help='FourCC codec for cv2.VideoWriter (default mp4v)')
    return p.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    out_path = Path(args.out)

    try:
        while True:
            cont = run_one_capture(cap, out_path, args.duration, args.preview, args.fps, args.auto, args.codec, countdown=args.countdown)
            if not cont:
                break
            if args.once:
                break
            # small pause between clips to allow the system to settle (and for user to change the scene if needed)
            time.sleep(0.2)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
