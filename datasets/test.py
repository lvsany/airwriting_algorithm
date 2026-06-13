"""
Exp 3 Data Collection Script - End-to-End Character & Word Recognition

Two operating modes, selected automatically by --method:

  Camera mode  (own_framework):
      Uses live camera input; saves raw video alongside trajectory JSON.

  Replay mode  (all other methods):
      Reads the raw video already recorded by own_framework, replays it
      frame-by-frame through the selected contact detector, and saves a
      new trajectory JSON — no camera required.
      Trial targets and order are loaded from the own_framework JSON so
      the comparison is fair (same sequence, same video).

Usage:
    # Camera — collect reference data with own framework
    python datasets/test.py --user U01 --method own_framework

    # Replay — run PalmPad on the same video (server, no camera needed)
    python datasets/test.py --user U01 --method palmpad

    # Replay with explicit paths (override auto-detection)
    python datasets/test.py --user U01 --method palmpad \\
        --video datasets/Exp3/own_framework/exp3_U01_raw_*.mp4 \\
        --trials datasets/Exp3/own_framework/exp3_U01.json
"""

import sys
import os
import re
import glob
import cv2
import time
import json
import random
import numpy as np
import argparse

_DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_DATASETS_DIR)
for _p in (_DATASETS_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contact_detectors import REGISTRY, build_detector, ContactDetectorBase

_DEFAULT_CKPT = os.path.join(_DATASETS_DIR, "models", "palmpad", "best.pt")
_EXP3_ROOT    = os.path.join(_DATASETS_DIR, "exp3")
_OWN_DIR      = os.path.join(_EXP3_ROOT, "own_framework")
_VIDEO_DIR    = os.path.join(_DATASETS_DIR, "video")

C = {
    'bg':      (20,  20,  25),
    'text':    (210, 210, 215),
    'accent':  (230, 165, 60),
    'writing': (80,  230, 120),
    'hover':   (200, 180, 80),
}

_SMOOTH_ALPHA = 0.4
_TAIL_EPS_PX  = 4


def _trim_tail(stroke: list) -> list:
    while len(stroke) >= 2:
        dx = stroke[-1]['x'] - stroke[-2]['x']
        dy = stroke[-1]['y'] - stroke[-2]['y']
        if dx * dx + dy * dy <= _TAIL_EPS_PX ** 2:
            stroke = stroke[:-1]
        else:
            break
    return stroke


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _find_latest_video(directory: str, user_id: str):
    """Return the most recently modified raw video for a given user."""
    pattern = os.path.join(directory, f"exp3_{user_id}_raw_*.mp4")
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def _seek_to_level1(cap: cv2.VideoCapture, video_path: str,
                    ref_data: list, buffer_sec: float = 3.0):
    """
    Seek the video to `buffer_sec` seconds before the first LEVEL1 stroke,
    using the Unix timestamp embedded in the video filename and the `t` field
    of the first recorded stroke point.
    """
    first_l1 = next(
        (r for r in ref_data
         if r.get("level") == "LEVEL1" and r.get("strokes") and r["strokes"]),
        None,
    )
    if not first_l1:
        return

    try:
        first_t = first_l1["strokes"][0][0]["t"]
    except (IndexError, KeyError):
        return

    m = re.search(r'_raw_(\d+)\.mp4$', video_path)
    if not m:
        return

    video_start_t = int(m.group(1))
    offset_sec    = max(0.0, first_t - video_start_t - buffer_sec)
    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seek_frame    = int(offset_sec * fps)

    if seek_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        print(f"[Replay] Seeking to frame {seek_frame}  "
              f"({offset_sec:.1f} s into video, {buffer_sec} s buffer before first stroke)")


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class Exp3Collector:
    def __init__(self, user_id: str, detector: ContactDetectorBase,
                 exp3_root: str = _EXP3_ROOT,
                 start_level: str = None,
                 is_replay: bool = False,
                 l1_seq: list = None,
                 l2_seq: list = None,
                 l3_seq: list = None):
        self.user_id   = user_id
        self.detector  = detector
        self.is_replay = is_replay

        self.out_dir   = os.path.join(exp3_root, detector.name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_file  = os.path.join(self.out_dir, f"exp3_{user_id}.json")
        # out_video: camera mode only; videos always go to Exp3/video/
        _vid_dir = os.path.join(exp3_root, "video")
        os.makedirs(_vid_dir, exist_ok=True)
        self.out_video = os.path.join(_vid_dir,
                                      f"exp3_{user_id}_raw_{int(time.time())}.mp4")

        # Workflow states
        self.states    = ["CALIB", "PRACTICE", "LEVEL1", "LEVEL2", "LEVEL3", "DONE"]
        self.state_idx = 0
        self.return_state_idx     = None
        self.initial_target_state = start_level

        # Jump past CALIB when detector needs no calibration or in replay mode
        if not detector.needs_calibration or is_replay:
            target = (start_level or ("LEVEL1" if is_replay else "PRACTICE")).upper()
            try:
                self.state_idx = self.states.index(target)
            except ValueError:
                self.state_idx = 1  # PRACTICE
            self.initial_target_state = None

        # Load existing data (resume support)
        self.all_data = []
        if os.path.exists(self.out_file):
            try:
                with open(self.out_file, "r", encoding="utf-8") as f:
                    self.all_data = json.load(f)
                print(f"[INFO] Resumed {len(self.all_data)} records from {self.out_file}")
            except Exception as e:
                print(f"[WARN] Could not load existing data: {e}")

        # Trial sequences — use provided (replay) or generate randomly (camera)
        self._fixed_sequences = (l1_seq is not None)

        if l1_seq is not None:
            self.l1_targets = list(l1_seq)
        else:
            chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            sampled = random.sample(chars, 10)
            self.l1_targets = sampled * 2
            random.shuffle(self.l1_targets)

        if l2_seq is not None:
            self.l2_targets = list(l2_seq)
        else:
            words_file = os.path.join(_DATASETS_DIR, "words.txt")
            try:
                with open(words_file, "r", encoding="utf-8") as f:
                    words = [ln.strip() for ln in f
                             if ln.strip() and not ln.startswith("#")]
            except FileNotFoundError:
                words = ["hello", "world", "test", "testing", "longest"]
            words_l2 = [w for w in words if 3 <= len(w) <= 5] or ["hello", "world"]
            self.l2_targets = random.sample(words_l2, min(15, len(words_l2)))

        if l3_seq is not None:
            self.l3_targets = list(l3_seq)
        else:
            if not hasattr(self, '_words_loaded'):
                words_file = os.path.join(_DATASETS_DIR, "words.txt")
                try:
                    with open(words_file, "r", encoding="utf-8") as f:
                        words = [ln.strip() for ln in f
                                 if ln.strip() and not ln.startswith("#")]
                except FileNotFoundError:
                    words = ["hello", "world", "test", "testing", "longest"]
            words_l3 = [w for w in words if len(w) >= 6] or ["testing", "longest"]
            self.l3_targets = random.sample(words_l3, min(5, len(words_l3)))

        self.current_trial_idx = 0

        # Trajectory state
        self.current_strokes = []
        self.current_stroke  = []
        self.prev_writing    = False

        # Keyboard event log (camera mode only)
        self.keyboard_events = []
        self.events_file     = self.out_file.replace('.json', '_events.json')

    # ------------------------------------------------------------------
    @property
    def state(self):
        return self.states[self.state_idx]

    def _targets(self):
        if self.state == "LEVEL1": return self.l1_targets
        if self.state == "LEVEL2": return self.l2_targets
        if self.state == "LEVEL3": return self.l3_targets
        return []

    def _current_target(self):
        t = self._targets()
        return t[self.current_trial_idx] if (t and self.current_trial_idx < len(t)) else ""

    def get_ui_text(self):
        tag = f"[{'REPLAY' if self.is_replay else 'LIVE'}·{self.detector.name}]"
        if self.state == "CALIB":
            return [f"Hover calibration.  {tag}", "Hold both hands still..."]
        elif self.state == "PRACTICE":
            return [f"PRACTICE: Free writing.  {tag}",
                    "Hold still ~1 s to clear canvas.",
                    "Press SPACE to start Level 1."]
        elif self.state == "LEVEL1":
            return [
                f"L1 (Chars) — Trial {self.current_trial_idx+1}/{len(self.l1_targets)}  {tag}",
                f"Target: {self._current_target()}",
                "Write char. Hold still ~1 s to auto-confirm, or SPACE.",
            ]
        elif self.state == "LEVEL2":
            return [
                f"L2 (Short words) — Trial {self.current_trial_idx+1}/{len(self.l2_targets)}  {tag}",
                f"Target: {self._current_target()}",
                "Write entire word. Hold still ~1 s to auto-confirm.",
            ]
        elif self.state == "LEVEL3":
            return [
                f"L3 (Long words) — Trial {self.current_trial_idx+1}/{len(self.l3_targets)}  {tag}",
                f"Target: {self._current_target()}",
                "Write entire word. Hold still ~1 s to auto-confirm.",
            ]
        return ["DONE. Press Q to exit."]

    # ------------------------------------------------------------------
    def _flush_stroke(self):
        if self.current_stroke:
            self.current_strokes.append(_trim_tail(self.current_stroke))
            self.current_stroke = []

    def _write_json(self):
        with open(self.out_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, indent=2, ensure_ascii=False)

    def save_current_trial(self):
        self._flush_stroke()
        if self.state in ["LEVEL1", "LEVEL2", "LEVEL3"]:
            self.all_data.append({
                "user_id":   self.user_id,
                "method":    self.detector.name,
                "level":     self.state,
                "timestamp": time.time(),
                "target":    self._current_target(),
                "strokes":   self.current_strokes,
            })
            self._write_json()
        self.current_strokes = []

    def next_trial(self):
        if self.state in ["LEVEL1", "LEVEL2", "LEVEL3"]:
            self.save_current_trial()
            self.current_trial_idx += 1
            if self.current_trial_idx >= len(self._targets()):
                self.state_idx        += 1
                self.current_trial_idx = 0
        elif self.state == "PRACTICE":
            self.state_idx        += 1
            self.current_trial_idx = 0
        self._clear_stroke()

    def restart_current_phase(self):
        if self.state == "CALIB": return
        print(f"[RESET] Restarting phase: {self.state}")
        self.all_data = [d for d in self.all_data if d.get("level") != self.state]
        self._write_json()
        self.current_trial_idx = 0
        # Don't shuffle pre-defined sequences — they must stay aligned with the video
        if not self._fixed_sequences:
            if self.state == "LEVEL1":   random.shuffle(self.l1_targets)
            elif self.state == "LEVEL2": random.shuffle(self.l2_targets)
            elif self.state == "LEVEL3": random.shuffle(self.l3_targets)
        self._clear_stroke()

    def trigger_recalibration(self):
        if self.state == "DONE": return
        self.detector.reset()
        if self.detector.needs_calibration and self.state != "CALIB":
            self.return_state_idx = self.state_idx
            self.state_idx = 0
            print(f"[RECALIB] Will resume at {self.states[self.return_state_idx]}.")
        self._clear_stroke()

    def skip_phase(self):
        if self.state not in ["CALIB", "DONE"]:
            print(f"[SKIP] {self.state}")
            self.state_idx        += 1
            self.current_trial_idx = 0
            self._clear_stroke()

    def _clear_stroke(self):
        self.current_strokes = []
        self.current_stroke  = []

    def record_key_event(self, key_name: str):
        if self.is_replay:
            return
        self.keyboard_events.append({"key": key_name, "t": time.time()})
        self._write_events()

    def _write_events(self):
        with open(self.events_file, 'w', encoding='utf-8') as f:
            json.dump(self.keyboard_events, f, indent=2)

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_id: int):
        is_writing = self.detector.process(frame)
        pos        = self.detector.get_screen_position()
        pos_palm   = self.detector.get_writing_position()

        # ---- CALIB state ------------------------------------------------
        if self.state == "CALIB":
            hr = self.detector.hover_result
            if hr and hr.phase == 'ready':
                if self.return_state_idx is not None:
                    self.state_idx = self.return_state_idx
                    self.return_state_idx = None
                    print(f"[RECALIB] Done. Returning to {self.state}.")
                elif self.initial_target_state:
                    try:
                        self.state_idx = self.states.index(
                            self.initial_target_state.upper())
                        print(f"[START] Jumping to {self.state}.")
                    except ValueError:
                        self.state_idx += 1
                    self.initial_target_state = None
                else:
                    self.state_idx += 1
            return is_writing, pos

        # ---- Still-hold auto-advance ------------------------------------
        if self.detector.consume_still_hold_event():
            self._flush_stroke()
            if self.state == "PRACTICE":
                self.current_strokes = []
                print("[PRACTICE] Still-hold. Canvas cleared.")
            elif self.state in ["LEVEL1", "LEVEL2", "LEVEL3"]:
                if self.current_strokes:
                    print(f"[{self.state}] Still-hold. Auto-advancing.")
                    self.next_trial()

        # ---- Record trajectory (EMA-smoothed) ---------------------------
        if is_writing and pos_palm is not None:
            raw_u, raw_v = float(pos_palm[0]), float(pos_palm[1])
            if self.current_stroke:
                prev = self.current_stroke[-1]
                su = _SMOOTH_ALPHA * raw_u + (1 - _SMOOTH_ALPHA) * prev['u']
                sv = _SMOOTH_ALPHA * raw_v + (1 - _SMOOTH_ALPHA) * prev['v']
            else:
                su, sv = raw_u, raw_v
            self.current_stroke.append({
                "x": pos[0], "y": pos[1],
                "u": su, "v": sv,
                "t": time.time(), "f": frame_id,
            })

        if not is_writing and self.prev_writing:
            self._flush_stroke()
        self.prev_writing = is_writing

        return is_writing, pos


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def draw_hud(frame, texts: list):
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (660, 44 + len(texts) * 30), C['bg'], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    y = 50
    for text in texts:
        cv2.putText(frame, text, (35, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, C['text'], 1, cv2.LINE_AA)
        y += 30


def draw_strokes(frame, strokes, current, color):
    for stroke in strokes:
        for i in range(1, len(stroke)):
            cv2.line(frame,
                     (stroke[i-1]['x'], stroke[i-1]['y']),
                     (stroke[i]['x'],   stroke[i]['y']),
                     color, 3, cv2.LINE_AA)
    for i in range(1, len(current)):
        cv2.line(frame,
                 (current[i-1]['x'], current[i-1]['y']),
                 (current[i]['x'],   current[i]['y']),
                 color, 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_replay_events(events_json: str) -> list:
    if not os.path.exists(events_json):
        return []
    try:
        with open(events_json, 'r', encoding='utf-8') as f:
            return sorted(json.load(f), key=lambda e: e['t'])
    except Exception:
        return []


def _fire_replay_event(key_name: str, collector: Exp3Collector):
    if key_name == 'space' and collector.state not in ["CALIB", "DONE"]:
        collector.next_trial()
        print(f"\n[REPLAY EVT] SPACE → next_trial  (state={collector.state}  trial={collector.current_trial_idx})")
    elif key_name == 'clear':
        collector._clear_stroke()
        print(f"\n[REPLAY EVT] C → _clear_stroke")
    elif key_name == 'restart' and collector.state not in ["CALIB", "DONE"]:
        collector.restart_current_phase()
        print(f"\n[REPLAY EVT] R → restart_phase  ({collector.state})")
    elif key_name == 'skip':
        collector.skip_phase()
        print(f"\n[REPLAY EVT] N → skip_phase")


def main():
    parser = argparse.ArgumentParser(description="Exp 3 Data Collection / Replay")
    parser.add_argument("--user",       default="test_01", help="User ID (e.g. U01)")
    parser.add_argument("--method",     default="own_framework", choices=sorted(REGISTRY),
                        help="Contact detection method")
    parser.add_argument("--start",      default=None,
                        help="[camera] Jump to phase after CALIB (e.g. LEVEL2)")
    # Replay-only arguments
    parser.add_argument("--video",      default=None,
                        help="[replay] Override auto-detected reference video path")
    parser.add_argument("--trials",     default=None,
                        help="[replay] Override auto-detected reference JSON path")
    # Method-specific
    parser.add_argument("--checkpoint", default=_DEFAULT_CKPT,
                        help="[palmpad] Path to best.pt checkpoint")
    parser.add_argument("--headless", action="store_true",
                        help="Disable OpenCV window (auto-enabled when DISPLAY is unset)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Mode selection + fast pre-flight checks (before loading any model) #
    # ------------------------------------------------------------------ #
    is_replay = (args.method != "own_framework")
    headless  = args.headless or (not os.environ.get("DISPLAY", ""))

    if is_replay:
        ref_json = (args.trials
                    or os.path.join(_OWN_DIR, f"exp3_{args.user}.json"))
        if not os.path.exists(ref_json):
            print(f"[ERROR] Reference JSON not found: {ref_json}")
            print("  Collect reference data first:")
            print(f"    python datasets/test.py --user {args.user} --method own_framework")
            sys.exit(1)

        video_path = args.video or _find_latest_video(_VIDEO_DIR, args.user)
        if not video_path:
            print(f"[ERROR] No reference video found in {_VIDEO_DIR}/")
            print("  Specify one explicitly with --video path/to/file.mp4")
            sys.exit(1)

    # Build contact detector (after pre-flight so we fail fast on missing files)
    det_kwargs = {}
    if args.method == "palmpad":
        det_kwargs["checkpoint_path"] = args.checkpoint
    detector = build_detector(args.method, **det_kwargs)

    if is_replay:
        # ── Replay mode: read from existing video, no camera needed ────

        with open(ref_json, "r", encoding="utf-8") as f:
            ref_data = json.load(f)

        l1_seq = [r["target"] for r in ref_data if r.get("level") == "LEVEL1"]
        l2_seq = [r["target"] for r in ref_data if r.get("level") == "LEVEL2"]
        l3_seq = [r["target"] for r in ref_data if r.get("level") == "LEVEL3"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            sys.exit(1)

        collector = Exp3Collector(
            args.user, detector,
            is_replay=True,
            start_level=args.start or "LEVEL1",
            l1_seq=l1_seq, l2_seq=l2_seq, l3_seq=l3_seq,
        )

        _seek_to_level1(cap, video_path, ref_data)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vout_raw     = None   # no re-recording in replay mode

        # Load keyboard events recorded during own_framework collection
        _events_json = ref_json.replace('.json', '_events.json')
        replay_events = _load_replay_events(_events_json)
        _m = re.search(r'_raw_(\d+)\.mp4$', video_path)
        replay_video_start_t = int(_m.group(1)) if _m else None

        print(f"\nExp3  user={args.user}  method={args.method}  [REPLAY MODE]")
        print(f"  Source video → {video_path}")
        print(f"  Trials from  → {ref_json}  "
              f"({len(l1_seq)} L1, {len(l2_seq)} L2, {len(l3_seq)} L3)")
        print(f"  Events       → {_events_json}  ({len(replay_events)} keyboard events)")
        print(f"  Output       → {collector.out_dir}")

    else:
        # ── Camera mode: live input + record raw video ──────────────────
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        collector = Exp3Collector(
            args.user, detector,
            start_level=args.start,
        )

        fourcc   = cv2.VideoWriter.fourcc(*'mp4v')
        rec_fps  = min(max(cap.get(cv2.CAP_PROP_FPS), 1.0), 30.0)
        vout_raw = cv2.VideoWriter(collector.out_video, fourcc, rec_fps, (1280, 720))

        total_frames = 0
        replay_events        = []
        replay_video_start_t = None

        print(f"\nExp3  user={args.user}  method={args.method}  [CAMERA MODE]")
        print(f"  Output → {collector.out_dir}")
        print(f"  Video  → {collector.out_video}")

    print("\nKeys: SPACE=confirm  C=clear  R=restart phase  N=skip  H=recalib  Q=quit\n")

    # ------------------------------------------------------------------ #
    #  Main loop (identical for both modes)                               #
    # ------------------------------------------------------------------ #
    frame_id = 0
    _fps_t   = time.time()
    _fps_cnt = 0
    _fps_val = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or collector.state == "DONE":
            break

        frame_raw = frame.copy()
        frame_id += 1

        _fps_cnt += 1
        _now = time.time()
        if _now - _fps_t >= 0.5:
            _fps_val = _fps_cnt / (_now - _fps_t)
            _fps_cnt = 0
            _fps_t   = _now

        is_writing, pos = collector.process_frame(frame, frame_id)

        # ---- Replay keyboard events at the correct video timestamp ------
        if is_replay and replay_video_start_t is not None and replay_events:
            cur_t = replay_video_start_t + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            while replay_events and replay_events[0]['t'] <= cur_t:
                _fire_replay_event(replay_events.pop(0)['key'], collector)

        if vout_raw is not None:
            vout_raw.write(frame_raw)

        if not headless:
            # HUD
            draw_hud(frame, collector.get_ui_text())

            # Top-right info
            h_f, w_f = frame.shape[:2]
            if is_replay and total_frames > 0:
                info = f"{frame_id}/{total_frames} ({100*frame_id//total_frames}%)"
            else:
                info = f"FPS: {_fps_val:.1f}"
            (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            cv2.putText(frame, info, (w_f - tw - 20, th + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C['accent'], 1, cv2.LINE_AA)

            # Cursor
            if pos != (0, 0):
                color = C['writing'] if is_writing else C['hover']
                cv2.circle(frame, pos, 12 if is_writing else 8,
                           color, -1 if is_writing else 2, cv2.LINE_AA)

            # Strokes
            draw_strokes(frame, collector.current_strokes,
                         collector.current_stroke, C['accent'])

            # Calibration progress bar
            if collector.state == "CALIB":
                hr = collector.detector.hover_result
                if hr:
                    cv2.rectangle(frame, (35, 120),
                                  (35 + int(hr.progress * 300), 135), C['hover'], -1)
                    cv2.rectangle(frame, (35, 120), (335, 135), C['text'], 2)

            cv2.imshow("Exp 3", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord(' '):
                if collector.state not in ["CALIB", "DONE"]:
                    collector.next_trial()
                    collector.record_key_event('space')
            elif key == ord('c'):
                collector._clear_stroke()
                collector.record_key_event('clear')
                print("[CLEAR] Current trial cleared.")
            elif key == ord('r'):
                if collector.state not in ["CALIB", "DONE"]:
                    collector.restart_current_phase()
                    collector.record_key_event('restart')
            elif key == ord('n'):
                collector.skip_phase()
                collector.record_key_event('skip')
            elif key in (ord('h'), ord('H')):
                collector.trigger_recalibration()
        else:
            # Headless: print progress every 300 frames
            if frame_id % 300 == 0:
                pct = f"{100*frame_id//total_frames}%" if total_frames else f"{frame_id}f"
                state_info = collector.state
                if collector.state in ("LEVEL1","LEVEL2","LEVEL3"):
                    state_info += f" trial {collector.current_trial_idx+1}/{len(collector._targets())}"
                print(f"\r  [{pct}] {state_info}  writing={is_writing}", end="", flush=True)

    cap.release()
    if vout_raw is not None:
        vout_raw.release()
    if not headless:
        cv2.destroyAllWindows()
    if headless and is_replay:
        print()  # newline after progress line
    print(f"\nDone. Data → {collector.out_file}")
    if is_replay:
        print(f"  Trials processed: L1={sum(1 for r in collector.all_data if r['level']=='LEVEL1')}"
              f"  L2={sum(1 for r in collector.all_data if r['level']=='LEVEL2')}"
              f"  L3={sum(1 for r in collector.all_data if r['level']=='LEVEL3')}")


if __name__ == '__main__':
    main()
