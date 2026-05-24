"""
Exp 3 Data Collection Script - End-to-End Character & Word Recognition
This script collects writing trajectories (both chars and words) to evaluate
the recognition accuracy of different contact detection methods.

Usage:
    python datasets/test.py --user U01
"""

import sys
import os
import cv2
import time
import json
import random
import numpy as np
import argparse

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hand_track.dual_hand_detector import DualHandDetector

C = {
    'bg': (20, 20, 25), 'text': (210, 210, 215), 'accent': (230, 165, 60),
    'writing': (80, 230, 120), 'hover': (200, 180, 80), 'warn': (60, 60, 240)
}

class Exp3Collector:
    def __init__(self, user_id, out_dir="datasets/Exp3", start_level=None):
        self.user_id = user_id
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_file = os.path.join(self.out_dir, f"exp3_{user_id}.json")
        self.out_video = os.path.join(self.out_dir, f"exp3_{user_id}_raw_{int(time.time())}.mp4")
        
        self.detector = DualHandDetector()
        
        # Workflow phases
        self.states = ["CALIB", "PRACTICE", "LEVEL1", "LEVEL2", "LEVEL3", "DONE"]
        self.state_idx = 0
        self.return_state_idx = None # Used to return to current state after recalibration
        self.initial_target_state = start_level
        
        # Load existing data if resuming
        self.all_data = []
        if os.path.exists(self.out_file):
            try:
                with open(self.out_file, "r", encoding="utf-8") as f:
                    self.all_data = json.load(f)
                print(f"[INFO] Loaded existing data: {len(self.all_data)} records.")
            except Exception as e:
                print(f"[WARN] Failed to load existing JSON: {e}")
                
        # Target Data
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        sampled_chars = random.sample(chars, 10)  # Randomly sample 10 characters
        self.l1_targets = sampled_chars * 2       # Write each 2 times (20 trials total)
        random.shuffle(self.l1_targets)
        
        # Load Level 2 & 3 Target Words
        words_file = os.path.join(os.path.dirname(__file__), "words.txt")
        try:
            with open(words_file, "r", encoding="utf-8") as f:
                words = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except FileNotFoundError:
            print(f"[ERR] {words_file} not found. Using fallback words.")
            words = ["hello", "world", "test", "testing", "longest"]
            
        words_l2 = [w for w in words if 5 <= len(w) <= 6]
        words_l3 = [w for w in words if len(w) >= 7]
        
        # Fallbacks just in case the lists are too small
        if not words_l2: words_l2 = ["hello", "world"]
        if not words_l3: words_l3 = ["testing", "longest"]
            
        self.l2_targets = random.sample(words_l2, min(15, len(words_l2)))
        self.l3_targets = random.sample(words_l3, min(5, len(words_l3)))
        
        self.current_trial_idx = 0
        
        # State tracking (Now unified for L1, L2, L3: just a list of strokes)
        self.current_strokes = [] 
        self.current_stroke = []
        
        self.prev_writing = False

    @property
    def state(self):
        return self.states[self.state_idx]

    def get_ui_text(self):
        if self.state == "CALIB":
            return ["Hover calibration.", "Hold both hands still..."]
        elif self.state == "PRACTICE":
            return ["PRACTICE: Free writing.", "Hold still for 0.5s to clear canvas.", "Press SPACE to start Level 1."]
        elif self.state == "LEVEL1":
            return [
                f"L1 (Chars) - Trial {self.current_trial_idx+1}/{len(self.l1_targets)}",
                f"Target: {self.l1_targets[self.current_trial_idx]}",
                "Write char, hold still for 0.5s to auto-confirm (or SPACE)."
            ]
        elif self.state == "LEVEL2":
            return [
                f"L2 (Med Words, 5-6 chars) - Trial {self.current_trial_idx+1}/{len(self.l2_targets)}",
                f"Target: {self.l2_targets[self.current_trial_idx]}",
                "Write entire word, hold still for 0.5s to auto-confirm."
            ]
        elif self.state == "LEVEL3":
            return [
                f"L3 (Long Words, 7+ chars) - Trial {self.current_trial_idx+1}/{len(self.l3_targets)}",
                f"Target: {self.l3_targets[self.current_trial_idx]}",
                "Write entire word, hold still for 0.5s to auto-confirm."
            ]
        return ["DONE. Press Q to exit."]

    def save_current_trial(self):
        record = {
            "user_id": self.user_id,
            "level": self.state,
            "timestamp": time.time(),
        }
        
        # Flush any remaining strokes
        if self.current_stroke:
            self.current_strokes.append(self.current_stroke)
            self.current_stroke = []
            
        if self.state == "LEVEL1":
            record["target"] = self.l1_targets[self.current_trial_idx]
            record["strokes"] = self.current_strokes
            self.all_data.append(record)
            
        elif self.state == "LEVEL2":
            record["target"] = self.l2_targets[self.current_trial_idx]
            record["strokes"] = self.current_strokes
            self.all_data.append(record)
            
        elif self.state == "LEVEL3":
            record["target"] = self.l3_targets[self.current_trial_idx]
            record["strokes"] = self.current_strokes
            self.all_data.append(record)
            
        self.current_strokes = []
        
        # Write to file incrementally to prevent data loss
        with open(self.out_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, indent=2, ensure_ascii=False)

    def next_trial(self):
        if self.state in ["LEVEL1", "LEVEL2", "LEVEL3"]:
            self.save_current_trial()
            self.current_trial_idx += 1
            targets = self.l1_targets if self.state == "LEVEL1" else \
                      self.l2_targets if self.state == "LEVEL2" else \
                      self.l3_targets
            
            if self.current_trial_idx >= len(targets):
                self.state_idx += 1
                self.current_trial_idx = 0
        elif self.state == "PRACTICE":
            self.state_idx += 1
            self.current_trial_idx = 0
            
        # Ensure traces are fully cleared when moving to the next trial or phase
        self.current_strokes = []
        self.current_stroke = []

    def restart_current_phase(self):
        """Restarts the current phase from trial 0 and deletes saved data for this phase."""
        if self.state == "CALIB": return
        
        print(f"[RESET] Restarting phase: {self.state}")
        # Remove any saved data for the current level
        self.all_data = [d for d in self.all_data if d.get("level") != self.state]
        self.current_trial_idx = 0
        self.current_strokes = []
        self.current_stroke = []
        
        # Shuffle targets again if restarting
        if self.state == "LEVEL1":
            random.shuffle(self.l1_targets)
        elif self.state == "LEVEL2":
            random.shuffle(self.l2_targets)
        elif self.state == "LEVEL3":
            random.shuffle(self.l3_targets)
            
        # Save the cleaned state to file
        with open(self.out_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, indent=2, ensure_ascii=False)
            
    def trigger_recalibration(self):
        """Resets the detector and goes back to the CALIB state, storing the current state to return later."""
        if self.state == "DONE": return
        print("[RECALIB] Triggered manual recalibration.")
        self.detector.reset()
        
        if self.state != "CALIB":
            self.return_state_idx = self.state_idx
            self.state_idx = 0 # Force back to CALIB
        
        self.current_strokes = []
        self.current_stroke = []

    def skip_phase(self):
        """Force skip the current phase."""
        if self.state not in ["CALIB", "DONE"]:
            print(f"[SKIP] Skipping phase {self.state}...")
            self.state_idx += 1
            self.current_trial_idx = 0
            self.current_strokes = []
            self.current_stroke = []

    def process_frame(self, frame, frame_id):
        is_writing = self.detector.process(frame)
        pos = self.detector.get_screen_position()
        pos_palm = self.detector.get_writing_position()
        
        if self.state == "CALIB":
            hr = self.detector.hover_result
            if hr and hr.phase == 'ready':
                # Move to next phase, or return to the phase we were in before recalibration
                if self.return_state_idx is not None:
                    self.state_idx = self.return_state_idx
                    self.return_state_idx = None
                    print(f"[RECALIB] Done. Returning to {self.state}.")
                elif self.initial_target_state:
                    try:
                        target_idx = self.states.index(self.initial_target_state.upper())
                        self.state_idx = target_idx
                        print(f"[START] Skipping directly to {self.state}.")
                    except ValueError:
                        print(f"[ERR] Invalid start level: {self.initial_target_state}. Proceeding normally.")
                        self.state_idx += 1
                    self.initial_target_state = None
                else:
                    self.state_idx += 1 # Auto move to PRACTICE
            return is_writing, pos
            
        # Handle still-hold (0.5s pause to stop writing/clear/advance)
        if self.detector.consume_still_hold_event():
            if self.current_stroke:
                self.current_strokes.append(self.current_stroke)
                self.current_stroke = []
                
            if self.state == "PRACTICE":
                self.current_strokes = []
                print("[PRACTICE] Still-hold triggered. Canvas cleared.")
            elif self.state in ["LEVEL1", "LEVEL2", "LEVEL3"]:
                if self.current_strokes:
                    print(f"[{self.state}] Still-hold triggered. Auto-advancing.")
                    self.next_trial()
                
        # Record trajectories
        if is_writing and pos_palm is not None:
            self.current_stroke.append({
                "x": pos[0], "y": pos[1],
                "u": float(pos_palm[0]), "v": float(pos_palm[1]),
                "t": time.time(),
                "f": frame_id
            })
            
        if not is_writing and self.prev_writing:
            if self.current_stroke:
                self.current_strokes.append(self.current_stroke)
                self.current_stroke = []
                
        self.prev_writing = is_writing
        return is_writing, pos


def draw_hud(frame, texts):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (600, 40 + len(texts)*30 + 20), C['bg'], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    y = 50
    for text in texts:
        cv2.putText(frame, text, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C['text'], 1, cv2.LINE_AA)
        y += 35

def main():
    parser = argparse.ArgumentParser(description="Exp 3 Data Collection")
    parser.add_argument("--user", type=str, default="test_01", help="User ID (e.g., U01)")
    parser.add_argument("--start", type=str, default=None, help="Phase to jump to after CALIB (e.g., LEVEL2)")
    args = parser.parse_args()
    
    collector = Exp3Collector(args.user, start_level=args.start)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Video Recording Setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rec_fps = min(cap.get(cv2.CAP_PROP_FPS), 30)
    if rec_fps <= 0: rec_fps = 30
    vout_raw = cv2.VideoWriter(collector.out_video, fourcc, rec_fps, (1280, 720))
    
    print(f"Starting Exp3 Collection for {args.user}")
    print(f"Video saving to: {collector.out_video}")
    print("Commands:")
    print("  SPACE - Advance / Confirm current trial")
    print("  C     - Clear current trial")
    print("  R     - Restart current phase from beginning")
    print("  N     - Skip current phase completely")
    print("  H     - Recalibrate hover baseline")
    print("  Q     - Quit")
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_raw = frame.copy()
        frame_id += 1
        
        is_writing, pos = collector.process_frame(frame, frame_id)
        
        # Write raw frame to video
        vout_raw.write(frame_raw)
        
        # Draw UI texts
        draw_hud(frame, collector.get_ui_text())
        
        # Visual feedback for writing cursor
        if pos != (0, 0):
            color = C['writing'] if is_writing else C['hover']
            cv2.circle(frame, pos, 12 if is_writing else 8, color, -1 if is_writing else 2, cv2.LINE_AA)
            
        # Draw recorded strokes (previous segments + current active segment)
        for stroke in collector.current_strokes:
            for i in range(1, len(stroke)):
                cv2.line(frame, (stroke[i-1]['x'], stroke[i-1]['y']), (stroke[i]['x'], stroke[i]['y']), C['accent'], 3, cv2.LINE_AA)
        if collector.current_stroke:
            for i in range(1, len(collector.current_stroke)):
                cv2.line(frame, (collector.current_stroke[i-1]['x'], collector.current_stroke[i-1]['y']), 
                                (collector.current_stroke[i]['x'], collector.current_stroke[i]['y']), C['accent'], 3, cv2.LINE_AA)
        
        # Show calibration progress bar
        if collector.state == "CALIB":
            hr = collector.detector.hover_result
            if hr:
                prog = hr.progress
                cv2.rectangle(frame, (35, 120), (35 + int(prog * 300), 135), C['hover'], -1)
                cv2.rectangle(frame, (35, 120), (335, 135), C['text'], 2)

        cv2.imshow("Exp 3 Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            if collector.state not in ["CALIB", "DONE"]:
                collector.next_trial()
        elif key == ord('n'):
            collector.skip_phase()
        elif key == ord('c'):
            # Clear current input (useful if user makes a mistake)
            collector.current_strokes = []
            collector.current_stroke = []
            print("[CLEAR] Cleared current trial data.")
        elif key == ord('r'):
            if collector.state not in ["CALIB", "DONE"]:
                collector.restart_current_phase()
        elif key in (ord('h'), ord('H')):
            collector.trigger_recalibration()
                
    cap.release()
    vout_raw.release()
    cv2.destroyAllWindows()
    print(f"Data collection finished. Saved to {collector.out_file}")

if __name__ == '__main__':
    main()
