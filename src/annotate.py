"""
回放标注工具

用法:
    python annotate.py                          # 自动加载最新 session
    python annotate.py results/test/session_XX.csv

操作:
    空格      — 切换 CONTACT / IDLE 状态（持续到下次切换）
    P         — 暂停 / 继续
    + / =     — 加速
    - / _     — 减速
    W         — 保存标注到 CSV
    Q / ESC   — 退出
"""

import csv, cv2, numpy as np, os, shutil, sys


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows


def save_csv(path, rows, manual_states):
    fieldnames = list(rows[0].keys())
    if 'manual_state' not in fieldnames:
        fieldnames.append('manual_state')
    backup = path.replace('.csv', '_orig.csv')
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            r = dict(row)
            r['manual_state'] = manual_states[i] if i < len(manual_states) else 'idle'
            w.writerow(r)
    print(f"[SAVED] {path}")


def main(session_path=None):
    # ── 找文件 ──────────────────────────────────────────────────────────────
    if session_path is None:
        base_dir = os.path.join(os.path.dirname(__file__), 'results', 'test')
        csvs = sorted(f for f in os.listdir(base_dir)
                      if f.endswith('.csv') and '_orig' not in f)
        session_path = os.path.join(base_dir, csvs[-1])
        print(f"[AUTO] {session_path}")

    stem = session_path.rsplit('.', 1)[0]
    csv_path   = stem + '.csv'
    video_path = stem + '.mp4'

    rows = load_csv(csv_path)
    n    = len(rows)
    algo_states   = [r.get('final_state', 'idle') for r in rows]
    manual_states = [r.get('manual_state') or 'idle' for r in rows]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── 播放状态 ─────────────────────────────────────────────────────────────
    cur_state = 'idle'
    frame_idx = 0
    modified  = False
    speed     = 1.0
    paused    = False

    print(f"[INFO] {n} frames  |  空格=切换状态  P=暂停  +/-=速度  W=保存  Q=退出")

    last_frame = None   # 缓存暂停时的画面

    while True:
        # ── 读帧（顺序读，只在暂停时复用上一帧）────────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret or frame_idx >= n:
                break
            last_frame = frame
            manual_states[frame_idx] = cur_state
        else:
            frame = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), np.uint8)

        # ── 绘制 ─────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        algo_s = algo_states[frame_idx]

        ov = frame.copy()
        cv2.rectangle(ov, (0, h - 52), (w, h), (20, 20, 25), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        alg_col = (80, 230, 120) if algo_s == 'contact' else (80, 80, 95)
        cv2.putText(frame, f"ALG: {algo_s.upper()}", (12, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, alg_col, 1, cv2.LINE_AA)

        man_col = (80, 230, 120) if cur_state == 'contact' else (80, 80, 95)
        cv2.putText(frame, f"MAN: {cur_state.upper()}", (12, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, man_col, 2, cv2.LINE_AA)

        pause_str = "PAUSED  " if paused else f"x{speed:.2g}  "
        cv2.putText(frame, pause_str + f"{frame_idx + 1}/{n}", (w - 180, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 130), 1, cv2.LINE_AA)

        cv2.putText(frame, "SPACE=toggle  P=pause  +/-=speed  W=save  Q=quit",
                    (w // 2 - 185, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 110), 1, cv2.LINE_AA)

        cv2.imshow('Annotate', frame)

        # ── waitKey 直接承担帧率控制，无需 sleep ─────────────────────────────
        wait_ms = max(1, int(1000 / (fps * speed))) if not paused else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' '):
            cur_state = 'idle' if cur_state == 'contact' else 'contact'
            modified = True
        elif key in (ord('+'), ord('=')):
            speed = min(speed * 1.5, 16.0)
        elif key in (ord('-'), ord('_')):
            speed = max(speed / 1.5, 0.1)
        elif key == ord('w'):
            save_csv(csv_path, rows, manual_states)
            modified = False

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if modified:
        ans = input("有未保存的标注，现在保存？(y/n): ").strip().lower()
        if ans == 'y':
            save_csv(csv_path, rows, manual_states)

    contact_n = sum(1 for s in manual_states if s == 'contact')
    algo_n    = sum(1 for s in algo_states   if s == 'contact')
    agree     = sum(1 for a, m in zip(algo_states, manual_states) if a == m)
    print(f"[STATS] Manual contact={contact_n}  Algo contact={algo_n}"
          f"  Agreement={agree / max(n, 1) * 100:.1f}%")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
