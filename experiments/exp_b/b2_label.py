"""
Exp-B2 第二阶段标注脚本：
读取 B1 采集生成的视频与特征 CSV，人工回放并填充 contact_label。
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_B_DIR = PROJECT_ROOT / "experiments" / "data_b"
WINDOW_NAME = "Exp-B2 Label"


def _fmt_hms(total_sec: float) -> str:
    sec = max(0, int(total_sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_text(
    frame,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.68,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _list_feature_files() -> List[Path]:
    if not DATA_B_DIR.exists():
        return []
    files = [
        p for p in DATA_B_DIR.glob("exp_b1_*_features.csv")
        if not p.name.endswith("_labeled.csv")
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _choose_feature_csv() -> Path:
    files = _list_feature_files()
    if not files:
        raise RuntimeError(f"未找到可标注文件：{DATA_B_DIR}/exp_b1_*_features.csv")

    print("\n可标注的特征文件：")
    for i, p in enumerate(files, start=1):
        print(f"  [{i}] {p.name}")

    while True:
        raw = input("选择编号（直接回车默认 1）: ").strip()
        if raw == "":
            return files[0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        except ValueError:
            pass
        print("输入无效，请重新输入。")


def _read_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not fields:
            raise RuntimeError(f"CSV 为空或无表头：{csv_path}")
        if "contact_label" not in fields:
            raise RuntimeError(f"CSV 缺少 contact_label 列：{csv_path}")
        rows = list(reader)
    return fields, rows


def _auto_video_path(features_csv: Path) -> Path:
    # exp_b1_xxx_features.csv -> exp_b1_xxx.mp4
    if not features_csv.name.endswith("_features.csv"):
        return features_csv.with_suffix(".mp4")
    return features_csv.with_name(features_csv.name.replace("_features.csv", ".mp4"))


def _parse_label(raw: str) -> str:
    v = str(raw).strip()
    if v == "":
        return ""
    if v in {"0", "1"}:
        return v
    try:
        return "1" if float(v) >= 0.5 else "0"
    except ValueError:
        return ""


def _label_stats(rows: List[Dict[str, str]]) -> Tuple[int, int, int]:
    n1 = 0
    n0 = 0
    ne = 0
    for r in rows:
        v = _parse_label(r.get("contact_label", ""))
        if v == "1":
            n1 += 1
        elif v == "0":
            n0 += 1
        else:
            ne += 1
    return n1, n0, ne


def _save_rows(csv_path: Path, fields: List[str], rows: List[Dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    DATA_B_DIR.mkdir(parents=True, exist_ok=True)

    print("========== Exp-B2 标注 ==========")
    features_csv = _choose_feature_csv()
    video_path = _auto_video_path(features_csv)
    if not video_path.exists():
        raise RuntimeError(f"找不到对应视频：{video_path}")

    fields, rows = _read_rows(features_csv)
    if not rows:
        raise RuntimeError(f"CSV 无数据行：{features_csv}")

    out_csv = features_csv.with_name(features_csv.stem + "_labeled.csv")
    out_meta = features_csv.with_name(features_csv.stem + "_labeled_meta.json")

    print("\n---")
    print(f"输入特征：{features_csv.relative_to(PROJECT_ROOT)}")
    print(f"输入视频：{video_path.relative_to(PROJECT_ROOT)}")
    print(f"输出标注：{out_csv.relative_to(PROJECT_ROOT)}")
    print("---")
    cmd = input("按 Enter 开始标注，输入 q 退出: ").strip().lower()
    if cmd == "q":
        raise SystemExit("已退出标注。")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3:
        fps = 15.0
    frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_csv = len(rows)
    n_frames = min(frame_count_video if frame_count_video > 0 else frame_count_csv, frame_count_csv)
    if n_frames <= 0:
        raise RuntimeError("视频或 CSV 帧数无效，无法标注。")

    if frame_count_video != frame_count_csv:
        print(f"[提示] 视频帧数={frame_count_video}，CSV行数={frame_count_csv}，将按最小值 {n_frames} 标注。")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx = 0
    prev_read_idx = -2
    current_label = 0
    playing = True
    overwrite_mode = True
    speed_levels = [0.25, 0.5, 1.0, 1.5, 2.0]
    speed_idx = 2  # 1.0x
    dirty = False
    t0 = time.time()

    while True:
        if frame_idx != prev_read_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        prev_read_idx = frame_idx
        if not ok or frame is None:
            break

        if playing:
            old = _parse_label(rows[frame_idx].get("contact_label", ""))
            new = str(current_label)
            if overwrite_mode or old == "":
                if old != new:
                    rows[frame_idx]["contact_label"] = new
                    dirty = True

        n1, n0, ne = _label_stats(rows[:n_frames])
        elapsed = frame_idx / fps
        mode_text = "PLAY" if playing else "PAUSE"
        state_text = "CONTACT(1)" if current_label == 1 else "IDLE(0)"
        speed_text = f"{speed_levels[speed_idx]:.2f}x"
        frame_label = _parse_label(rows[frame_idx].get("contact_label", ""))
        frame_label_text = frame_label if frame_label != "" else "EMPTY"

        show = frame.copy()
        _draw_text(show, f"B2 Label | Frame {frame_idx + 1}/{n_frames} | {elapsed:.2f}s", (20, 40))
        _draw_text(
            show,
            f"Current: {state_text} | Mode: {mode_text} | Speed: {speed_text} | Overwrite: {'ON' if overwrite_mode else 'OFF'}",
            (20, 74),
        )
        _draw_text(show, f"This frame label: {frame_label_text}", (20, 108))
        _draw_text(show, f"Stats 1:{n1}  0:{n0}  empty:{ne}", (20, 142), (180, 230, 120))
        _draw_text(show, "Keys: [space]toggle  [0]/[1]set  [p]play/pause", (20, 176), (180, 180, 180), 0.6, 1)
        _draw_text(show, "      [j]/[l] -/+1  [a]/[d] -/+10  [c]clear  [o]overwrite", (20, 204), (180, 180, 180), 0.6, 1)
        _draw_text(show, "      [z/x] speed  [-/=] speed  [,/.] speed  [s]save  [q]quit", (20, 232), (180, 180, 180), 0.6, 1)
        _draw_text(show, f"File: {features_csv.name}", (20, show.shape[0] - 20), (130, 130, 130), 0.58, 1)
        cv2.imshow(WINDOW_NAME, show)

        play_fps = fps * speed_levels[speed_idx]
        delay = max(1, int(1000.0 / play_fps)) if playing else 0
        key = cv2.waitKey(delay) & 0xFF

        moved = False
        if key == ord("q"):
            break
        if key == ord(" "):
            current_label = 1 - current_label
        elif key == ord("0"):
            current_label = 0
        elif key == ord("1"):
            current_label = 1
        elif key == ord("p"):
            playing = not playing
        elif key == ord("o"):
            overwrite_mode = not overwrite_mode
        elif key in (ord("["), ord("z"), ord("-"), ord(",")):
            speed_idx = max(0, speed_idx - 1)
        elif key in (ord("]"), ord("x"), ord("="), ord(".")):
            speed_idx = min(len(speed_levels) - 1, speed_idx + 1)
        elif key == ord("c"):
            if rows[frame_idx].get("contact_label", "") != "":
                rows[frame_idx]["contact_label"] = ""
                dirty = True
        elif key == ord("j"):
            frame_idx = max(0, frame_idx - 1)
            playing = False
            moved = True
        elif key == ord("l"):
            frame_idx = min(n_frames - 1, frame_idx + 1)
            playing = False
            moved = True
        elif key == ord("a"):
            frame_idx = max(0, frame_idx - 10)
            playing = False
            moved = True
        elif key == ord("d"):
            frame_idx = min(n_frames - 1, frame_idx + 10)
            playing = False
            moved = True
        elif key == ord("s"):
            _save_rows(out_csv, fields, rows)
            dirty = False
            print(f"[checkpoint] 已保存：{out_csv.relative_to(PROJECT_ROOT)}")

        if not moved and playing:
            if frame_idx < n_frames - 1:
                frame_idx += 1
            else:
                playing = False

    _save_rows(out_csv, fields, rows)

    n1, n0, ne = _label_stats(rows[:n_frames])
    meta = {
        "source_features_csv": str(features_csv.relative_to(PROJECT_ROOT)),
        "source_video": str(video_path.relative_to(PROJECT_ROOT)),
        "output_features_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "total_frames": n_frames,
        "label_1_frames": n1,
        "label_0_frames": n0,
        "empty_frames": ne,
        "overwrite_mode_final": overwrite_mode,
        "play_speed_final": speed_levels[speed_idx],
        "play_speed_levels": speed_levels,
        "duration_sec": int(round(time.time() - t0)),
        "labeled_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cap.release()
    cv2.destroyAllWindows()

    print("\n========== 标注完成 ==========")
    print(f"总帧数：{n_frames}")
    print(f"label=1：{n1}帧")
    print(f"label=0：{n0}帧")
    print(f"未标注：{ne}帧")
    print("--------------------------------")
    print(f"标注CSV：{out_csv.relative_to(PROJECT_ROOT)}")
    print(f"标注Meta：{out_meta.relative_to(PROJECT_ROOT)}")
    print("=============================")


if __name__ == "__main__":
    main()
