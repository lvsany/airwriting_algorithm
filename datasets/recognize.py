"""
Air-writing recognition evaluation via Claude Vision API.

Pipeline:
  stroke data (u, v) → render PNG → Claude Vision → compare with target → accuracy

Usage:
    python datasets/recognize.py datasets/datasets/Exp3/exp3_user_01.json
    python datasets/recognize.py datasets/datasets/Exp3/exp3_user_01.json --level LEVEL2
    python datasets/recognize.py datasets/datasets/Exp3/exp3_user_01.json --save-images ./imgs
"""

import argparse
import base64
import json
import os

import cv2
import numpy as np
from openai import OpenAI

# ── API 配置 ──────────────────────────────────────────────────────────────────
_API_KEY  = "sk-6vAtYqBsNcAnnjcLTU4qZyXARJmzL5y5W6cy4a03Tx3NlYe2"
_BASE_URL = "https://api.chatanywhere.tech/v1"
_MODEL    = "gpt-4.1"

# ── 候选词表 ──────────────────────────────────────────────────────────────────
_WORDS_FILE = os.path.join(os.path.dirname(__file__), "words.txt")

def _load_words(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        words = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return ", ".join(words)

# ── 渲染参数 ──────────────────────────────────────────────────────────────────
_CANVAS = 256
_MARGIN = 24

# ── 渲染 ──────────────────────────────────────────────────────────────────────

def _deskew(strokes: list) -> list:
    """Rotate strokes so the principal writing direction is horizontal."""
    all_pts = [pt for s in strokes for pt in s]
    if len(all_pts) < 2:
        return strokes
    us = np.array([p['u'] for p in all_pts])
    vs = np.array([p['v'] for p in all_pts])
    mean_u, mean_v = float(np.mean(us)), float(np.mean(vs))
    pts = np.column_stack([us - mean_u, vs - mean_v])
    _, eigvecs = np.linalg.eigh(np.cov(pts.T))
    principal = eigvecs[:, 1]  # eigenvector for largest eigenvalue
    angle = -np.arctan2(principal[1], principal[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    result = []
    for s in strokes:
        rot = []
        for pt in s:
            du, dv = pt['u'] - mean_u, pt['v'] - mean_v
            new_pt = dict(pt)
            new_pt['u'] = cos_a * du - sin_a * dv + mean_u
            new_pt['v'] = sin_a * du + cos_a * dv + mean_v
            rot.append(new_pt)
        result.append(rot)
    return result


def render_strokes(strokes: list) -> np.ndarray:
    """将笔画 (u, v) 渲染为白底黑字图像。"""
    img = np.full((_CANVAS, _CANVAS, 3), 255, dtype=np.uint8)

    all_pts = [pt for s in strokes for pt in s]
    if not all_pts:
        return img

    us = [p['u'] for p in all_pts]
    vs = [p['v'] for p in all_pts]
    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)
    draw  = _CANVAS - 2 * _MARGIN
    scale = draw / max(u_max - u_min, v_max - v_min, 1e-6)
    u_off = _MARGIN + (draw - (u_max - u_min) * scale) / 2
    v_off = _MARGIN + (draw - (v_max - v_min) * scale) / 2

    def to_px(u, v):
        px = _CANVAS - 1 - int((u - u_min) * scale + u_off)
        py = _CANVAS - 1 - int((v - v_min) * scale + v_off)
        return px, py

    for stroke in strokes:
        if not stroke:
            continue
        pts = [to_px(p['u'], p['v']) for p in stroke]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(img, pts[0], 4, (60, 60, 220), -1, cv2.LINE_AA)  # 起笔蓝点

    cx, cy = _CANVAS // 2, _CANVAS // 2
    M = cv2.getRotationMatrix2D((cx, cy), -45, 1.0)
    img = cv2.warpAffine(img, M, (_CANVAS, _CANVAS), borderValue=(255, 255, 255))
    return img


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    return base64.standard_b64encode(buf.tobytes()).decode()


# ── Claude Vision 识别 ────────────────────────────────────────────────────────

_PROMPT_CHAR = (
    "This image shows a single character traced in the air with a fingertip on someone's palm. "
    "It is exactly one of: an uppercase letter (A–Z) or a digit (0–9). "
    "The writing is imprecise and may look slanted or shaky — judge by the overall stroke structure, not perfection. "
    "The red dot marks the very start of the stroke. "
    "Common confusions to watch for: 1 vs I/L, 7 vs Z, B vs 8, W vs M/N, Y vs V/K, L vs J. "
    "Reply with ONLY that single character in uppercase, nothing else."
)

_PROMPT_WORD_TMPL = (
    "This image shows an English word written in the air with a fingertip on someone's palm. "
    "The handwriting resembles messy cursive or connected print — strokes may be shaky, slanted, or run together. "
    "The red dot marks where each pen-down stroke begins; letters are written left to right. "
    "You MUST pick exactly one word from the following candidate list:\n{word_list}\n"
    "Choose the candidate that best matches the handwriting. "
    "Reply with ONLY that single word in lowercase, nothing else."
)

_WORD_LIST: str = ""


def recognize(client: OpenAI, img: np.ndarray, level: str) -> str:
    if level == "LEVEL1":
        prompt = _PROMPT_CHAR
    else:
        prompt = _PROMPT_WORD_TMPL.format(word_list=_WORD_LIST)
    resp = client.chat.completions.create(
        model=_MODEL,
        max_tokens=32,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_to_b64(img)}"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


# ── 主评估逻辑 ─────────────────────────────────────────────────────────────────

def evaluate(data_path: str, level_filter: str = None, save_dir: str = None):
    global _WORD_LIST
    _WORD_LIST = _load_words(_WORDS_FILE)
    client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)

    with open(data_path, encoding='utf-8') as f:
        records = json.load(f)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    buckets: dict[str, list] = {"LEVEL1": [], "LEVEL2": [], "LEVEL3": []}

    for i, rec in enumerate(records):
        level   = rec.get("level", "")
        target  = rec.get("target", "").strip()
        strokes = rec.get("strokes", [])

        if level not in buckets:
            continue
        if level_filter and level != level_filter:
            continue
        if not strokes:
            continue

        img     = render_strokes(strokes)
        pred    = recognize(client, img, level)
        correct = pred.strip().upper() == target.upper()
        buckets[level].append({"target": target, "pred": pred, "correct": correct})

        mark = "✓" if correct else "✗"
        print(f"[{level}] #{i+1:3d}  target={target:<14s}  pred={pred:<16s}  {mark}")

        if save_dir:
            fname = f"{level}_{i+1:03d}_{target}_pred{pred}.png"
            cv2.imwrite(os.path.join(save_dir, fname), img)

    def _char_acc(target: str, pred: str) -> tuple[int, int]:
        """返回 (匹配字符数, 目标字符数)，按最短对齐逐字符比较。"""
        t, p = target.lower(), pred.lower()
        matched = sum(a == b for a, b in zip(t, p))
        return matched, len(t)

    print("\n── Word Accuracy ─────────────────────────────")
    total_n = total_ok = 0
    for lvl, lst in buckets.items():
        if not lst:
            continue
        n  = len(lst)
        ok = sum(r['correct'] for r in lst)
        total_n  += n
        total_ok += ok
        print(f"  {lvl:<8s}  {ok}/{n}  ({ok/n*100:.1f}%)")
    if total_n:
        print(f"  {'ALL':<8s}  {total_ok}/{total_n}  ({total_ok/total_n*100:.1f}%)")

    print("\n── Char Accuracy ─────────────────────────────")
    total_ch = total_ch_ok = 0
    for lvl, lst in buckets.items():
        if not lst:
            continue
        ch_ok = ch_n = 0
        for r in lst:
            ok, n = _char_acc(r['target'], r['pred'])
            ch_ok += ok
            ch_n  += n
        total_ch_ok += ch_ok
        total_ch    += ch_n
        print(f"  {lvl:<8s}  {ch_ok}/{ch_n}  ({ch_ok/ch_n*100:.1f}%)")
    if total_ch:
        print(f"  {'ALL':<8s}  {total_ch_ok}/{total_ch}  ({total_ch_ok/total_ch*100:.1f}%)")

    errors = [r for lst in buckets.values() for r in lst if not r['correct']]
    if errors:
        print(f"\n── Errors ({len(errors)}) ──────────────────────────")
        for r in errors:
            print(f"  target={r['target']:<14s}  pred={r['pred']}")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    result_path = os.path.splitext(data_path)[0] + "_results.json"
    summary = {}
    for lvl, lst in buckets.items():
        if not lst:
            continue
        word_ok = sum(r['correct'] for r in lst)
        ch_ok = ch_n = 0
        for r in lst:
            ok, n = _char_acc(r['target'], r['pred'])
            ch_ok += ok; ch_n += n
        summary[lvl] = {
            "word_acc": round(word_ok / len(lst), 4),
            "char_acc": round(ch_ok / ch_n, 4) if ch_n else 0,
            "records": lst,
        }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"model": _MODEL, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Air-writing recognition via Claude Vision")
    parser.add_argument("data",          help="Path to exp3_userXX.json")
    parser.add_argument("--level",       default=None, choices=["LEVEL1", "LEVEL2", "LEVEL3"])
    parser.add_argument("--save-images", metavar="DIR", default=None,
                        help="Save rendered images to DIR for inspection")
    args = parser.parse_args()
    evaluate(args.data, level_filter=args.level, save_dir=args.save_images)


if __name__ == "__main__":
    main()
