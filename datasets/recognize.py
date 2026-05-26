import argparse
import base64
import json
import math
import os
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI

# ── API 配置 ──────────────────────────────────────────────────────────────────
_API_KEY  = "sk-6vAtYqBsNcAnnjcLTU4qZyXARJmzL5y5W6cy4a03Tx3NlYe2"
_BASE_URL = "https://api.chatanywhere.tech/v1"
_MODEL    = "gpt-5.4-mini"

# ── 候选词表 ──────────────────────────────────────────────────────────────────
_WORDS_FILE = os.path.join(os.path.dirname(__file__), "words.txt")

def _load_words(path: str) -> tuple[str, str, set, set]:
    with open(path, encoding="utf-8") as f:
        words = [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]
    l2_words = [w for w in words if len(w) < 6]
    l3_words = [w for w in words if len(w) >= 6]
    return ", ".join(l2_words), ", ".join(l3_words), set(l2_words), set(l3_words)

# ── 渲染参数 ──────────────────────────────────────────────────────────────────
_CANVAS = 256
_MARGIN = 24

# ── 渲染 ──────────────────────────────────────────────────────────────────────

_ROTATE_DEG: float = 0.0

def render_strokes(strokes: list) -> np.ndarray:
    """将笔画 (u, v) 渲染为白底黑字图像。"""
    img = np.full((_CANVAS, _CANVAS, 3), 255, dtype=np.uint8)

    all_pts = [pt for s in strokes for pt in s]
    if not all_pts:
        return img

    xs = [p['x'] for p in all_pts]
    ys = [p['y'] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 旋转后内容仍需完整落在画布内：d*(|cosθ|+|sinθ|) ≤ CANVAS
    if _ROTATE_DEG != 0.0:
        rad = math.radians(abs(_ROTATE_DEG))
        safe = _CANVAS / (abs(math.cos(rad)) + abs(math.sin(rad)))
        margin = max(_MARGIN, int((_CANVAS - safe) / 2) + 2)
    else:
        margin = _MARGIN

    draw  = _CANVAS - 2 * margin
    scale = draw / max(x_max - x_min, y_max - y_min, 1e-6)
    x_off = margin + (draw - (x_max - x_min) * scale) / 2
    y_off = margin + (draw - (y_max - y_min) * scale) / 2

    def to_px(x, y):
        return int((x - x_min) * scale + x_off), int((y - y_min) * scale + y_off)

    for stroke in strokes:
        if not stroke:
            continue
        pts = [to_px(p['x'], p['y']) for p in stroke]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(img, pts[0], 4, (60, 60, 220), -1, cv2.LINE_AA)  # 起笔蓝点

    if _ROTATE_DEG != 0.0:
        cx, cy = _CANVAS // 2, _CANVAS // 2
        M = cv2.getRotationMatrix2D((cx, cy), -_ROTATE_DEG, 1.0)
        img = cv2.warpAffine(img, M, (_CANVAS, _CANVAS), borderValue=(255, 255, 255))
    return img


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    return base64.standard_b64encode(buf.tobytes()).decode()


# ── 提示词 ───────────────────────────────────────────────────────────────────

_PROMPT_LETTER = (
    "This image shows a handwritten single uppercase English letter (A-Z). "
    "The stroke is rendered from raw palm coordinates, so the character may appear rotated, slanted, or shaky. "
    "Focus on the topology and stroke count of the shape — ignore orientation. "
    "The red dot is the pen-down starting point. "
    "Reply with ONLY the single uppercase letter, nothing else."
)

_PROMPT_DIGIT = (
    "This image shows a handwritten single digit (0–9). "
    "The stroke is rendered from raw palm coordinates, so the digit may appear rotated, slanted, or shaky. "
    "Focus on the overall shape and loop structure — ignore orientation. "
    "The red dot is the pen-down starting point. "
    "Reply with ONLY the single digit, nothing else."
)

_PROMPT_LEVEL2 = (
    "This image shows a short English word (fewer than 6 letters) written in the air by tracing on a palm with a fingertip. "
    "The writing may appear cursive, slanted, or slightly distorted. "
    "The blue dot marks the pen-down starting point; reading direction is left to right. "
    "Candidate list (choose ONLY from these):\n{word_list}\n\n"
    "Instructions:\n"
    "1. Estimate the number of letters from the stroke count and word length.\n"
    "2. Identify the starting letter shape near the blue dot.\n"
    "3. Select 3 DISTINCT candidates from the list ranked by likelihood.\n"
    "Reply with EXACTLY 3 different words separated by commas, all lowercase. Example: word1, word2, word3"
)

_PROMPT_LEVEL3 = (
    "This image shows a long English word (6 or more letters) written in the air by tracing on a palm with a fingertip. "
    "The writing may appear cursive, slanted, or slightly distorted. "
    "The blue dot marks the pen-down starting point; reading direction is left to right. "
    "Candidate list (choose ONLY from these):\n{word_list}\n\n"
    "Instructions:\n"
    "1. Estimate the number of letters from the overall word length.\n"
    "2. Identify the starting letter shape near the blue dot.\n"
    "3. Select 3 DISTINCT candidates from the list ranked by likelihood.\n"
    "Reply with EXACTLY 3 different words separated by commas, all lowercase. Example: word1, word2, word3"
)

_WORD_LIST_L2: str = ""
_WORD_LIST_L3: str = ""
_WORDS_L2: set = set()
_WORDS_L3: set = set()


def _call(client: OpenAI, prompt: str, img: np.ndarray) -> str:
    resp = client.chat.completions.create(
        model=_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(img)}"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_top3(raw: str, word_set: set) -> list[str]:
    """解析模型返回的逗号分隔候选词，去重、过滤词表外词，不足3个用词表补充。"""
    candidates = [w.strip().lower() for w in raw.replace("\n", ",").split(",") if w.strip()]
    seen, deduped = set(), []
    for w in candidates:
        if w in word_set and w not in seen:
            seen.add(w)
            deduped.append(w)
    # 不足3个时从词表里按字母顺序补不重复的词
    if len(deduped) < 3:
        for w in sorted(word_set):
            if w not in seen:
                seen.add(w)
                deduped.append(w)
            if len(deduped) == 3:
                break
    return deduped[:3]


def recognize(client: OpenAI, img: np.ndarray, level: str, target: str):
    if level == "LEVEL1":
        return _call(client, _PROMPT_DIGIT if target.isdigit() else _PROMPT_LETTER, img)

    prompt = _PROMPT_LEVEL2.format(word_list=_WORD_LIST_L2) if level == "LEVEL2" \
        else _PROMPT_LEVEL3.format(word_list=_WORD_LIST_L3)
    word_set = _WORDS_L2 if level == "LEVEL2" else _WORDS_L3

    raw = _call(client, prompt, img)
    candidates = _parse_top3(raw, word_set)
    if not candidates[0]:  # 完全解析失败则重试一次
        raw = _call(client, prompt, img)
        candidates = _parse_top3(raw, word_set)
    return candidates


# ── 主评估逻辑 ─────────────────────────────────────────────────────────────────

def evaluate(data_path: str, level_filter: Optional[str] = None, save_dir: Optional[str] = None, rotate: float = 0.0):
    global _WORD_LIST_L2, _WORD_LIST_L3, _WORDS_L2, _WORDS_L3, _ROTATE_DEG
    _WORD_LIST_L2, _WORD_LIST_L3, _WORDS_L2, _WORDS_L3 = _load_words(_WORDS_FILE)
    _ROTATE_DEG = rotate
    client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)

    with open(data_path, encoding='utf-8') as f:
        records = json.load(f)

    if save_dir is None:
        user_id  = os.path.splitext(os.path.basename(data_path))[0]
        save_dir = os.path.join(os.path.dirname(data_path), "rendered_imgs", user_id)
    os.makedirs(save_dir, exist_ok=True)

    # ── 阶段一：渲染全部图片 ──────────────────────────────────────────────────
    items = []
    for i, rec in enumerate(records):
        level   = rec.get("level", "")
        target  = rec.get("target", "").strip()
        strokes = rec.get("strokes", [])
        if level not in ("LEVEL1", "LEVEL2", "LEVEL3"):
            continue
        if level_filter and level != level_filter:
            continue
        if not strokes:
            continue
        img   = render_strokes(strokes)
        fname = f"{level}_{i+1:03d}_{target}.png"
        path  = os.path.join(save_dir, fname)
        cv2.imwrite(path, img)
        items.append({"idx": i + 1, "level": level, "target": target, "img_path": path, "img": img})

    print(f"[Phase 1] 渲染完成，共 {len(items)} 张 → {save_dir}")

    # ── 阶段二：逐张送 AI 识别 ────────────────────────────────────────────────
    print(f"[Phase 2] 开始识别 (model={_MODEL}) ...")
    buckets: dict[str, list] = {"LEVEL1": [], "LEVEL2": [], "LEVEL3": []}

    for item in items:
        level, target, img = item["level"], item["target"], item["img"]
        result  = recognize(client, img, level, target)

        if isinstance(result, list):
            pred      = result[0]
            top3      = result
            top1_ok   = pred.strip().upper() == target.upper()
            top3_ok   = any(r.strip().upper() == target.upper() for r in top3)
            mark      = "✓" if top1_ok else ("△" if top3_ok else "✗")
            pred_str  = ", ".join(top3)
        else:
            pred = result
            top3 = [pred]
            top1_ok = top3_ok = pred.strip().upper() == target.upper()
            mark     = "✓" if top1_ok else "✗"
            pred_str = pred

        buckets[level].append({
            "target": target, "pred": pred, "top3": top3,
            "correct": top1_ok, "top3_correct": top3_ok
        })
        print(f"[{level}] #{item['idx']:3d}  target={target:<14s}  pred={pred_str:<36s}  {mark}")

    def _char_acc(target: str, pred: str) -> tuple[int, int]:
        t, p = target.lower(), pred.lower()
        matched = sum(a == b for a, b in zip(t, p))
        return matched, len(t)

    print("\n── Word Accuracy ─────────────────────────────")
    total_n = total_ok = total_ok3 = 0
    for lvl, lst in buckets.items():
        if not lst:
            continue
        n   = len(lst)
        ok  = sum(r['correct'] for r in lst)
        ok3 = sum(r['top3_correct'] for r in lst)
        total_n   += n
        total_ok  += ok
        total_ok3 += ok3
        print(f"  {lvl:<8s}  Top-1: {ok}/{n} ({ok/n*100:.1f}%)   Top-3: {ok3}/{n} ({ok3/n*100:.1f}%)")
    if total_n:
        print(f"  {'ALL':<8s}  Top-1: {total_ok}/{total_n} ({total_ok/total_n*100:.1f}%)   Top-3: {total_ok3}/{total_n} ({total_ok3/total_n*100:.1f}%)")

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
            top3_str = ", ".join(r['top3']) if len(r['top3']) > 1 else r['pred']
            hit = "△(top3)" if r['top3_correct'] else "✗"
            print(f"  target={r['target']:<14s}  pred={top3_str:<36s}  {hit}")

    # ── 保存结果（合并已有文件，避免指定 --level 时覆盖其他 level）────────────
    _exp3_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exp3")
    os.makedirs(_exp3_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(data_path))[0]
    result_path = os.path.join(_exp3_dir, basename + "_results.json")
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            existing = json.load(f)
        summary = existing.get("summary", {})
    else:
        summary = {}

    for lvl, lst in buckets.items():
        if not lst:
            continue
        word_ok = sum(r['correct'] for r in lst)
        ch_ok = ch_n = 0
        for r in lst:
            ok, n = _char_acc(r['target'], r['pred'])
            ch_ok += ok; ch_n += n
        top3_ok = sum(r['top3_correct'] for r in lst)
        summary[lvl] = {
            "word_acc":      round(word_ok / len(lst), 4),
            "top3_word_acc": round(top3_ok / len(lst), 4),
            "char_acc":      round(ch_ok / ch_n, 4) if ch_n else 0,
            "records": lst,
        }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"model": _MODEL, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Air-writing recognition via Vision LLM")
    parser.add_argument("data",          help="Path to exp3_userXX.json")
    parser.add_argument("--level",       default=None, choices=["LEVEL1", "LEVEL2", "LEVEL3"])
    parser.add_argument("--save-images", metavar="DIR", default=None)
    parser.add_argument("--rotate",      type=float, default=0,
                        help="Clockwise rotation in degrees (default: 0)")
    args = parser.parse_args()
    evaluate(args.data, level_filter=args.level, save_dir=args.save_images, rotate=args.rotate)


if __name__ == "__main__":
    main()
