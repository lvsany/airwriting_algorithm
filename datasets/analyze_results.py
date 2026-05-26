"""
Aggregate recognition results across all participants.

Metrics:
  LEVEL1 - Letter accuracy (A-Z) and Digit accuracy (0-9)
  LEVEL2 - Word accuracy (WA) and Character Error Rate (CER)
  LEVEL3 - Word accuracy (WA) and Character Error Rate (CER)

CER = edit_distance(pred, target) / len(target)

Usage:
    python datasets/analyze_results.py
    python datasets/analyze_results.py --dir datasets/datasets/Exp3
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, Optional, Union


# ── 编辑距离 ──────────────────────────────────────────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    s1, s2 = s1.lower(), s2.lower()
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def cer(target: str, pred: str) -> float:
    if not target:
        return 0.0
    return edit_distance(target, pred) / len(target)


# ── 统计单个结果文件 ───────────────────────────────────────────────────────────

def analyze_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    result: dict[str, Any] = {"path": path}

    # ── LEVEL1：字母 vs 数字分开统计 ──────────────────────────────────────────
    l1 = summary.get("LEVEL1", {})
    letters = {"ok": 0, "n": 0}
    digits  = {"ok": 0, "n": 0}
    for r in l1.get("records", []):
        t = r["target"].strip().upper()
        bucket = digits if t.isdigit() else letters
        bucket["n"] += 1
        if r["correct"]:
            bucket["ok"] += 1
    result["LEVEL1"] = {
        "letter_acc": letters["ok"] / letters["n"] if letters["n"] else None,
        "letter_n":   letters["n"],
        "digit_acc":  digits["ok"]  / digits["n"]  if digits["n"]  else None,
        "digit_n":    digits["n"],
    }

    # ── LEVEL2 / LEVEL3：WA + CER + Top3-WA + Top3-CER ──────────────────────
    for lvl in ("LEVEL2", "LEVEL3"):
        lx = summary.get(lvl, {})
        records = lx.get("records", [])
        if not records:
            result[lvl] = None
            continue
        word_ok   = sum(1 for r in records if r["correct"])
        top3_ok   = sum(1 for r in records if r.get("top3_correct", r["correct"]))
        total_ed  = sum(edit_distance(r["target"], r["pred"]) for r in records)
        total_ed3 = sum(
            min(edit_distance(r["target"], c) for c in r.get("top3", [r["pred"]]))
            for r in records
        )
        total_ref = sum(len(r["target"]) for r in records)
        result[lvl] = {
            "wa":       word_ok  / len(records),
            "top3_wa":  top3_ok  / len(records),
            "cer":      total_ed  / total_ref if total_ref else 0.0,
            "top3_cer": total_ed3 / total_ref if total_ref else 0.0,
            "n":        len(records),
        }

    return result


# ── 汇总所有用户 ──────────────────────────────────────────────────────────────

def aggregate(file_results: list) -> dict:
    agg = {
        "LEVEL1": {"letter_ok": 0, "letter_n": 0, "digit_ok": 0, "digit_n": 0},
        "LEVEL2": {"word_ok": 0, "total_ed": 0, "total_ref": 0, "n": 0},
        "LEVEL3": {"word_ok": 0, "total_ed": 0, "total_ref": 0, "n": 0},
    }
    for res in file_results:
        l1 = res["LEVEL1"]
        if l1["letter_n"]:
            agg["LEVEL1"]["letter_ok"] += round(l1["letter_acc"] * l1["letter_n"])
            agg["LEVEL1"]["letter_n"]  += l1["letter_n"]
        if l1["digit_n"]:
            agg["LEVEL1"]["digit_ok"] += round(l1["digit_acc"] * l1["digit_n"])
            agg["LEVEL1"]["digit_n"]  += l1["digit_n"]

        for lvl in ("LEVEL2", "LEVEL3"):
            lx = res[lvl]
            if lx is None:
                continue
            agg[lvl]["word_ok"]   += round(lx["wa"] * lx["n"])
            agg[lvl]["total_ed"]  += round(lx["cer"] * lx["n"] * 5)  # approx; recalculate below
            agg[lvl]["n"]         += lx["n"]

    # WA from raw counts
    out = {}
    a1 = agg["LEVEL1"]
    out["LEVEL1"] = {
        "letter_acc": a1["letter_ok"] / a1["letter_n"] if a1["letter_n"] else None,
        "letter_n":   a1["letter_n"],
        "digit_acc":  a1["digit_ok"]  / a1["digit_n"]  if a1["digit_n"]  else None,
        "digit_n":    a1["digit_n"],
    }
    for lvl in ("LEVEL2", "LEVEL3"):
        a = agg[lvl]
        if a["n"] == 0:
            out[lvl] = None
        else:
            out[lvl] = {
                "wa":  a["word_ok"] / a["n"],
                "n":   a["n"],
            }
    return out


# ── 打印 ──────────────────────────────────────────────────────────────────────

def fmt_pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "  N/A  "


def print_user_table(file_results: list):
    print("\n" + "=" * 100)
    print(f"{'User':<20} {'L1-Letter':>10} {'L1-Digit':>10} "
          f"{'L2-WA':>7} {'L2-T3WA':>8} {'L2-CER':>7} {'L2-T3CER':>9} "
          f"{'L3-WA':>7} {'L3-T3WA':>8} {'L3-CER':>7} {'L3-T3CER':>9}")
    print("-" * 100)
    for res in file_results:
        user = os.path.basename(res["path"]).replace("_results.json", "")
        l1   = res["LEVEL1"]
        l2   = res["LEVEL2"]
        l3   = res["LEVEL3"]
        print(
            f"{user:<20} "
            f"{fmt_pct(l1['letter_acc']):>10} "
            f"{fmt_pct(l1['digit_acc']):>10} "
            f"{fmt_pct(l2['wa']       if l2 else None):>7} "
            f"{fmt_pct(l2['top3_wa']  if l2 else None):>8} "
            f"{fmt_pct(l2['cer']      if l2 else None):>7} "
            f"{fmt_pct(l2['top3_cer'] if l2 else None):>9} "
            f"{fmt_pct(l3['wa']       if l3 else None):>7} "
            f"{fmt_pct(l3['top3_wa']  if l3 else None):>8} "
            f"{fmt_pct(l3['cer']      if l3 else None):>7} "
            f"{fmt_pct(l3['top3_cer'] if l3 else None):>9} "
        )


def print_overall(file_results: list):
    # Recompute CER properly from raw records
    totals = {
        "LEVEL1": {"letter_ok": 0, "letter_n": 0, "digit_ok": 0, "digit_n": 0},
        "LEVEL2": {"word_ok": 0, "top3_ok": 0, "ed": 0, "ed3": 0, "ref": 0, "n": 0},
        "LEVEL3": {"word_ok": 0, "top3_ok": 0, "ed": 0, "ed3": 0, "ref": 0, "n": 0},
    }

    for res in file_results:
        path = res["path"]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})

        for r in summary.get("LEVEL1", {}).get("records", []):
            t = r["target"].strip().upper()
            if t.isdigit():
                totals["LEVEL1"]["digit_n"] += 1
                totals["LEVEL1"]["digit_ok"] += int(r["correct"])
            else:
                totals["LEVEL1"]["letter_n"] += 1
                totals["LEVEL1"]["letter_ok"] += int(r["correct"])

        for lvl in ("LEVEL2", "LEVEL3"):
            for r in summary.get(lvl, {}).get("records", []):
                totals[lvl]["n"]       += 1
                totals[lvl]["word_ok"] += int(r["correct"])
                totals[lvl]["top3_ok"] += int(r.get("top3_correct", r["correct"]))
                totals[lvl]["ed"]      += edit_distance(r["target"], r["pred"])
                totals[lvl]["ed3"]     += min(edit_distance(r["target"], c) for c in r.get("top3", [r["pred"]]))
                totals[lvl]["ref"]     += len(r["target"])

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    t1 = totals["LEVEL1"]
    letter_acc = t1["letter_ok"] / t1["letter_n"] if t1["letter_n"] else None
    digit_acc  = t1["digit_ok"]  / t1["digit_n"]  if t1["digit_n"]  else None
    print(f"\nLEVEL1  (n_letters={t1['letter_n']}, n_digits={t1['digit_n']})")
    print(f"  Letter Accuracy : {fmt_pct(letter_acc)}")
    print(f"  Digit  Accuracy : {fmt_pct(digit_acc)}")

    for lvl in ("LEVEL2", "LEVEL3"):
        t = totals[lvl]
        if t["n"] == 0:
            print(f"\n{lvl}  no data")
            continue
        wa      = t["word_ok"]  / t["n"]
        top3_wa = t["top3_ok"]  / t["n"]
        cer_val  = t["ed"]  / t["ref"] if t["ref"] else 0.0
        cer3_val = t["ed3"] / t["ref"] if t["ref"] else 0.0
        print(f"\n{lvl}  (n={t['n']})")
        print(f"  Word Accuracy Top-1 (WA)       : {fmt_pct(wa)}")
        print(f"  Word Accuracy Top-3 (T3WA)     : {fmt_pct(top3_wa)}")
        print(f"  Char Error Rate Top-1 (CER)    : {fmt_pct(cer_val)}")
        print(f"  Char Error Rate Best-3 (T3CER) : {fmt_pct(cer3_val)}")

    print()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="Exp3",
                        help="Directory containing *_results.json files")
    args = parser.parse_args()

    pattern = os.path.join(args.dir, "*_results.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No *_results.json files found in {args.dir}")
        return

    print(f"Found {len(paths)} result file(s):")
    for p in paths:
        print(f"  {p}")

    file_results = [analyze_file(p) for p in paths]
    print_user_table(file_results)
    print_overall(file_results)
    save_results(args.dir, paths, file_results)


def save_results(out_dir: str, paths: list, file_results: list):
    # ── 重新从原始记录精确计算汇总 ────────────────────────────────────────────
    totals = {
        "LEVEL1": {"letter_ok": 0, "letter_n": 0, "digit_ok": 0, "digit_n": 0},
        "LEVEL2": {"word_ok": 0, "top3_ok": 0, "ed": 0, "ed3": 0, "ref": 0, "n": 0},
        "LEVEL3": {"word_ok": 0, "top3_ok": 0, "ed": 0, "ed3": 0, "ref": 0, "n": 0},
    }
    user_rows = []

    for res, path in zip(file_results, paths):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        user = os.path.basename(path).replace("_results.json", "")

        row = {"用户": user}

        for r in summary.get("LEVEL1", {}).get("records", []):
            t = r["target"].strip().upper()
            if t.isdigit():
                totals["LEVEL1"]["digit_n"]  += 1
                totals["LEVEL1"]["digit_ok"] += int(r["correct"])
            else:
                totals["LEVEL1"]["letter_n"]  += 1
                totals["LEVEL1"]["letter_ok"] += int(r["correct"])

        l1 = res["LEVEL1"]
        row["L1字母准确率"] = round(l1["letter_acc"], 4) if l1["letter_acc"] is not None else None
        row["L1字母数"]     = l1["letter_n"]
        row["L1数字准确率"] = round(l1["digit_acc"], 4)  if l1["digit_acc"]  is not None else None
        row["L1数字数"]     = l1["digit_n"]

        lv_cn = {"LEVEL2": "L2", "LEVEL3": "L3"}
        for lvl in ("LEVEL2", "LEVEL3"):
            px = lv_cn[lvl]
            recs = summary.get(lvl, {}).get("records", [])
            for r in recs:
                totals[lvl]["n"]       += 1
                totals[lvl]["word_ok"] += int(r["correct"])
                totals[lvl]["top3_ok"] += int(r.get("top3_correct", r["correct"]))
                totals[lvl]["ed"]      += edit_distance(r["target"], r["pred"])
                totals[lvl]["ed3"]     += min(edit_distance(r["target"], c) for c in r.get("top3", [r["pred"]]))
                totals[lvl]["ref"]     += len(r["target"])
            lx = res[lvl]
            if lx:
                ed  = sum(edit_distance(r["target"], r["pred"]) for r in recs)
                ed3 = sum(min(edit_distance(r["target"], c) for c in r.get("top3", [r["pred"]])) for r in recs)
                ref = sum(len(r["target"]) for r in recs)
                row[f"{px}词准确率"]      = round(lx["wa"], 4)
                row[f"{px}Top3词准确率"]  = round(lx["top3_wa"], 4)
                row[f"{px}字符错误率"]    = round(ed  / ref, 4) if ref else None
                row[f"{px}Top3字符错误率"] = round(ed3 / ref, 4) if ref else None
                row[f"{px}样本数"]        = lx["n"]
            else:
                row[f"{px}词准确率"] = row[f"{px}Top3词准确率"] = \
                row[f"{px}字符错误率"] = row[f"{px}Top3字符错误率"] = row[f"{px}样本数"] = None
        user_rows.append(row)

    t1 = totals["LEVEL1"]
    level1_overall: Dict[str, Optional[Union[float, int]]] = {
        "字母准确率": round(t1["letter_ok"] / t1["letter_n"], 4) if t1["letter_n"] else None,
        "字母数":     t1["letter_n"],
        "数字准确率": round(t1["digit_ok"]  / t1["digit_n"],  4) if t1["digit_n"]  else None,
        "数字数":     t1["digit_n"],
    }
    overall: Dict[str, Optional[Dict[str, Optional[Union[float, int]]]]] = {
        "LEVEL1": level1_overall
    }
    for lvl in ("LEVEL2", "LEVEL3"):
        t = totals[lvl]
        overall[lvl] = {
            "词准确率":       round(t["word_ok"]  / t["n"],   4) if t["n"]   else None,
            "Top3词准确率":   round(t["top3_ok"]  / t["n"],   4) if t["n"]   else None,
            "字符错误率":     round(t["ed"]  / t["ref"], 4)       if t["ref"] else None,
            "Top3字符错误率": round(t["ed3"] / t["ref"], 4)       if t["ref"] else None,
            "样本数":         t["n"],
        } if t["n"] else None

    # ── 保存 JSON ──────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "analysis_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"各用户": user_rows, "总体": overall}, f, ensure_ascii=False, indent=2)
    print(f"JSON 已保存 → {json_path}")

    # ── 保存 CSV ───────────────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(out_dir, "analysis_summary.csv")
    fieldnames = [
        "用户",
        "L1字母准确率", "L1字母数",
        "L1数字准确率", "L1数字数",
        "L2词准确率", "L2Top3词准确率", "L2字符错误率", "L2Top3字符错误率", "L2样本数",
        "L3词准确率", "L3Top3词准确率", "L3字符错误率", "L3Top3字符错误率", "L3样本数",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(user_rows)
        overall_row = {
            "用户":     "总体",
            "L1字母准确率": level1_overall["字母准确率"],
            "L1字母数":     level1_overall["字母数"],
            "L1数字准确率": level1_overall["数字准确率"],
            "L1数字数":     level1_overall["数字数"],
        }
        for lvl in ("LEVEL2", "LEVEL3"):
            px = lv_cn[lvl]
            ov = overall.get(lvl)
            overall_row[f"{px}词准确率"]       = ov["词准确率"]       if ov else None
            overall_row[f"{px}Top3词准确率"]   = ov["Top3词准确率"]   if ov else None
            overall_row[f"{px}字符错误率"]     = ov["字符错误率"]     if ov else None
            overall_row[f"{px}Top3字符错误率"] = ov["Top3字符错误率"] if ov else None
            overall_row[f"{px}样本数"]         = ov["样本数"]         if ov else None
        writer.writerow(overall_row)
    print(f"CSV 已保存 → {csv_path}")


if __name__ == "__main__":
    main()
