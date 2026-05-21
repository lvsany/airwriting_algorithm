#!/usr/bin/env python3
"""
Tune ENTER_SCALE and EXIT_SCALE by comparing predicted contact vs manual_state in a session CSV.
"""

import argparse
import csv
from typing import List, Dict, Optional, Tuple


def _parse_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _norm_state(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = val.strip().lower()
    if s in ("idle", "contact"):
        return s
    return None


def _load_rows(path: str) -> List[Dict[str, Optional[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prepare(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    data = []
    for r in rows:
        manual = _norm_state(r.get("manual_state"))
        if manual is None:
            continue
        data.append({
            "phase": (r.get("phase") or "").strip().lower(),
            "distance": _parse_float(r.get("distance")),
            "threshold": _parse_float(r.get("threshold")),
            "z0": _parse_float(r.get("z_0")),
            "manual": manual,
        })
    return data


def _eval_scales(
    data: List[Dict[str, object]],
    enter_scale: float,
    exit_scale: float,
    ready_only: bool,
) -> Tuple[float, Dict[str, int]]:
    in_contact = False
    correct = total = 0
    tp = tn = fp = fn = 0

    for r in data:
        phase = r["phase"]
        D = r["distance"]
        tau = r["threshold"]
        z0 = r["z0"]
        manual = r["manual"]

        if phase != "ready" or D is None or tau is None:
            if ready_only:
                continue
            in_contact = False
            pred = "idle"
        else:
            z0_ok = (z0 is not None) and (z0 != 0.0) and (z0 < 0.0)
            enter_tau = tau * enter_scale
            exit_tau = tau * exit_scale
            if not in_contact:
                pred_contact = (D > enter_tau) and z0_ok
            else:
                pred_contact = D > exit_tau
            in_contact = pred_contact
            pred = "contact" if pred_contact else "idle"

        total += 1
        if pred == manual:
            correct += 1
        if pred == "contact" and manual == "contact":
            tp += 1
        elif pred == "idle" and manual == "idle":
            tn += 1
        elif pred == "contact" and manual == "idle":
            fp += 1
        elif pred == "idle" and manual == "contact":
            fn += 1

    acc = correct / total if total else 0.0
    return acc, {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total}


def _frange(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round((end - start) / step)) + 1
    out = []
    for i in range(n):
        v = start + i * step
        out.append(round(v, 6))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Session CSV path with manual_state column")
    ap.add_argument("--enter-min", type=float, default=0.8, dest="enter_min",
                    help="Minimum ENTER_SCALE to test")
    ap.add_argument("--enter-max", type=float, default=1.2, dest="enter_max",
                    help="Maximum ENTER_SCALE to test")
    ap.add_argument("--enter-step", type=float, default=0.01, dest="enter_step",
                    help="Step size for ENTER_SCALE sweep")
    ap.add_argument("--exit-min", "--min", type=float, default=0.6, dest="exit_min",
                    help="Minimum EXIT_SCALE to test")
    ap.add_argument("--exit-max", "--max", type=float, default=1.1, dest="exit_max",
                    help="Maximum EXIT_SCALE to test")
    ap.add_argument("--exit-step", "--step", type=float, default=0.01, dest="exit_step",
                    help="Step size for EXIT_SCALE sweep")
    ap.add_argument("--topk", type=int, default=5,
                    help="Show top-k scale pairs by accuracy")
    ap.add_argument("--ready-only", action="store_true",
                    help="Only evaluate frames with phase=ready")
    args = ap.parse_args()

    rows = _load_rows(args.csv)
    data = _prepare(rows)
    if not data:
        print("[ERR] No usable rows with manual_state in CSV.")
        return 1

    enter_scales = _frange(args.enter_min, args.enter_max, args.enter_step)
    exit_scales = _frange(args.exit_min, args.exit_max, args.exit_step)
    results = []
    for es in enter_scales:
        for xs in exit_scales:
            acc, counts = _eval_scales(data, es, xs, args.ready_only)
            results.append((es, xs, acc, counts))

    results.sort(key=lambda x: (x[2], -x[0], -x[1]), reverse=True)
    best_enter, best_exit, best_acc, best_counts = results[0]

    mode = "ready-only" if args.ready_only else "all-frames"
    print(f"[BEST] ENTER_SCALE={best_enter:.3f}  EXIT_SCALE={best_exit:.3f}  "
          f"acc={best_acc*100:.2f}%  mode={mode}")
    print(f"       tp={best_counts['tp']}  tn={best_counts['tn']}  "
          f"fp={best_counts['fp']}  fn={best_counts['fn']}  total={best_counts['total']}")

    if args.topk > 1:
        print("\n[TOP]")
        for es, xs, acc, counts in results[:args.topk]:
            print(f"  enter={es:.3f}  exit={xs:.3f}  acc={acc*100:.2f}%  "
                  f"tp={counts['tp']}  tn={counts['tn']}  fp={counts['fp']}  fn={counts['fn']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
