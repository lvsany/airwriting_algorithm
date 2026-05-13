"""
Exp-B3 完整分析脚本（Palm-on-Hand Writing）

任务覆盖：
  Task B1 ~ B8
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =========================
# 全局常量
# =========================
RANDOM_STATE = 42
FPR_GRID = np.linspace(0.0, 1.0, 200)
WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

EXPA_RESULTS = {
    "baseline": {"auroc": 0.484, "std": 0.039, "prauc": 0.745},
    "geo_wrist": {"auroc": 0.943, "std": 0.017, "prauc": 0.971},
    "geo_5pt": {"auroc": 0.957, "std": 0.017, "prauc": 0.972},
    "kinematic": {"auroc": 0.480, "std": 0.035, "prauc": 0.747},
    "geo+theta": {"auroc": 0.964, "std": 0.023, "prauc": 0.982},
    "geo+optical": {"auroc": 0.949, "std": 0.019, "prauc": 0.974},
    "all_fusion": {"auroc": 0.977, "std": 0.003, "prauc": 0.992},
}

EXPA_SINGLE = {
    "dist2d_palm_0": 0.943,
    "approach_theta": 0.789,
    "dist2d_palm_17": 0.761,
    "sigma_d": 0.660,
    "brightness_contact": 0.650,
    "shadow_score": 0.633,
    "dist2d_palm_13": 0.632,
    "dist2d_palm_5": 0.626,
    "flow_mag": 0.592,
    "dist2d_palm_9": 0.536,
    "v_t": 0.510,
    "v_n": 0.500,
    "a_n": 0.495,
    "dist_raw": 0.484,
    "dist_local": 0.484,
    "hull_overlap_ratio": 0.469,
}

FEATURE_COLUMNS = list(EXPA_SINGLE.keys())

COMBOS_TRANSFER = {
    "baseline": ["dist_raw"],
    "geo_wrist": ["dist2d_palm_0"],
    "geo_5pt": ["dist2d_palm_0", "dist2d_palm_5", "dist2d_palm_9", "dist2d_palm_13", "dist2d_palm_17"],
    "kinematic": ["dist_raw", "v_n", "sigma_d"],
    "geo+theta": ["dist2d_palm_0", "approach_theta"],
    "geo+optical": ["dist2d_palm_0", "shadow_score", "flow_mag", "brightness_contact"],
    "all_fusion": [
        "dist_raw", "dist_local", "v_n", "a_n", "sigma_d", "v_t",
        "approach_theta", "shadow_score", "flow_mag", "brightness_contact",
        "dist2d_palm_0", "dist2d_palm_5", "dist2d_palm_9", "dist2d_palm_13",
        "dist2d_palm_17", "hull_overlap_ratio",
    ],
}

COMBOS_WRITING = {
    "geo+vt": ["dist2d_palm_0", "v_t"],
    "geo+theta+vt": ["dist2d_palm_0", "approach_theta", "v_t"],
    "geo+optical+vt": ["dist2d_palm_0", "shadow_score", "flow_mag", "brightness_contact", "v_t"],
}

# ------ 配对受试者（预实验）------
# Exp-A sid  <->  Exp-B sid（s02=s02, s01=s01）
SUBJECT_PAIRS = [
    {"sid_a": "s01", "sid_b": "s01", "label": "S1 (s01)"},
    {"sid_a": "s02", "sid_b": "s02", "label": "S2 (s02)"},
]
KDE_FEATS_PAIRED = ["dist2d_palm_0", "approach_theta", "v_t", "sigma_d", "brightness_contact", "flow_mag"]
COMBOS_PAIRED = {
    "geo_wrist":   COMBOS_TRANSFER["geo_wrist"],
    "geo+theta":   COMBOS_TRANSFER["geo+theta"],
    "geo+optical": COMBOS_TRANSFER["geo+optical"],
    "all_fusion":  COMBOS_TRANSFER["all_fusion"],
}


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9


# =========================
# 基础工具
# =========================
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _sig_text(p: float) -> str:
    if np.isnan(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _combo_color(name: str) -> str:
    if name == "baseline":
        return "#999999"
    if name == "kinematic":
        return "#BBBBBB"
    color_map = {
        "geo_wrist": WONG[5],
        "geo_5pt": WONG[2],
        "geo+theta": WONG[6],
        "geo+optical": WONG[3],
        "all_fusion": WONG[1],
        "geo+vt": WONG[7],
        "geo+theta+vt": "#8B5CF6",
        "geo+optical+vt": "#14B8A6",
    }
    return color_map.get(name, WONG[0])


def _save_fig(fig: plt.Figure, save_dir: Path, stem: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf = save_dir / f"{stem}.pdf"
    png = save_dir / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print("图表已保存：")
    print(f"  {pdf.as_posix()}")
    print(f"  {png.as_posix()}")


def _ensure_label_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contact_label"] = pd.to_numeric(out["contact_label"], errors="coerce")
    return out


def _format_meta_from_filename(path: Path) -> Optional[Tuple[str, str, str]]:
    # 支持：
    # exp_b1_{sid}_{lighting}_{speed}.csv
    # exp_b1_{sid}_{lighting}_{speed}_features.csv
    # exp_b1_{sid}_{lighting}_{speed}_features_labeled.csv
    m = re.match(
        r"^exp_b1_(?P<sid>[^_]+)_(?P<lighting>[^_]+)_(?P<speed>[^_]+)"
        r"(?:_features(?:_labeled)?)?\.csv$",
        path.name,
    )
    if not m:
        return None
    return m.group("sid"), m.group("lighting"), m.group("speed")


def _contact_durations(df: pd.DataFrame) -> List[int]:
    durations: List[int] = []
    for _, g in df.groupby("source_file"):
        y = pd.to_numeric(g["contact_label"], errors="coerce").fillna(0).astype(int).to_numpy()
        run = 0
        for v in y:
            if v == 1:
                run += 1
            elif run > 0:
                durations.append(run)
                run = 0
        if run > 0:
            durations.append(run)
    return durations


def _count_onsets(df: pd.DataFrame) -> int:
    cnt = 0
    for _, g in df.groupby("source_file"):
        g = g.sort_values("frame_id")
        y = pd.to_numeric(g["contact_label"], errors="coerce").fillna(0).astype(int).to_numpy()
        if len(y) < 2:
            continue
        cnt += int(np.sum((y[1:] == 1) & (y[:-1] == 0)))
    return cnt


def _collect_onset_windows(df: pd.DataFrame, feature: str, pre: int = 20, post: int = 20) -> np.ndarray:
    windows: List[np.ndarray] = []
    for _, g in df.groupby("source_file"):
        g = g.sort_values("frame_id")
        y = pd.to_numeric(g["contact_label"], errors="coerce").fillna(0).astype(int).to_numpy()
        x = pd.to_numeric(g[feature], errors="coerce").to_numpy(dtype=float)
        if len(y) < pre + post + 2:
            continue
        onsets = np.where((y[1:] == 1) & (y[:-1] == 0))[0] + 1
        for idx in onsets:
            s = idx - pre
            e = idx + post + 1
            if s < 0 or e > len(y):
                continue
            windows.append(x[s:e])
    if not windows:
        return np.empty((0, pre + post + 1), dtype=float)
    return np.vstack(windows)


# =========================
# 数据加载
# =========================
def load_expb(data_dir: str) -> pd.DataFrame:
    """
    扫描 data_b/ 下所有 exp_b1_*.csv，合并，解析 sid/lighting/speed。
    打印各受试者帧数和检测率。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Exp-B 数据目录不存在：{data_path}")

    csv_files = sorted(data_path.glob("exp_b1_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 Exp-B CSV：{data_path}/exp_b1_*.csv")

    frames: List[pd.DataFrame] = []
    skipped: List[str] = []
    for f in csv_files:
        meta = _format_meta_from_filename(f)
        if meta is None:
            skipped.append(f.name)
            continue
        sid, lighting, speed = meta
        df = pd.read_csv(f)
        if "contact_label" not in df.columns:
            skipped.append(f.name)
            continue
        df["contact_label"] = pd.to_numeric(df["contact_label"], errors="coerce")
        # 只保留至少含有部分有效标签的数据（B2 后）
        if df["contact_label"].isin([0, 1]).sum() == 0:
            skipped.append(f.name)
            continue
        df["sid"] = sid
        df["lighting"] = lighting
        df["speed"] = speed
        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        raise RuntimeError("未找到可分析的 Exp-B 标注 CSV（contact_label 全为空）。")

    df_b = pd.concat(frames, ignore_index=True)
    df_b = _ensure_label_numeric(df_b)

    print("\n========== Exp-B 数据加载 ==========")
    print(f"已加载文件数：{len(frames)}")
    if skipped:
        print(f"跳过文件数：{len(skipped)}")
        for name in skipped:
            print(f"  - {name}")
    print(f"总帧数：{len(df_b)}")
    print(f"受试者数：{df_b['sid'].nunique()}")
    print("------------------------------------")
    print("按受试者统计：")
    for sid, g in df_b.groupby("sid"):
        total = len(g)
        y = g["contact_label"]
        valid_y = y.isin([0, 1]).sum()
        contact_ratio = (y.eq(1).sum() / valid_y) if valid_y > 0 else np.nan
        detect_rate = pd.to_numeric(g["dist2d_palm_0"], errors="coerce").notna().mean()
        print(
            f"  {sid:<8} total={total:<6d} "
            f"contact={contact_ratio * 100:>6.1f}% "
            f"detect={detect_rate * 100:>6.1f}%"
        )
    print("------------------------------------")
    print("按速度分组帧数：")
    for spd, n in df_b.groupby("speed").size().items():
        print(f"  {spd:<8} {n}")
    print("按光照分组帧数：")
    for lit, n in df_b.groupby("lighting").size().items():
        print(f"  {lit:<8} {n}")
    print("====================================")
    return df_b


def load_expa(csv_path: str) -> pd.DataFrame:
    """加载 Exp-A 数据。"""
    p = Path(csv_path)
    if not p.exists():
        candidates = sorted((Path(__file__).resolve().parents[2]).glob("data/**/exp_a1_s01.csv"))
        if not candidates:
            raise FileNotFoundError(f"Exp-A CSV 不存在：{csv_path}")
        p = candidates[0]
        print(f"[提示] 使用回退 Exp-A 文件：{p.as_posix()}")

    df = pd.read_csv(p)
    df = _ensure_label_numeric(df)
    df["sid"] = "s01"
    df["lighting"] = "normal"
    df["speed"] = "normal"
    df["source_file"] = p.name
    return df


def load_expa_all(data_dir: str) -> Dict[str, pd.DataFrame]:
    """扫描 data_dir/exp_a1_s*.csv，返回 {sid: df} 字典。"""
    import glob as _glob
    files = sorted(_glob.glob(str(Path(data_dir) / "exp_a1_s*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到 Exp-A CSV：{data_dir}/exp_a1_s*.csv")
    out: Dict[str, pd.DataFrame] = {}
    for f in files:
        m = re.match(r".*exp_a1_(s\d+)\.csv$", f)
        if not m:
            continue
        sid = m.group(1)
        df = pd.read_csv(f)
        df = _ensure_label_numeric(df)
        df["sid"] = sid
        df["source_file"] = Path(f).name
        out[sid] = df
        print(f"  [Exp-A] {sid}: {len(df)} frames")
    return out


def get_valid_subset(df: pd.DataFrame, features: Sequence[str]):
    """组合级别 dropna，返回 X, y, sample_ids，并打印有效帧数。"""
    cols = ["contact_label", *features]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺失列：{missing}")

    sub = df[cols].copy()
    sub["contact_label"] = pd.to_numeric(sub["contact_label"], errors="coerce")
    for f in features:
        sub[f] = pd.to_numeric(sub[f], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub[sub["contact_label"].isin([0, 1])]
    sub = sub.dropna(subset=features)
    X = sub[list(features)].to_numpy(dtype=float)
    y = sub["contact_label"].to_numpy(dtype=int)
    sample_ids = sub.index.to_numpy()
    print(f"  [get_valid_subset] features={list(features)}  n_valid={len(sub)}")
    return X, y, sample_ids


# =========================
# 模型评估
# =========================
def run_cv(X, y, combo_name):
    """
    5-fold StratifiedKFold + LR，返回完整结果字典。
    包含 oof_proba 和 oof_true 用于 DeLong test。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    result = {
        "combo_name": combo_name,
        "n_valid": int(n),
        "auroc_mean": np.nan,
        "auroc_std": np.nan,
        "prauc_mean": np.nan,
        "prauc_std": np.nan,
        "fold_aurocs": [],
        "fold_praucs": [],
        "fpr_grid": FPR_GRID.copy(),
        "tpr_mean": np.full_like(FPR_GRID, np.nan, dtype=float),
        "tpr_std": np.full_like(FPR_GRID, np.nan, dtype=float),
        "oof_true": np.array([], dtype=int),
        "oof_proba": np.array([], dtype=float),
    }

    if n < 20 or len(np.unique(y)) < 2:
        return result

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.full(n, np.nan, dtype=float)
    oof_true = y.copy()
    tprs: List[np.ndarray] = []
    fold_aurocs: List[float] = []
    fold_praucs: List[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            clf.fit(X_train_s, y_train)
            proba = clf.predict_proba(X_test_s)[:, 1]

        oof_proba[test_idx] = proba
        fold_aurocs.append(roc_auc_score(y_test, proba))
        fold_praucs.append(average_precision_score(y_test, proba))

        fpr, tpr, _ = roc_curve(y_test, proba)
        tpr_interp = np.interp(FPR_GRID, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tprs.append(tpr_interp)

    valid_oof = np.isfinite(oof_proba)
    if np.any(valid_oof):
        result["oof_true"] = oof_true[valid_oof]
        result["oof_proba"] = oof_proba[valid_oof]
    result["fold_aurocs"] = fold_aurocs
    result["fold_praucs"] = fold_praucs

    if fold_aurocs:
        result["auroc_mean"] = float(np.mean(fold_aurocs))
        result["auroc_std"] = float(np.std(fold_aurocs))
    if fold_praucs:
        result["prauc_mean"] = float(np.mean(fold_praucs))
        result["prauc_std"] = float(np.std(fold_praucs))
    if tprs:
        arr = np.vstack(tprs)
        result["tpr_mean"] = np.mean(arr, axis=0)
        result["tpr_std"] = np.std(arr, axis=0)
    return result


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Computes midranks used in DeLong algorithm."""
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        midranks[i:j] = mid
        i = j

    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """
    Fast DeLong implementation (Sun & Xu, 2014) for correlated ROC AUCs.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true_a, y_score_a, y_true_b, y_score_b) -> float:
    """
    手动实现 DeLong 1988。
    输入需是同一批样本（外层已做交集对齐）。
    """
    y_true_a = np.asarray(y_true_a).astype(int)
    y_true_b = np.asarray(y_true_b).astype(int)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)

    mask = (
        np.isfinite(y_score_a) &
        np.isfinite(y_score_b) &
        np.isin(y_true_a, [0, 1]) &
        np.isin(y_true_b, [0, 1])
    )
    y_true_a = y_true_a[mask]
    y_true_b = y_true_b[mask]
    y_score_a = y_score_a[mask]
    y_score_b = y_score_b[mask]

    if len(y_true_a) < 20:
        return np.nan
    if not np.array_equal(y_true_a, y_true_b):
        # 极端情况下若不一致，保守返回 NaN
        return np.nan
    if len(np.unique(y_true_a)) < 2:
        return np.nan

    order = np.argsort(-y_true_a)  # positives first
    y_true = y_true_a[order]
    preds = np.vstack([y_score_a[order], y_score_b[order]])
    m = int(np.sum(y_true))
    n = len(y_true) - m
    if m <= 0 or n <= 0:
        return np.nan

    aucs, cov = _fast_delong(preds, m)
    if np.ndim(cov) == 0:
        var = float(cov)
    else:
        var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 1e-12:
        return np.nan

    z = abs(aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * stats.norm.sf(z)
    return float(p)


def _align_oof_for_delong(res_a: dict, res_b: dict):
    s_a = pd.Series(res_a.get("oof_proba", []), index=res_a.get("sample_ids", []))
    s_b = pd.Series(res_b.get("oof_proba", []), index=res_b.get("sample_ids", []))
    y_a = pd.Series(res_a.get("oof_true", []), index=res_a.get("sample_ids", []))
    y_b = pd.Series(res_b.get("oof_true", []), index=res_b.get("sample_ids", []))
    common = s_a.index.intersection(s_b.index)
    if len(common) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (
        y_a.loc[common].to_numpy(),
        s_a.loc[common].to_numpy(),
        y_b.loc[common].to_numpy(),
        s_b.loc[common].to_numpy(),
    )


def cross_scene_eval(df_a, df_b, features, combo_name) -> dict:
    """
    setting_A: train on A, test on B（zero-shot）
    setting_B: train/test on B（5-fold CV）
    """
    out = {
        "combo_name": combo_name,
        "setting_A_auroc": np.nan,
        "setting_A_prauc": np.nan,
        "setting_A_n_train": 0,
        "setting_A_n_test": 0,
        "setting_B_auroc": np.nan,
        "setting_B_std": np.nan,
        "setting_B_prauc": np.nan,
        "setting_B_prauc_std": np.nan,
    }

    # A->B
    a_sub = df_a[["contact_label", *features]].copy()
    a_sub["contact_label"] = pd.to_numeric(a_sub["contact_label"], errors="coerce")
    for f in features:
        a_sub[f] = pd.to_numeric(a_sub[f], errors="coerce")
    a_sub = a_sub.replace([np.inf, -np.inf], np.nan)
    a_sub = a_sub[a_sub["contact_label"].isin([0, 1])].dropna(subset=features)
    b_sub = df_b[["contact_label", *features]].copy()
    b_sub["contact_label"] = pd.to_numeric(b_sub["contact_label"], errors="coerce")
    for f in features:
        b_sub[f] = pd.to_numeric(b_sub[f], errors="coerce")
    b_sub = b_sub.replace([np.inf, -np.inf], np.nan)
    b_sub = b_sub[b_sub["contact_label"].isin([0, 1])].dropna(subset=features)

    if len(a_sub) >= 20 and len(b_sub) >= 20 and a_sub["contact_label"].nunique() == 2 and b_sub["contact_label"].nunique() == 2:
        X_train = a_sub[list(features)].to_numpy(float)
        y_train = a_sub["contact_label"].to_numpy(int)
        X_test = b_sub[list(features)].to_numpy(float)
        y_test = b_sub["contact_label"].to_numpy(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            clf.fit(X_train_s, y_train)
            proba = clf.predict_proba(X_test_s)[:, 1]

        out["setting_A_auroc"] = float(roc_auc_score(y_test, proba))
        out["setting_A_prauc"] = float(average_precision_score(y_test, proba))
        out["setting_A_n_train"] = int(len(y_train))
        out["setting_A_n_test"] = int(len(y_test))

    # B in-domain
    X_b, y_b, _ids = get_valid_subset(df_b, features)
    res_b = run_cv(X_b, y_b, combo_name=f"{combo_name}_setting_B")
    out["setting_B_auroc"] = _safe_float(res_b.get("auroc_mean", np.nan))
    out["setting_B_std"] = _safe_float(res_b.get("auroc_std", np.nan))
    out["setting_B_prauc"] = _safe_float(res_b.get("prauc_mean", np.nan))
    out["setting_B_prauc_std"] = _safe_float(res_b.get("prauc_std", np.nan))
    return out


def loso_cv(df_b, features, combo_name) -> dict:
    """Leave-One-Subject-Out，按 sid 分组。"""
    out = {
        "combo_name": combo_name,
        "subject_auc": {},
        "mean": np.nan,
        "std": np.nan,
    }
    aucs = []
    for sid in sorted(df_b["sid"].dropna().unique()):
        train = df_b[df_b["sid"] != sid]
        test = df_b[df_b["sid"] == sid]

        train_sub = train[["contact_label", *features]].copy()
        train_sub["contact_label"] = pd.to_numeric(train_sub["contact_label"], errors="coerce")
        for f in features:
            train_sub[f] = pd.to_numeric(train_sub[f], errors="coerce")
        train_sub = train_sub.replace([np.inf, -np.inf], np.nan)
        train_sub = train_sub[train_sub["contact_label"].isin([0, 1])].dropna(subset=features)

        test_sub = test[["contact_label", *features]].copy()
        test_sub["contact_label"] = pd.to_numeric(test_sub["contact_label"], errors="coerce")
        for f in features:
            test_sub[f] = pd.to_numeric(test_sub[f], errors="coerce")
        test_sub = test_sub.replace([np.inf, -np.inf], np.nan)
        test_sub = test_sub[test_sub["contact_label"].isin([0, 1])].dropna(subset=features)

        if len(train_sub) < 20 or len(test_sub) < 20:
            out["subject_auc"][sid] = np.nan
            continue
        if train_sub["contact_label"].nunique() < 2 or test_sub["contact_label"].nunique() < 2:
            out["subject_auc"][sid] = np.nan
            continue

        X_tr = train_sub[list(features)].to_numpy(float)
        y_tr = train_sub["contact_label"].to_numpy(int)
        X_te = test_sub[list(features)].to_numpy(float)
        y_te = test_sub["contact_label"].to_numpy(int)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            clf.fit(X_tr_s, y_tr)
            proba = clf.predict_proba(X_te_s)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
        out["subject_auc"][sid] = auc
        aucs.append(auc)

    if aucs:
        out["mean"] = float(np.mean(aucs))
        out["std"] = float(np.std(aucs))
    return out


# =========================
# 单特征指标
# =========================
def _cohens_d_abs(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    v0 = np.var(x0, ddof=1)
    v1 = np.var(x1, ddof=1)
    pooled = np.sqrt(((len(x0) - 1) * v0 + (len(x1) - 1) * v1) / (len(x0) + len(x1) - 2))
    if pooled < 1e-12:
        return np.nan
    return float(abs((np.mean(x1) - np.mean(x0)) / pooled))


def _rank_biserial_abs(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) == 0 or len(x1) == 0:
        return np.nan
    try:
        u, _ = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        r = 2.0 * u / (len(x1) * len(x0)) - 1.0
        return float(abs(r))
    except Exception:
        return np.nan


def _bhattacharyya_overlap(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    low = min(np.percentile(x0, 1), np.percentile(x1, 1))
    high = max(np.percentile(x0, 99), np.percentile(x1, 99))
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
        return np.nan
    grid = np.linspace(low, high, 256)
    try:
        p0 = gaussian_kde(x0)(grid)
        p1 = gaussian_kde(x1)(grid)
        bc = np.trapz(np.sqrt(np.maximum(p0, 0) * np.maximum(p1, 0)), grid)
        return float(np.clip(bc, 0.0, 1.0))
    except Exception:
        return np.nan


def _single_feature_stats(df: pd.DataFrame, feature: str) -> Dict[str, float]:
    sub = df[["contact_label", feature]].copy()
    sub["contact_label"] = pd.to_numeric(sub["contact_label"], errors="coerce")
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub[sub["contact_label"].isin([0, 1])].dropna(subset=[feature])
    out = {"auroc": np.nan, "d_abs": np.nan, "rb_abs": np.nan, "bc": np.nan, "n": len(sub)}
    if len(sub) < 20:
        return out
    y = sub["contact_label"].astype(int).to_numpy()
    x = sub[feature].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return out
    x0 = x[y == 0]
    x1 = x[y == 1]
    try:
        res = run_cv(x.reshape(-1, 1), y, combo_name=f"single_{feature}")
        out["auroc"] = _safe_float(res["auroc_mean"])
    except Exception:
        pass
    out["d_abs"] = _cohens_d_abs(x0, x1)
    out["rb_abs"] = _rank_biserial_abs(x0, x1)
    out["bc"] = _bhattacharyya_overlap(x0, x1)
    return out


# =========================
# 绘图任务
# =========================
def plot_taskB1(df_b, df_a, save_dir):
    print("\n========== Task B1: Exp-B 数据概况 ==========")
    labels_b = pd.to_numeric(df_b["contact_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    labels_a = pd.to_numeric(df_a["contact_label"], errors="coerce").fillna(0).astype(int).to_numpy()

    durations_b = _contact_durations(df_b)
    durations_a = _contact_durations(df_a)
    med_b = float(np.median(durations_b)) if durations_b else np.nan
    med_a = float(np.median(durations_a)) if durations_a else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axa, axb, axc, axd = axes.ravel()

    # (a) contact_label 时间轴
    axa.plot(np.arange(len(labels_b)), labels_b, color=WONG[5], lw=0.8)
    axa.set_title("(a) Contact Label Timeline (Exp-B)")
    axa.set_xlabel("Global frame index")
    axa.set_ylabel("contact_label")
    axa.set_ylim(-0.1, 1.1)
    axa.grid(alpha=0.25)

    # (b) 接触事件持续帧数分布
    if durations_b:
        axb.hist(durations_b, bins=min(40, max(10, len(set(durations_b)))), color=WONG[1], alpha=0.8, edgecolor="k")
    if np.isfinite(med_b):
        axb.axvline(med_b, color=WONG[6], ls="--", lw=2, label=f"Exp-B median={med_b:.1f}")
    if np.isfinite(med_a):
        axb.axvline(med_a, color=WONG[2], ls="--", lw=2, label=f"Exp-A median={med_a:.1f}")
    axb.set_title("(b) Contact Event Duration Distribution")
    axb.set_xlabel("Duration (frames)")
    axb.set_ylabel("Count")
    axb.legend(loc="best")
    axb.grid(alpha=0.25)

    # (c) IDLE vs CONTACT 帧数
    b_idle = int(np.sum(labels_b == 0))
    b_contact = int(np.sum(labels_b == 1))
    total = max(1, b_idle + b_contact)
    bars = axc.bar(["IDLE", "CONTACT"], [b_idle, b_contact], color=[WONG[2], WONG[6]], edgecolor="k")
    axc.set_title("(c) IDLE vs CONTACT Frame Counts (Exp-B)")
    axc.set_ylabel("Frames")
    for rect, val in zip(bars, [b_idle, b_contact]):
        axc.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{val}\n({val / total * 100:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    axc.grid(axis="y", alpha=0.25)

    # (d) 有效帧比例并排（Exp-A vs Exp-B）
    feats = [f for f in FEATURE_COLUMNS if f in df_b.columns and f in df_a.columns]
    valid_b = np.array([pd.to_numeric(df_b[f], errors="coerce").notna().mean() for f in feats])
    valid_a = np.array([pd.to_numeric(df_a[f], errors="coerce").notna().mean() for f in feats])
    y = np.arange(len(feats))
    h = 0.35
    axd.barh(y + h / 2, valid_b, h, color=WONG[5], label="Exp-B Valid")
    axd.barh(y + h / 2, 1.0 - valid_b, h, left=valid_b, color="#dddddd", label="Exp-B NaN")
    axd.barh(y - h / 2, valid_a, h, color=WONG[2], alpha=0.9, label="Exp-A Valid")
    axd.barh(y - h / 2, 1.0 - valid_a, h, left=valid_a, color="#f0f0f0", label="Exp-A NaN")
    axd.set_yticks(y)
    axd.set_yticklabels(feats, fontsize=9)
    axd.set_xlim(0, 1.0)
    axd.set_title("(d) Feature Valid/NaN Ratio (Exp-A vs Exp-B)")
    axd.set_xlabel("Ratio")
    axd.grid(axis="x", alpha=0.25)
    legend_elements = [
        Patch(facecolor=WONG[5], label="Exp-B Valid"),
        Patch(facecolor="#dddddd", label="Exp-B NaN"),
        Patch(facecolor=WONG[2], label="Exp-A Valid"),
        Patch(facecolor="#f0f0f0", label="Exp-A NaN"),
    ]
    axd.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, save_dir, "task_b1_overview")
    plt.close(fig)

    print("额外统计：")
    print("每个受试者（total/contact_ratio/detect_rate）")
    for sid, g in df_b.groupby("sid"):
        total_s = len(g)
        valid_y = g["contact_label"].isin([0, 1]).sum()
        c_ratio = g["contact_label"].eq(1).sum() / max(1, valid_y)
        d_rate = pd.to_numeric(g["dist2d_palm_0"], errors="coerce").notna().mean()
        print(f"  {sid:<8} total={total_s:<6d} contact={c_ratio * 100:>6.1f}% detect={d_rate * 100:>6.1f}%")
    print("速度分组帧数：")
    print(df_b.groupby("speed").size().to_string())
    print("光照分组帧数：")
    print(df_b.groupby("lighting").size().to_string())
    print("==============================================")


def plot_taskB2(df_b, df_a, save_dir):
    print("\n========== Task B2: 单特征判别力 ==========")
    rows = []
    for feat in FEATURE_COLUMNS:
        if feat not in df_b.columns:
            continue
        st = _single_feature_stats(df_b, feat)
        st["feature"] = feat
        st["delta_auroc"] = st["auroc"] - EXPA_SINGLE.get(feat, np.nan)
        rows.append(st)
    if not rows:
        raise RuntimeError("Task B2 无可用特征数据。")

    res = pd.DataFrame(rows).sort_values("auroc", ascending=False, na_position="last").reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(22, 8))
    metrics = [("auroc", "AUROC (5-fold CV LR)"),
               ("d_abs", "|Cohen's d|"),
               ("rb_abs", "Rank-biserial |r|"),
               ("bc", "Bhattacharyya overlap")]

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        y = np.arange(len(res))
        colors = []
        for _, r in res.iterrows():
            if metric == "auroc" and r["feature"] == "v_t" and np.isfinite(r["auroc"]) and r["auroc"] > 0.65:
                colors.append(WONG[6])
            else:
                colors.append(WONG[5] if metric == "auroc" else WONG[2])
        ax.barh(y, res[metric].to_numpy(dtype=float), color=colors, edgecolor="k", alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels(res["feature"], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"({chr(ord('a') + i)}) {title}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        if metric == "auroc":
            for yi, r in res.iterrows():
                delta = r["delta_auroc"]
                if not np.isfinite(delta):
                    continue
                if abs(delta) < 0.02:
                    txt, c = f"→{delta:+.2f}", "#666666"
                elif delta > 0:
                    txt, c = f"↑{delta:+.2f}", "#2E8B57"
                else:
                    txt, c = f"↓{delta:+.2f}", "#C0392B"
                x_val = _safe_float(r["auroc"])
                if not np.isfinite(x_val):
                    continue
                ax.text(x_val + 0.01, yi, txt, va="center", ha="left", color=c, fontsize=9, fontweight="bold")
            ax.set_xlim(0.0, min(1.0, max(0.8, np.nanmax(res["auroc"]) + 0.18)))

    fig.tight_layout()
    _save_fig(fig, save_dir, "task_b2_discriminability")
    plt.close(fig)

    print(res[["feature", "auroc", "d_abs", "rb_abs", "bc", "delta_auroc"]].to_string(index=False))
    print("===========================================")


def _plot_kde_panel(ax, x0: np.ndarray, x1: np.ndarray, title: str, dataset: str, auroc: float, d_abs: float):
    c_idle = WONG[5]
    c_contact = WONG[6]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x0) > 1:
            sns.kdeplot(x=x0, ax=ax, color=c_idle, lw=2, label="IDLE")
        if len(x1) > 1:
            sns.kdeplot(x=x1, ax=ax, color=c_contact, lw=2, label="CONTACT")
    if len(x0) > 1 and len(x1) > 1:
        lo = min(np.percentile(x0, 1), np.percentile(x1, 1))
        hi = max(np.percentile(x0, 99), np.percentile(x1, 99))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            grid = np.linspace(lo, hi, 256)
            try:
                p0 = gaussian_kde(x0)(grid)
                p1 = gaussian_kde(x1)(grid)
                ax.fill_between(grid, np.minimum(p0, p1), color="#999999", alpha=0.25)
            except Exception:
                pass
    ax.set_title(f"{title} | {dataset}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.95,
        f"AUROC={auroc:.3f}\n|d|={d_abs:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )


def plot_taskB3(df_b, df_a, save_dir):
    print("\n========== Task B3: KDE 分布对比 ==========")
    feats = ["dist2d_palm_0", "approach_theta", "v_t", "sigma_d", "brightness_contact", "flow_mag"]
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharey=False)
    for i, feat in enumerate(feats):
        # Exp-A
        sa = _single_feature_stats(df_a, feat)
        sub_a = df_a[["contact_label", feat]].copy()
        sub_a["contact_label"] = pd.to_numeric(sub_a["contact_label"], errors="coerce")
        sub_a[feat] = pd.to_numeric(sub_a[feat], errors="coerce")
        sub_a = sub_a[sub_a["contact_label"].isin([0, 1])].dropna(subset=[feat])
        x0_a = sub_a.loc[sub_a["contact_label"] == 0, feat].to_numpy(float)
        x1_a = sub_a.loc[sub_a["contact_label"] == 1, feat].to_numpy(float)
        _plot_kde_panel(axes[0, i], x0_a, x1_a, feat, "Exp-A", _safe_float(sa["auroc"]), _safe_float(sa["d_abs"]))

        # Exp-B
        sb = _single_feature_stats(df_b, feat)
        sub_b = df_b[["contact_label", feat]].copy()
        sub_b["contact_label"] = pd.to_numeric(sub_b["contact_label"], errors="coerce")
        sub_b[feat] = pd.to_numeric(sub_b[feat], errors="coerce")
        sub_b = sub_b[sub_b["contact_label"].isin([0, 1])].dropna(subset=[feat])
        x0_b = sub_b.loc[sub_b["contact_label"] == 0, feat].to_numpy(float)
        x1_b = sub_b.loc[sub_b["contact_label"] == 1, feat].to_numpy(float)
        _plot_kde_panel(axes[1, i], x0_b, x1_b, feat, "Exp-B", _safe_float(sb["auroc"]), _safe_float(sb["d_abs"]))

        if i == 0:
            axes[0, i].legend(loc="upper right", fontsize=8)
            axes[1, i].legend(loc="upper right", fontsize=8)

    fig.suptitle("Task B3: Exp-A vs Exp-B KDE (IDLE vs CONTACT)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, save_dir, "task_b3_kde_compare")
    plt.close(fig)
    print("==========================================")


def plot_taskB4(df_b, df_a, save_dir):
    print("\n========== Task B4: 时序对齐对比 ==========")
    feats = ["dist2d_palm_0", "approach_theta", "v_t", "sigma_d", "flow_mag", "brightness_contact"]
    pre, post = 20, 20
    t = np.arange(-pre, post + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    axes = axes.ravel()
    for i, feat in enumerate(feats):
        ax = axes[i]
        wa = _collect_onset_windows(df_a, feat, pre=pre, post=post)
        wb = _collect_onset_windows(df_b, feat, pre=pre, post=post)

        if len(wa) > 0:
            ma = np.nanmean(wa, axis=0)
            sa = stats.sem(wa, axis=0, nan_policy="omit")
            ax.plot(t, ma, color=WONG[5], lw=2, label="Exp-A")
            ax.fill_between(t, ma - sa, ma + sa, color=WONG[5], alpha=0.2)
        if len(wb) > 0:
            mb = np.nanmean(wb, axis=0)
            sb = stats.sem(wb, axis=0, nan_policy="omit")
            ax.plot(t, mb, color=WONG[6], lw=2, ls="--", label="Exp-B")
            ax.fill_between(t, mb - sb, mb + sb, color=WONG[6], alpha=0.2)

        ax.axvline(0, color="#d62728", ls="--", lw=1.5)
        ax.set_title(f"({chr(ord('a') + i)}) {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frame offset from onset")
        ax.set_ylabel("Feature value")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="best")

    na = _count_onsets(df_a)
    nb = _count_onsets(df_b)
    fig.text(
        0.5,
        0.01,
        f"Exp-A: N={na} onset events (tap). Exp-B: N={nb} onset events (writing). Shading = ±1 SEM.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save_fig(fig, save_dir, "task_b4_temporal_alignment")
    plt.close(fig)
    print("==========================================")


def plot_taskB5(results_b, expa_results, delong_results, save_dir, df_b):
    print("\n========== Task B5: Feature Combination Ablation ==========")
    transfer_names = list(COMBOS_TRANSFER.keys())
    writing_names = list(COMBOS_WRITING.keys())
    all_names = transfer_names + writing_names

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # (a) ROC curves
    ax = axes[0]
    for name in transfer_names:
        res = results_b[name]
        color = _combo_color(name)
        if np.all(np.isfinite(res["tpr_mean"])):
            tpr_mean = res["tpr_mean"]
            tpr_std = np.nan_to_num(res["tpr_std"], nan=0.0)
            ax.plot(
                res["fpr_grid"],
                tpr_mean,
                color=color,
                lw=2,
                label=f"{name} (B:{res['auroc_mean']:.3f}, A:{expa_results[name]['auroc']:.3f})",
            )
            ax.fill_between(
                res["fpr_grid"],
                np.clip(tpr_mean - tpr_std, 0, 1),
                np.clip(tpr_mean + tpr_std, 0, 1),
                color=color,
                alpha=0.15,
            )
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
    ax.set_title("(a) ROC Curves (Transfer Combos)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # (b) AUROC bars
    ax = axes[1]
    x = np.arange(len(all_names))
    w = 0.38
    b_vals = np.array([_safe_float(results_b[n]["auroc_mean"]) for n in all_names])
    b_std = np.array([_safe_float(results_b[n]["auroc_std"]) for n in all_names])
    a_vals = np.array([_safe_float(expa_results[n]["auroc"]) if n in expa_results else np.nan for n in all_names])
    a_std = np.array([_safe_float(expa_results[n].get("std", np.nan)) if n in expa_results else np.nan for n in all_names])

    for i, name in enumerate(all_names):
        c = _combo_color(name)
        ax.bar(x[i] - w / 2, b_vals[i], w, color=c, yerr=b_std[i], capsize=4, edgecolor="k")
        if np.isfinite(a_vals[i]):
            ax.bar(x[i] + w / 2, a_vals[i], w, color=c, alpha=0.4, hatch="//", yerr=a_std[i], capsize=4, edgecolor="k")
            delta = b_vals[i] - a_vals[i]
            if np.isfinite(delta):
                dc = "#2E8B57" if delta >= 0 else "#C0392B"
                ax.text(x[i] - w / 2, b_vals[i] + 0.015, f"Δ{delta:+.3f}", ha="center", va="bottom", fontsize=8, color=dc)
        else:
            ax.text(x[i] + w / 2, 0.05, "N/A\n(new)", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.axhline(0.85, color="#d62728", ls="--", lw=1.2)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("(b) AUROC Cross-Scene Comparison", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # (c) PR-AUC bars
    ax = axes[2]
    b_p = np.array([_safe_float(results_b[n]["prauc_mean"]) for n in all_names])
    b_ps = np.array([_safe_float(results_b[n]["prauc_std"]) for n in all_names])
    a_p = np.array([_safe_float(expa_results[n]["prauc"]) if n in expa_results else np.nan for n in all_names])
    for i, name in enumerate(all_names):
        c = _combo_color(name)
        ax.bar(x[i] - w / 2, b_p[i], w, color=c, yerr=b_ps[i], capsize=4, edgecolor="k")
        if np.isfinite(a_p[i]):
            ax.bar(x[i] + w / 2, a_p[i], w, color=c, alpha=0.4, hatch="//", edgecolor="k")
        else:
            ax.text(x[i] + w / 2, 0.05, "N/A\n(new)", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=30, ha="right")
    ax.set_ylabel("PR-AUC")
    ax.set_title("(c) PR-AUC Cross-Scene Comparison", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        -0.02,
        "Dark bars = Exp-B (writing scene, 5-fold CV). "
        "Light bars = Exp-A (tap scene, reference). "
        "Δ = AUROC_B − AUROC_A. Error bars = ±1 std.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save_fig(fig, save_dir, "task_b5_ablation")
    plt.close(fig)

    # 控制台摘要
    n_subj = int(df_b["sid"].nunique())
    total_frames = int(len(df_b))
    print(f"数据：Exp-B 合并（N={n_subj}受试者，共{total_frames}帧）\n")
    print(f"{'combo':<18}{'n_valid':>8}   {'AUROC(B)':<17} {'PR-AUC(B)':<17} {'ΔAUROC(vs A)':<16}")
    print("-" * 72)
    for n in all_names:
        rb = results_b[n]
        n_valid = int(rb["n_valid"])
        au = rb["auroc_mean"]
        au_s = rb["auroc_std"]
        pr = rb["prauc_mean"]
        pr_s = rb["prauc_std"]
        if n in expa_results:
            delta = au - expa_results[n]["auroc"] if np.isfinite(au) else np.nan
            dtxt = f"{delta:+.3f}" if np.isfinite(delta) else "nan"
        else:
            dtxt = "N/A (new)"
        print(f"{n:<18}{n_valid:>8d}   {au:>5.3f} ± {au_s:<5.3f}   {pr:>5.3f} ± {pr_s:<5.3f}   {dtxt}")
    print("-" * 72)
    for key, p in delong_results.items():
        print(f"DeLong test {key:<35} p = {p:.4f} [{_sig_text(p)}]")
    print("============================================================")


def plot_taskB6(transfer_results, save_dir):
    print("\n========== Task B6: 跨场景迁移性 ==========")
    combos = ["geo_wrist", "geo+theta", "all_fusion"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(combos))
    w = 0.35
    for i, c in enumerate(combos):
        r = transfer_results[c]
        a = _safe_float(r["setting_A_auroc"])
        b = _safe_float(r["setting_B_auroc"])
        bstd = _safe_float(r["setting_B_std"])
        color = _combo_color(c)
        ax.bar(x[i] - w / 2, a, w, color=color, edgecolor="k", label="A->B zero-shot" if i == 0 else None)
        ax.bar(x[i] + w / 2, b, w, color=color, alpha=0.45, hatch="//", edgecolor="k", yerr=bstd, capsize=4,
               label="B in-domain CV" if i == 0 else None)
        if np.isfinite(a) and np.isfinite(b):
            d = b - a
            dc = "#2E8B57" if d >= 0 else "#C0392B"
            ax.text(x[i] + w / 2, b + 0.015, f"Δ{d:+.3f}", ha="center", va="bottom", fontsize=9, color=dc)
    ax.set_xticks(x)
    ax.set_xticklabels(combos)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("Task B6: Cross-Scene Transfer (A->B) vs In-Domain (B)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, save_dir, "task_b6_transfer")
    plt.close(fig)
    for c in combos:
        r = transfer_results[c]
        print(f"{c:<12} A->B={r['setting_A_auroc']:.3f}  B-CV={r['setting_B_auroc']:.3f} ± {r['setting_B_std']:.3f}")
    print("==========================================")


def plot_taskB7(loso_results, save_dir):
    print("\n========== Task B7: LOSO 泛化 ==========")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sids = sorted(
        set(loso_results["geo+theta"]["subject_auc"].keys()) |
        set(loso_results["all_fusion"]["subject_auc"].keys())
    )
    x = np.arange(len(sids))
    for name in ["geo+theta", "all_fusion"]:
        vals = [loso_results[name]["subject_auc"].get(s, np.nan) for s in sids]
        ax.plot(x, vals, marker="o", lw=2, label=f"{name} (mean={loso_results[name]['mean']:.3f}±{loso_results[name]['std']:.3f})",
                color=_combo_color(name))
    ax.axhline(EXPA_RESULTS["geo+theta"]["auroc"], color=_combo_color("geo+theta"), ls="--", alpha=0.6, label="Exp-A geo+theta ref")
    ax.axhline(EXPA_RESULTS["all_fusion"]["auroc"], color=_combo_color("all_fusion"), ls="--", alpha=0.6, label="Exp-A all_fusion ref")
    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("Task B7: Leave-One-Subject-Out AUROC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, save_dir, "task_b7_loso")
    plt.close(fig)

    for name in ["geo+theta", "all_fusion"]:
        print(f"{name}:")
        for sid, auc in loso_results[name]["subject_auc"].items():
            print(f"  {sid}: {auc:.3f}")
        print(f"  mean±std: {loso_results[name]['mean']:.3f} ± {loso_results[name]['std']:.3f}")
    print("========================================")


def plot_taskB8(df_b, save_dir):
    print("\n========== Task B8: 速度/光照分组 ==========")
    target_combo = COMBOS_TRANSFER["geo+theta"]
    speed_levels = ["slow", "normal", "fast"]
    light_levels = ["normal", "low", "side"]

    def _eval_group(df_g: pd.DataFrame, combo_name: str):
        X, y, _ = get_valid_subset(df_g, target_combo)
        if len(y) < 20 or len(np.unique(y)) < 2:
            return None
        return run_cv(X, y, combo_name)

    speed_res = {}
    for s in speed_levels:
        dg = df_b[df_b["speed"] == s]
        if len(dg) == 0:
            print(f"  [speed] 跳过 {s}: 无数据")
            continue
        r = _eval_group(dg, f"geo+theta_speed_{s}")
        if r is None:
            print(f"  [speed] 跳过 {s}: 类别不足或有效样本过少")
            continue
        speed_res[s] = r

    light_res = {}
    for l in light_levels:
        dg = df_b[df_b["lighting"] == l]
        if len(dg) == 0:
            print(f"  [lighting] 跳过 {l}: 无数据")
            continue
        r = _eval_group(dg, f"geo+theta_lighting_{l}")
        if r is None:
            print(f"  [lighting] 跳过 {l}: 类别不足或有效样本过少")
            continue
        light_res[l] = r

    if not speed_res and not light_res:
        print("Task B8 skipped: no valid grouped subsets.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # speed
    ax = axes[0]
    s_keys = list(speed_res.keys())
    s_vals = [_safe_float(speed_res[k]["auroc_mean"]) for k in s_keys]
    s_std = [_safe_float(speed_res[k]["auroc_std"]) for k in s_keys]
    if s_keys:
        ax.bar(s_keys, s_vals, yerr=s_std, capsize=4, color=WONG[6], edgecolor="k")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) geo+theta by speed")
    ax.set_ylabel("AUROC")
    ax.grid(axis="y", alpha=0.25)

    # lighting
    ax = axes[1]
    l_keys = list(light_res.keys())
    l_vals = [_safe_float(light_res[k]["auroc_mean"]) for k in l_keys]
    l_std = [_safe_float(light_res[k]["auroc_std"]) for k in l_keys]
    if l_keys:
        ax.bar(l_keys, l_vals, yerr=l_std, capsize=4, color=WONG[2], edgecolor="k")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) geo+theta by lighting")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    _save_fig(fig, save_dir, "task_b8_grouped")
    plt.close(fig)

    print("速度分组：")
    for k, r in speed_res.items():
        print(f"  {k:<8} {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")
    print("光照分组：")
    for k, r in light_res.items():
        print(f"  {k:<8} {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")
    print("========================================")


# =========================
# 配对分析（Task P5/P6/P7）
# =========================

def _fast_auroc(df: pd.DataFrame, feat: str) -> float:
    """直接 roc_auc_score（不做 CV），确保返回 >= 0.5 方向。"""
    sub = df[["contact_label", feat]].copy()
    sub["contact_label"] = pd.to_numeric(sub["contact_label"], errors="coerce")
    sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
    sub = sub[sub["contact_label"].isin([0, 1])].dropna(subset=[feat])
    if len(sub) < 10 or sub["contact_label"].nunique() < 2:
        return np.nan
    try:
        auc = roc_auc_score(sub["contact_label"].astype(int), sub[feat])
        return float(max(auc, 1.0 - auc))
    except Exception:
        return np.nan


def _zero_shot(df_a: pd.DataFrame, df_b: pd.DataFrame, features: List[str]) -> float:
    """Train on A, test on B — 直接返回 AUROC。"""
    def _prep(df):
        sub = df[["contact_label", *features]].copy()
        sub["contact_label"] = pd.to_numeric(sub["contact_label"], errors="coerce")
        for f in features:
            sub[f] = pd.to_numeric(sub[f], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan)
        return sub[sub["contact_label"].isin([0, 1])].dropna(subset=features)

    a_sub = _prep(df_a)
    b_sub = _prep(df_b)
    if len(a_sub) < 20 or len(b_sub) < 20:
        return np.nan
    if a_sub["contact_label"].nunique() < 2 or b_sub["contact_label"].nunique() < 2:
        return np.nan
    scaler = StandardScaler()
    Xa = scaler.fit_transform(a_sub[list(features)].to_numpy(float))
    Xb = scaler.transform(b_sub[list(features)].to_numpy(float))
    ya = a_sub["contact_label"].to_numpy(int)
    yb = b_sub["contact_label"].to_numpy(int)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xa, ya)
        prob = clf.predict_proba(Xb)[:, 1]
    try:
        return float(roc_auc_score(yb, prob))
    except Exception:
        return np.nan


def _indomain_cv_auroc(df_b: pd.DataFrame, features: List[str]) -> float:
    """B 域内 5-fold CV AUROC 均值。"""
    sub = df_b[["contact_label", *features]].copy()
    sub["contact_label"] = pd.to_numeric(sub["contact_label"], errors="coerce")
    for f in features:
        sub[f] = pd.to_numeric(sub[f], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub[sub["contact_label"].isin([0, 1])].dropna(subset=features)
    if len(sub) < 20 or sub["contact_label"].nunique() < 2:
        return np.nan
    X, y = sub[list(features)].to_numpy(float), sub["contact_label"].to_numpy(int)
    res = run_cv(X, y, combo_name="paired_cv")
    return _safe_float(res["auroc_mean"])


def _build_pairs(df_b: pd.DataFrame, expa_all: Dict[str, pd.DataFrame]) -> List[dict]:
    """按 SUBJECT_PAIRS 配对 Exp-A/B 受试者数据。"""
    pairs = []
    for p in SUBJECT_PAIRS:
        sid_a, sid_b = p["sid_a"], p["sid_b"]
        if sid_a not in expa_all:
            print(f"  [配对] 跳过 {p['label']}：Exp-A {sid_a} 未找到")
            continue
        df_b_subj = df_b[df_b["sid"] == sid_b]
        if len(df_b_subj) == 0:
            print(f"  [配对] 跳过 {p['label']}：Exp-B {sid_b} 无数据")
            continue
        pairs.append({**p, "df_a": expa_all[sid_a], "df_b": df_b_subj.copy()})
    return pairs


# ---------- Fig P5 ----------
def plot_taskP5_kde_paired(pairs: List[dict], save_dir: Path):
    """Within-subject KDE：每格 4 条曲线（A-IDLE/A-CONTACT/B-IDLE/B-CONTACT）。"""
    print("\n========== Task P5: Within-Subject KDE (配对) ==========")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch as MPatch

    C_A_IDLE    = WONG[5]    # blue solid
    C_A_CONTACT = WONG[6]    # vermillion solid
    C_B_IDLE    = "#003f7f"  # dark navy dashed
    C_B_CONTACT = WONG[1]    # orange dashed

    n_subj = len(pairs)
    n_feat = len(KDE_FEATS_PAIRED)
    fig, axes = plt.subplots(n_subj, n_feat, figsize=(3.6 * n_feat, 3.2 * n_subj), squeeze=False)
    fig.suptitle("Task P5 — Within-Subject Feature Distribution Shift (Tap vs Writing)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, pair in enumerate(pairs):
        df_a_p, df_b_p = pair["df_a"], pair["df_b"]
        for col, feat in enumerate(KDE_FEATS_PAIRED):
            ax = axes[row][col]
            auc_a = _fast_auroc(df_a_p, feat)
            auc_b = _fast_auroc(df_b_p, feat)

            def _draw(df, lv, color, ls, fill):
                vals = pd.to_numeric(df.loc[df["contact_label"] == lv, feat], errors="coerce").dropna().values
                vals = vals[np.isfinite(vals)]
                if len(vals) < 10:
                    return
                lo, hi = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
                if hi <= lo:
                    return
                xg = np.linspace(lo, hi, 300)
                try:
                    yg = gaussian_kde(vals, bw_method="scott")(xg)
                    ax.plot(xg, yg, color=color, ls=ls, lw=1.6)
                    if fill:
                        ax.fill_between(xg, yg, alpha=0.12, color=color)
                except Exception:
                    pass

            _draw(df_a_p, 0, C_A_IDLE,    "-",  True)
            _draw(df_a_p, 1, C_A_CONTACT, "-",  True)
            _draw(df_b_p, 0, C_B_IDLE,    "--", False)
            _draw(df_b_p, 1, C_B_CONTACT, "--", False)

            ax.set_yticks([])
            ax.set_xlabel(feat, fontsize=8)
            if col == 0:
                ax.set_ylabel(pair["label"], fontsize=10)

            lines = []
            if np.isfinite(auc_a):
                lines.append(f"A:{auc_a:.2f}")
            if np.isfinite(auc_b):
                lines.append(f"B:{auc_b:.2f}")
            if np.isfinite(auc_a) and np.isfinite(auc_b):
                d = auc_b - auc_a
                lines.append(f"Δ:{d:+.2f}")
            ax.set_title("\n".join(lines), fontsize=7.5, pad=2)

    legend_handles = [
        MPatch(facecolor=C_A_IDLE,    label="A  IDLE"),
        MPatch(facecolor=C_A_CONTACT, label="A  CONTACT"),
        MPatch(facecolor=C_B_IDLE,    label="B  IDLE"),
        MPatch(facecolor=C_B_CONTACT, label="B  CONTACT"),
        Line2D([0], [0], color="gray", ls="-",  lw=1.5, label="Exp-A (solid)"),
        Line2D([0], [0], color="gray", ls="--", lw=1.5, label="Exp-B (dashed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save_fig(fig, save_dir, "task_p5_kde_paired")
    plt.close(fig)
    print("=======================================================")


# ---------- Fig P6 ----------
def plot_taskP6_delta_heatmap(pairs: List[dict], save_dir: Path):
    """Δ AUROC 特征热图（features × subjects），3 子图。"""
    print("\n========== Task P6: Δ AUROC 热图 ==========")
    n_subj = len(pairs)
    subject_labels = [p["label"] for p in pairs]

    auroc_a_mat = np.full((len(FEATURE_COLUMNS), n_subj), np.nan)
    auroc_b_mat = np.full((len(FEATURE_COLUMNS), n_subj), np.nan)
    for j, pair in enumerate(pairs):
        for i, feat in enumerate(FEATURE_COLUMNS):
            auroc_a_mat[i, j] = _fast_auroc(pair["df_a"], feat)
            auroc_b_mat[i, j] = _fast_auroc(pair["df_b"], feat)
            print(f"  {pair['label']} | {feat:<22} A={auroc_a_mat[i,j]:.3f}  B={auroc_b_mat[i,j]:.3f}")

    delta_mat = auroc_b_mat - auroc_a_mat
    order = np.argsort(np.nanmean(auroc_a_mat, axis=1))[::-1]
    feat_sorted = [FEATURE_COLUMNS[i] for i in order]
    a_sorted = auroc_a_mat[order]
    b_sorted = auroc_b_mat[order]
    d_sorted = delta_mat[order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    fig.suptitle("Task P6 — Per-Feature AUROC Shift: Tap (A) → Writing (B)",
                 fontsize=13, fontweight="bold")

    def _hm(ax, data, cmap, vmin, vmax, title):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_subj))
        ax.set_xticklabels(subject_labels, fontsize=10)
        ax.set_yticks(range(len(feat_sorted)))
        ax.set_yticklabels(feat_sorted, fontsize=9)
        ax.set_title(title, fontsize=11, pad=8)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                txt = "NaN" if np.isnan(val) else f"{val:+.2f}" if "Δ" in title else f"{val:.2f}"
                norm = 0.5 if np.isnan(val) else (val - vmin) / (vmax - vmin + 1e-9)
                color = "white" if (norm < 0.3 or norm > 0.75) else "black"
                ax.text(c, r, txt, ha="center", va="center", fontsize=7.5, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    _hm(axes[0], d_sorted, "RdYlGn", -0.4, 0.4, "(a) Δ AUROC  (B − A)")
    _hm(axes[1], a_sorted, "Blues",   0.4, 1.0, "(b) Exp-A AUROC")
    _hm(axes[2], b_sorted, "Blues",   0.4, 1.0, "(c) Exp-B AUROC")

    fig.tight_layout()
    _save_fig(fig, save_dir, "task_p6_delta_heatmap")
    plt.close(fig)
    print("===========================================")


# ---------- Fig P7 ----------
def plot_taskP7_transfer_paired(pairs: List[dict], save_dir: Path):
    """Per-subject: zero-shot A→B AUROC vs B 域内 5-fold CV AUROC。"""
    print("\n========== Task P7: 配对受试者迁移对比 ==========")
    combo_names  = list(COMBOS_PAIRED.keys())
    n_combos     = len(combo_names)
    n_subj       = len(pairs)

    fig, axes = plt.subplots(1, n_subj, figsize=(5.5 * n_subj, 5.5), sharey=True, squeeze=False)
    fig.suptitle("Task P7 — Per-Subject Zero-Shot A→B vs In-Domain B 5-fold CV",
                 fontsize=13, fontweight="bold")

    x    = np.arange(n_combos)
    bar_w = 0.36
    C_ZS = WONG[5]
    C_CV = WONG[1]

    for col, pair in enumerate(pairs):
        ax = axes[0][col]
        zs_vals = [_zero_shot(pair["df_a"], pair["df_b"], COMBOS_PAIRED[c]) for c in combo_names]
        cv_vals = [_indomain_cv_auroc(pair["df_b"], COMBOS_PAIRED[c]) for c in combo_names]
        print(f"\n  {pair['label']}:")
        for c, zv, cv in zip(combo_names, zs_vals, cv_vals):
            gap = cv - zv if (np.isfinite(zv) and np.isfinite(cv)) else np.nan
            print(f"    {c:<14} zero-shot={zv:.3f}  cv={cv:.3f}  gap={gap:+.3f}" if np.isfinite(gap) else
                  f"    {c:<14} zero-shot={zv}  cv={cv}")

        b_zs = ax.bar(x - bar_w / 2, zs_vals, bar_w, color=C_ZS, alpha=0.88,
                      label="A→B zero-shot", zorder=3)
        b_cv = ax.bar(x + bar_w / 2, cv_vals, bar_w, color=C_CV, alpha=0.88,
                      label="B in-domain CV", zorder=3)

        for i, (zv, cv) in enumerate(zip(zs_vals, cv_vals)):
            if not (np.isfinite(zv) and np.isfinite(cv)):
                continue
            gap = cv - zv
            y_top = max(zv, cv) + 0.01
            ax.annotate("", xy=(x[i], y_top + 0.06), xytext=(x[i], y_top + 0.01),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
            ax.text(x[i], y_top + 0.065, f"{gap:+.2f}", ha="center", va="bottom",
                    fontsize=8.5, color="#555555", fontweight="bold")

        for bar in list(b_zs) + list(b_cv):
            h = bar.get_height()
            if np.isfinite(h) and h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(combo_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0.4, 1.2)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_title(pair["label"], fontsize=11)
        ax.set_ylabel("AUROC" if col == 0 else "", fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        if col == 0:
            ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    _save_fig(fig, save_dir, "task_p7_transfer_paired")
    plt.close(fig)
    print("==================================================")


# =========================
# 主流程
# =========================
def main():
    EXPA_CSV = Path("experiments/data/exp_a1_s01.csv")
    EXPB_DIR = Path("experiments/data_b/")
    SAVE_DIR = Path("experiments/data_b/figures/")

    df_a = load_expa(str(EXPA_CSV))
    df_b = load_expb(str(EXPB_DIR))

    # Task B1
    try:
        plot_taskB1(df_b, df_a, SAVE_DIR)
    except Exception as e:
        print(f"[Task B1] failed: {e}")

    # Task B2
    try:
        plot_taskB2(df_b, df_a, SAVE_DIR)
    except Exception as e:
        print(f"[Task B2] failed: {e}")

    # Task B3
    try:
        plot_taskB3(df_b, df_a, SAVE_DIR)
    except Exception as e:
        print(f"[Task B3] failed: {e}")

    # Task B4
    try:
        plot_taskB4(df_b, df_a, SAVE_DIR)
    except Exception as e:
        print(f"[Task B4] failed: {e}")

    # Task B5
    try:
        results_b: Dict[str, dict] = {}
        # transfer combos
        for name, feats in COMBOS_TRANSFER.items():
            X, y, ids = get_valid_subset(df_b, feats)
            res = run_cv(X, y, combo_name=name)
            res["sample_ids"] = ids
            results_b[name] = res
        # writing combos
        for name, feats in COMBOS_WRITING.items():
            X, y, ids = get_valid_subset(df_b, feats)
            res = run_cv(X, y, combo_name=name)
            res["sample_ids"] = ids
            results_b[name] = res

        # DeLong tests on Exp-B
        delong_results = {}
        pairs = [
            ("(geo+theta vs geo_wrist)", "geo+theta", "geo_wrist"),
            ("(geo+theta vs geo+theta+vt)", "geo+theta", "geo+theta+vt"),
            ("(geo+theta vs all_fusion)", "geo+theta", "all_fusion"),
        ]
        for key, a, b in pairs:
            y1, s1, y2, s2 = _align_oof_for_delong(results_b[a], results_b[b])
            p = delong_test(y1, s1, y2, s2)
            delong_results[key] = p

        plot_taskB5(results_b, EXPA_RESULTS, delong_results, SAVE_DIR, df_b)
    except Exception as e:
        print(f"[Task B5] failed: {e}")

    # Task B6
    try:
        transfer_results = {}
        for name in ["geo_wrist", "geo+theta", "all_fusion"]:
            transfer_results[name] = cross_scene_eval(df_a, df_b, COMBOS_TRANSFER[name], name)
        plot_taskB6(transfer_results, SAVE_DIR)
    except Exception as e:
        print(f"[Task B6] failed: {e}")

    # Task B7
    try:
        n_sid = df_b["sid"].nunique()
        if n_sid < 3:
            print(f"Task B7 skipped: insufficient subjects (need >= 3, got {n_sid})")
        else:
            loso_results = {
                "geo+theta": loso_cv(df_b, COMBOS_TRANSFER["geo+theta"], "geo+theta"),
                "all_fusion": loso_cv(df_b, COMBOS_TRANSFER["all_fusion"], "all_fusion"),
            }
            plot_taskB7(loso_results, SAVE_DIR)
    except Exception as e:
        print(f"[Task B7] failed: {e}")

    # Task B8
    try:
        plot_taskB8(df_b, SAVE_DIR)
    except Exception as e:
        print(f"[Task B8] failed: {e}")

    # ---- 配对预实验分析 P5 / P6 / P7 ----
    try:
        print("\n====== 加载全部 Exp-A 受试者数据 ======")
        expa_all = load_expa_all(str(Path("experiments/data/")))
        pairs = _build_pairs(df_b, expa_all)
        if not pairs:
            print("[配对分析] 未找到有效配对，跳过 P5/P6/P7。")
        else:
            try:
                plot_taskP5_kde_paired(pairs, SAVE_DIR)
            except Exception as e:
                print(f"[Task P5] failed: {e}")
            try:
                plot_taskP6_delta_heatmap(pairs, SAVE_DIR)
            except Exception as e:
                print(f"[Task P6] failed: {e}")
            try:
                plot_taskP7_transfer_paired(pairs, SAVE_DIR)
            except Exception as e:
                print(f"[Task P7] failed: {e}")
    except Exception as e:
        print(f"[配对分析] 加载失败: {e}")


if __name__ == "__main__":
    main()
