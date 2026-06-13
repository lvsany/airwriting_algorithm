"""
Pre-study 分析脚本

输出三个 RQ 的指标数据（AUROC、F1、DeLong p值），不含绘图逻辑。

用法:
  python -m datasets.prestudy.analyze
  python -m datasets.prestudy.analyze --tap-dir datasets/prestudy/data/tap --write-dir datasets/prestudy/data/write

输出:
  终端打印表格 + datasets/prestudy/data/prestudy_results.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# ── 特征集定义 ────────────────────────────────────────────────────────────────
ALL_FEATURES = [
    "dist_raw", "dist_local",
    "v_n", "a_n", "sigma_d", "v_t", "approach_theta",
    "shadow_score", "flow_mag", "brightness_contact",
    "dist2d_palm_0", "dist2d_palm_5", "dist2d_palm_9", "dist2d_palm_13", "dist2d_palm_17",
    "hull_overlap_ratio",
]

FEATURE_SETS = {
    # ── 单特征 ──────────────────────────────────────────────
    "dist_raw":           ["dist_raw"],
    "dist_local":         ["dist_local"],
    "v_n":                ["v_n"],
    "a_n":                ["a_n"],
    "sigma_d":            ["sigma_d"],
    "v_t":                ["v_t"],
    "approach_theta":     ["approach_theta"],
    "shadow_score":       ["shadow_score"],        # 外观：接触阴影（Laplacian 方差）
    "flow_mag":           ["flow_mag"],            # 外观：光流幅值
    "brightness_contact": ["brightness_contact"],  # 外观：接触区亮度
    "dist2d_palm_0":      ["dist2d_palm_0"],       # geo_wrist
    "hull_overlap_ratio": ["hull_overlap_ratio"],
    # ── 组合 ────────────────────────────────────────────────
    "kinematic":          ["v_n", "a_n", "sigma_d", "v_t"],
    "appearance":         ["shadow_score", "flow_mag", "brightness_contact"],
    "geo_wrist":          ["dist2d_palm_0"],
    "geo+theta":          ["dist2d_palm_0", "approach_theta"],
    "geo+theta+vt":       ["dist2d_palm_0", "approach_theta", "v_t"],
    "geo+appearance":     ["dist2d_palm_0", "shadow_score", "flow_mag", "brightness_contact"],
    "geo+theta+appear":   ["dist2d_palm_0", "approach_theta",
                           "shadow_score", "flow_mag", "brightness_contact"],
    "all_fusion":         ALL_FEATURES,
}

CV_FOLDS = 5
RANDOM_STATE = 42


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_csvs(directory: str) -> pd.DataFrame:
    """读取目录下所有 prestudy_*.csv，合并返回。"""
    dfs = []
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return pd.DataFrame()
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(directory, fname)
        try:
            df = pd.read_csv(path)
            df["_source"] = fname
            dfs.append(df)
        except Exception as e:
            print(f"  [warn] 无法读取 {fname}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def prepare_xy(df: pd.DataFrame, feature_cols: list):
    """
    从 DataFrame 提取特征矩阵 X 和标签 y。
    - 去掉 contact_label 缺失的行
    - 特征列缺失值用列均值填充
    - 只保留 feature_cols 中实际存在的列
    """
    df = df.copy()
    df = df[df["contact_label"].notna() & (df["contact_label"] != "")]
    df["contact_label"] = df["contact_label"].astype(int)

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [warn] 特征列不存在，跳过: {missing}")

    X = df[available].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())
    y = df["contact_label"].values
    return X.values, y, available


# ── 评估函数 ──────────────────────────────────────────────────────────────────

def _make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced",
                                   max_iter=1000,
                                   random_state=RANDOM_STATE)),
    ])


def cv_auroc(X, y, n_splits=CV_FOLDS):
    """5-fold Stratified CV，返回每折 AUROC 列表及均值/std。"""
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return [], np.nan, np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    all_probs, all_labels = [], []

    for tr_idx, val_idx in skf.split(X, y):
        pipe = _make_pipeline()
        pipe.fit(X[tr_idx], y[tr_idx])
        probs = pipe.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], probs))
        all_probs.extend(probs)
        all_labels.extend(y[val_idx])

    return scores, float(np.mean(scores)), float(np.std(scores))


def cross_domain_auroc(X_train, y_train, X_test, y_test):
    """在训练集上拟合，在测试集上评估 AUROC。"""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return np.nan, None
    pipe = _make_pipeline()
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, probs)), probs


# ── DeLong 检验 ───────────────────────────────────────────────────────────────

def _placement_values(y_true, scores):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    V10 = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
    V01 = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])
    return V10, V01


def delong_compare(y_true, scores_a, scores_b):
    """
    DeLong 1988: 比较两个分类器的 AUROC 差异。
    返回 (auc_a, auc_b, z, p_value)。
    """
    y = np.asarray(y_true)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)

    V10_a, V01_a = _placement_values(y, a)
    V10_b, V01_b = _placement_values(y, b)
    n_pos, n_neg = len(V10_a), len(V01_a)

    auc_a = float(np.mean(V10_a))
    auc_b = float(np.mean(V10_b))

    mat10 = np.vstack([V10_a, V10_b])   # (2, n_pos)
    mat01 = np.vstack([V01_a, V01_b])   # (2, n_neg)

    S10 = np.cov(mat10, ddof=1) / n_pos if n_pos > 1 else np.zeros((2, 2))
    S01 = np.cov(mat01, ddof=1) / n_neg if n_neg > 1 else np.zeros((2, 2))
    cov = S10 + S01

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return auc_a, auc_b, np.nan, np.nan

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return auc_a, auc_b, float(z), p


def _cv_collect_scores(X, y, n_splits=CV_FOLDS):
    """CV 时同时收集 (oof_labels, oof_probs)，用于 DeLong 检验。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_labels = [], []
    for tr, val in skf.split(X, y):
        pipe = _make_pipeline()
        pipe.fit(X[tr], y[tr])
        probs = pipe.predict_proba(X[val])[:, 1]
        all_probs.extend(probs)
        all_labels.extend(y[val])
    return np.array(all_labels), np.array(all_probs)


# ── 格式化输出 ────────────────────────────────────────────────────────────────

def _fmt(v, precision=3):
    return f"{v:.{precision}f}" if not np.isnan(v) else "N/A"


def _print_table(title, rows, headers):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    header_str = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_str)
    print("  ".join("─" * w for w in col_w))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))


# ── 主分析逻辑 ────────────────────────────────────────────────────────────────

def run_analysis(tap_dir: str, write_dir: str, out_path: str):
    print("\n" + "=" * 60)
    print("  Pre-study Analysis")
    print("=" * 60)

    tap_df   = load_csvs(tap_dir)
    write_df = load_csvs(write_dir)

    def _data_info(df, name):
        if df.empty:
            print(f"  [{name}] 未找到数据")
        else:
            n1 = int((df["contact_label"].astype(str) == "1").sum())
            n0 = int((df["contact_label"].astype(str) == "0").sum())
            print(f"  [{name}] {len(df)} 帧  contact=1:{n1}  contact=0:{n0}  "
                  f"来源: {df['_source'].nunique()} 个文件")

    _data_info(tap_df,   "TAP  ")
    _data_info(write_df, "WRITE")

    results = {}

    # ── RQ1 & RQ2: Tap 数据消融 ──────────────────────────────────────────────
    rq1_sets  = ["dist_raw", "kinematic",
                 "shadow_score", "flow_mag", "brightness_contact", "appearance"]
    rq2_sets  = ["geo_wrist", "approach_theta",
                 "geo+theta", "geo+theta+vt",
                 "geo+appearance", "geo+theta+appear",
                 "all_fusion"]

    tap_rows = []
    tap_auroc_cache = {}  # fname → (labels, probs) for DeLong

    if not tap_df.empty:
        for name in rq1_sets + rq2_sets:
            feats = FEATURE_SETS[name]
            X, y, used = prepare_xy(tap_df, feats)
            if len(used) == 0 or len(np.unique(y)) < 2:
                tap_rows.append([name, "N/A", "N/A", "N/A"])
                continue
            folds, mean_auc, std_auc = cv_auroc(X, y)
            tap_rows.append([name, _fmt(mean_auc), _fmt(std_auc),
                             f"({', '.join(_fmt(s) for s in folds)})"])
            results[f"tap_{name}_auroc"] = mean_auc
            results[f"tap_{name}_auroc_std"] = std_auc

            # collect oof scores for DeLong
            lbl, prb = _cv_collect_scores(X, y)
            tap_auroc_cache[name] = (lbl, prb)

    _print_table("RQ1 & RQ2 — Tap 场景 (5-fold CV AUROC)",
                 tap_rows,
                 ["feature_set", "AUROC", "±std", "per-fold"])

    # ── RQ3: Write 场景（域内）──────────────────────────────────────────────
    write_rows = []
    write_auroc_cache = {}

    rq3_sets = ["geo+theta", "geo+theta+vt",
                "appearance", "geo+appearance", "geo+theta+appear",
                "all_fusion"]

    if not write_df.empty:
        for name in rq3_sets:
            feats = FEATURE_SETS[name]
            X, y, used = prepare_xy(write_df, feats)
            if len(used) == 0 or len(np.unique(y)) < 2:
                write_rows.append([name, "N/A", "N/A"])
                continue
            folds, mean_auc, std_auc = cv_auroc(X, y)
            write_rows.append([name, _fmt(mean_auc), _fmt(std_auc)])
            results[f"write_{name}_auroc"] = mean_auc
            results[f"write_{name}_auroc_std"] = std_auc

            lbl, prb = _cv_collect_scores(X, y)
            write_auroc_cache[name] = (lbl, prb)

    _print_table("RQ3 — Write 场景域内 (5-fold CV AUROC)",
                 write_rows, ["feature_set", "AUROC", "±std"])

    # ── RQ3: Zero-shot 跨域 (Tap→Write) ─────────────────────────────────────
    cross_rows = []
    if not tap_df.empty and not write_df.empty:
        for name in rq3_sets:
            feats = FEATURE_SETS[name]
            X_tr, y_tr, _ = prepare_xy(tap_df, feats)
            X_te, y_te, _ = prepare_xy(write_df, feats)
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                cross_rows.append([name, "N/A"])
                continue
            auc, _ = cross_domain_auroc(X_tr, y_tr, X_te, y_te)
            cross_rows.append([name, _fmt(auc)])
            results[f"cross_tap2write_{name}_auroc"] = auc

    _print_table("RQ3 — Zero-shot 跨域 Tap→Write AUROC",
                 cross_rows, ["feature_set", "AUROC"])

    # ── DeLong 检验 (Write 域内 OOF scores) ─────────────────────────────────
    delong_pairs = [
        ("geo+theta",       "geo+theta+vt"),
        ("geo+theta",       "geo+appearance"),
        ("geo+theta+vt",    "geo+theta+appear"),
        ("geo+appearance",  "geo+theta+appear"),
        ("geo+theta+appear","all_fusion"),
    ]
    delong_rows = []
    for name_a, name_b in delong_pairs:
        if name_a not in write_auroc_cache or name_b not in write_auroc_cache:
            continue
        lbl_a, prb_a = write_auroc_cache[name_a]
        lbl_b, prb_b = write_auroc_cache[name_b]
        auc_a, auc_b, z, p = delong_compare(lbl_a, prb_a, prb_b)
        delong_rows.append([name_a, name_b, _fmt(auc_a), _fmt(auc_b),
                            _fmt(z, 3), _fmt(p, 4)])
        key = f"delong_{name_a}_vs_{name_b}_write"
        results[key] = {"auc_a": auc_a, "auc_b": auc_b, "z": z, "p": p}

    _print_table("DeLong 检验 (Write 域内 OOF scores)",
                 delong_rows,
                 ["model_A", "model_B", "AUROC_A", "AUROC_B", "z", "p"])

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64, float)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(results), f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {out_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-dir",   default=os.path.join(
        os.path.dirname(__file__), "data", "tap"))
    parser.add_argument("--write-dir", default=os.path.join(
        os.path.dirname(__file__), "data", "write"))
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "data", "prestudy_results.json"))
    args = parser.parse_args()

    run_analysis(
        tap_dir=os.path.abspath(args.tap_dir),
        write_dir=os.path.abspath(args.write_dir),
        out_path=os.path.abspath(args.out),
    )


if __name__ == "__main__":
    main()
