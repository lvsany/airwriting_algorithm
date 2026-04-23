"""
Exp-A3: 各特征组合 ROC 曲线与 AUC 对比

8 种特征组合（对应 PLAN.md Exp-A3 表格）：
  1. d only                          (baseline)
  2. d + v_n
  3. d + sigma_d
  4. d + v_n + sigma_d               (运动学主方案)
  5. d + shadow
  6. d + flow
  7. d + shadow + flow
  8. d + v_n + sigma_d + shadow + flow  (融合上界)

分类器：逻辑回归（推理快、可解释、适合小样本）
评估：5-fold cross-validation，报告 mean AUC ± std

输出:
  - ROC 曲线对比图 (8条曲线叠加)
  - AUC 柱状图（带误差棒）
  - 决策点判断打印
  - 保存到 data/figures/exp_a3_{subject}_*.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

AUC_THRESHOLD_KINEMATIC = 0.85   # 运动学主方案决策门槛
AUC_IMPROVE_APPEARANCE  = 0.03   # 外观特征显著提升门槛

COMBINATIONS = [
    ('d only (baseline)',              ['dist_raw']),
    ('d + v_n',                        ['dist_raw', 'v_n']),
    ('d + σ_d',                        ['dist_raw', 'sigma_d']),
    ('d + v_n + σ_d  [kinematic]',     ['dist_raw', 'v_n', 'sigma_d']),
    ('d + shadow',                     ['dist_raw', 'shadow_score']),
    ('d + flow',                       ['dist_raw', 'flow_mag']),
    ('d + shadow + flow',              ['dist_raw', 'shadow_score', 'flow_mag']),
    ('d + v_n + σ_d + shadow + flow',  ['dist_raw', 'v_n', 'sigma_d',
                                        'shadow_score', 'flow_mag']),
]

COLORS = [
    '#888888', '#4878CF', '#6ACC65', '#D65F5F',
    '#B47CC7', '#C4AD66', '#77BEDB', '#E05E4B',
]


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ['dist_raw', 'v_n', 'sigma_d', 'v_t', 'shadow_score', 'flow_mag']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['contact_label'] = df['contact_label'].astype(int)
    return df


def _build_pipeline():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
        ('clf',    LogisticRegression(max_iter=500, C=1.0, random_state=42)),
    ])


def _eval_combination(X: np.ndarray, y: np.ndarray, n_splits=5):
    """5-fold CV，返回 mean_fpr, mean_tpr, mean_auc, std_auc"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 200)

    for train, test in cv.split(X, y):
        pipe = _build_pipeline()
        pipe.fit(X[train], y[train])
        proba = pipe.predict_proba(X[test])[:, 1]
        fpr, tpr, _ = roc_curve(y[test], proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    return mean_fpr, mean_tpr, mean_auc, std_auc


def _plot_roc(results: list, subject: str, out_dir: str):
    """ROC 曲线叠加图"""
    fig, ax = plt.subplots(figsize=(9, 7))

    for (name, _), (fpr, tpr, mean_auc, std_auc), color in zip(
            COMBINATIONS, results, COLORS):
        label = f'{name}  (AUC={mean_auc:.3f}±{std_auc:.3f})'
        lw = 2.5 if 'kinematic' in name else 1.5
        ls = '-' if 'kinematic' in name or 'baseline' in name else '--'
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.axhline(y=0.85, color='gray', linestyle=':', lw=1.2,
               label='AUC=0.85 threshold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Exp-A3: ROC Curves — Feature Combination Comparison\n'
                 f'Subject: {subject}  (5-fold CV, Logistic Regression)',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f'exp_a3_{subject}_roc.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [A3] ROC 曲线图 → {path}")


def _plot_auc_bar(results: list, subject: str, out_dir: str):
    """AUC 柱状图（带误差棒）"""
    names = [c[0].replace('  [kinematic]', '*') for c in COMBINATIONS]
    aucs  = [r[2] for r in results]
    stds  = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(names)), aucs, yerr=stds,
                  color=COLORS, alpha=0.8, capsize=5)

    ax.axhline(y=AUC_THRESHOLD_KINEMATIC, color='red', linestyle='--',
               lw=1.5, label=f'kinematic threshold ({AUC_THRESHOLD_KINEMATIC})')
    ax.set_ylim([0.4, 1.05])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('AUC (mean ± std)', fontsize=11)
    ax.set_title(f'Exp-A3: AUC Comparison — {subject}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, v, s in zip(bars, aucs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f'exp_a3_{subject}_auc_bar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [A3] AUC 柱状图 → {path}")


def _decision_report(results: list):
    """打印决策点判断"""
    kinematic_auc = results[3][2]   # d + v_n + σ_d
    baseline_auc  = results[0][2]   # d only
    fusion_auc    = results[7][2]   # 全特征

    print(f"\n  ── 决策点报告 ──────────────────────────────────────")
    print(f"  baseline (d only)          AUC = {baseline_auc:.4f}")
    print(f"  kinematic (d+v_n+σ_d)      AUC = {kinematic_auc:.4f}  "
          f"{'✅ ≥ 0.85，进入阶段二' if kinematic_auc >= AUC_THRESHOLD_KINEMATIC else '❌ < 0.85，重新评估'}")

    appear_improvement = fusion_auc - kinematic_auc
    print(f"  fusion vs kinematic        ΔAUC = {appear_improvement:+.4f}  "
          f"{'✅ 外观特征有显著贡献' if appear_improvement >= AUC_IMPROVE_APPEARANCE else '⚠️  外观特征贡献有限'}")
    print(f"  ────────────────────────────────────────────────────\n")

    print(f"  完整 AUC 表格:")
    print(f"  {'特征组合':<45} {'AUC':>7} {'std':>7}")
    print(f"  {'-'*62}")
    for (name, _), (_, _, mean_auc, std_auc) in zip(COMBINATIONS, results):
        marker = ' ←' if 'kinematic' in name else ''
        print(f"  {name:<45} {mean_auc:>7.4f} {std_auc:>7.4f}{marker}")


def run_roc(subject: str, data_dir: str):
    csv_path = os.path.join(data_dir, f'exp_a1_{subject}.csv')
    if not os.path.exists(csv_path):
        print(f"  [A3] 找不到数据文件: {csv_path}")
        return

    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    df = _load_csv(csv_path)
    # 只保留手在手掌范围内的帧（dist_raw 有值）
    df_valid = df.dropna(subset=['dist_raw']).reset_index(drop=True)
    y = df_valid['contact_label'].values

    print(f"  [A3] 有效帧: {len(df_valid)}  "
          f"IDLE: {(y==0).sum()}  CONTACT: {(y==1).sum()}")

    if (y == 1).sum() < 20:
        print("  [A3] 接触帧过少（<20），无法进行可靠的 ROC 评估")
        return

    results = []
    for name, feats in COMBINATIONS:
        avail = [f for f in feats if f in df_valid.columns]
        if not avail:
            results.append((np.linspace(0,1,200), np.zeros(200), 0.0, 0.0))
            continue
        X = df_valid[avail].values
        fpr, tpr, mean_auc, std_auc = _eval_combination(X, y)
        results.append((fpr, tpr, mean_auc, std_auc))
        print(f"  [A3] {name:<45} AUC={mean_auc:.4f}±{std_auc:.4f}")

    _plot_roc(results, subject, fig_dir)
    _plot_auc_bar(results, subject, fig_dir)
    _decision_report(results)
