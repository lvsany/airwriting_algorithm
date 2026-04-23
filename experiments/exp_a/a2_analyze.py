"""
Exp-A2: 运动学 + 外观特征统计分析

输入: exp_a1_{subject}.csv
输出:
  - 每个接触事件前后 ±N 帧的特征时序曲线（叠加均值 ± std）
  - 各特征在 contact=0/1 下的箱线图
  - 保存到 data/figures/exp_a2_{subject}_*.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

WINDOW = 15          # 事件前后各 ±15 帧
MIN_IDLE_GAP = 10    # 两个接触事件之间最少间隔帧数（防止粘连事件）

FEATURES_KINEMATIC = ['dist_raw', 'v_n', 'sigma_d', 'v_t']
FEATURES_APPEARANCE = ['shadow_score', 'flow_mag']
ALL_FEATURES = FEATURES_KINEMATIC + FEATURES_APPEARANCE

FEATURE_LABELS = {
    'dist_raw':    'Distance d (mm)',
    'v_n':         'Normal velocity v_n (mm/frame)',
    'sigma_d':     'Distance std σ_d (mm)',
    'v_t':         'Tangential velocity v_t (mm/s)',
    'shadow_score':'Shadow score (Laplacian var)',
    'flow_mag':    'Optical flow magnitude',
}


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['contact_label'] = df['contact_label'].astype(int)
    return df


def _find_contact_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """返回每个连续接触段的 (start, end) 帧索引"""
    events = []
    in_contact = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_contact:
            in_contact = True
            start = i
        elif v == 0 and in_contact:
            in_contact = False
            events.append((start, i - 1))
    if in_contact:
        events.append((start, len(labels) - 1))
    # 过滤极短事件（< 3 帧）
    events = [(s, e) for s, e in events if e - s >= 2]
    return events


def _extract_event_windows(df: pd.DataFrame, events: list) -> dict:
    """
    对每个接触事件，以事件中心帧为基准，提取 [-WINDOW, +WINDOW] 的特征窗口。
    返回 {feature: (N_events, 2*WINDOW+1) 数组}
    """
    n = 2 * WINDOW + 1
    windows = {feat: [] for feat in ALL_FEATURES}

    for s, e in events:
        center = (s + e) // 2
        for feat in ALL_FEATURES:
            if feat not in df.columns:
                continue
            seg = []
            for offset in range(-WINDOW, WINDOW + 1):
                idx = center + offset
                if 0 <= idx < len(df):
                    val = df[feat].iloc[idx]
                    seg.append(float(val) if not np.isnan(val) else np.nan)
                else:
                    seg.append(np.nan)
            windows[feat].append(seg)

    return {k: np.array(v) for k, v in windows.items() if v}


def _plot_timeseries(windows: dict, events: list, subject: str, out_dir: str):
    """图1: 各特征的接触事件时序叠加曲线（均值 ± std）"""
    n_feats = len(ALL_FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    x = np.arange(-WINDOW, WINDOW + 1)

    for ax, feat in zip(axes, ALL_FEATURES):
        arr = windows.get(feat)
        if arr is None or len(arr) == 0:
            ax.set_visible(False)
            continue
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        ax.plot(x, mean, color='steelblue', linewidth=2, label='mean')
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.25, color='steelblue', label='±std')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.2,
                   label='contact onset')
        ax.axvspan(-WINDOW, 0, alpha=0.05, color='orange')
        ax.axvspan(0, WINDOW, alpha=0.05, color='green')
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlabel('Frame offset')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Exp-A2: Feature timeseries around contact events\n'
                 f'Subject: {subject}  |  N events = {len(events)}',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f'exp_a2_{subject}_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [A2] 时序曲线 → {path}")


def _plot_boxplots(df: pd.DataFrame, subject: str, out_dir: str):
    """图2: 各特征在 IDLE / CONTACT 状态下的分布箱线图 + t 检验"""
    df_valid = df.dropna(subset=['dist_raw'])  # 只保留手在范围内的帧
    idle = df_valid[df_valid['contact_label'] == 0]
    contact = df_valid[df_valid['contact_label'] == 1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, feat in zip(axes, ALL_FEATURES):
        if feat not in df_valid.columns:
            ax.set_visible(False)
            continue
        data_idle = idle[feat].dropna().values
        data_contact = contact[feat].dropna().values
        if len(data_idle) < 5 or len(data_contact) < 5:
            ax.set_visible(False)
            continue

        ax.boxplot([data_idle, data_contact],
                   labels=['IDLE', 'CONTACT'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))

        # t 检验
        t_stat, p_val = stats.ttest_ind(data_idle, data_contact,
                                        equal_var=False)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01
              else ('*' if p_val < 0.05 else 'ns'))
        ax.set_title(f"{FEATURE_LABELS.get(feat, feat)}\n"
                     f"t={t_stat:.2f}  p={p_val:.4f}  {sig}", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Exp-A2: Feature distributions IDLE vs CONTACT\n'
                 f'Subject: {subject}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f'exp_a2_{subject}_boxplot.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [A2] 箱线图   → {path}")


def _print_summary(df: pd.DataFrame, events: list):
    df_valid = df.dropna(subset=['dist_raw'])
    idle = df_valid[df_valid['contact_label'] == 0]
    contact = df_valid[df_valid['contact_label'] == 1]

    print(f"\n  {'特征':<18} {'IDLE 均值':>12} {'CONTACT 均值':>14} {'p 值':>10}")
    print(f"  {'-'*58}")
    for feat in ALL_FEATURES:
        if feat not in df_valid.columns:
            continue
        d0 = idle[feat].dropna()
        d1 = contact[feat].dropna()
        if len(d0) < 5 or len(d1) < 5:
            continue
        _, p = stats.ttest_ind(d0, d1, equal_var=False)
        sig = '***' if p < 0.001 else ('**' if p < 0.01
              else ('*' if p < 0.05 else 'ns'))
        print(f"  {feat:<18} {d0.mean():>12.3f} {d1.mean():>14.3f} {p:>8.4f} {sig}")
    print(f"\n  接触事件数: {len(events)}")
    print(f"  总帧数: {len(df)}  "
          f"IDLE: {(df['contact_label']==0).sum()}  "
          f"CONTACT: {(df['contact_label']==1).sum()}")


def run_analyze(subject: str, data_dir: str):
    csv_path = os.path.join(data_dir, f'exp_a1_{subject}.csv')
    if not os.path.exists(csv_path):
        print(f"  [A2] 找不到数据文件: {csv_path}")
        return

    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    df = _load_csv(csv_path)
    events = _find_contact_events(df['contact_label'].values)
    print(f"  [A2] 加载 {len(df)} 帧，检测到 {len(events)} 个接触事件")

    if len(events) == 0:
        print("  [A2] 未检测到接触事件，请检查标注数据")
        return

    windows = _extract_event_windows(df, events)
    _plot_timeseries(windows, events, subject, fig_dir)
    _plot_boxplots(df, subject, fig_dir)
    _print_summary(df, events)
