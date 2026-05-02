"""
Exp-A: 综合可视化分析（顶会发表标准）
七个任务：数据概况、单特征判别力、KDE分布、时序对齐、消融ROC、错误分析、
         填0 vs NaN处理方式的AUROC对比验证
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.special import betaln
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ── Wong 8-color colorblind-friendly palette ──────────────────────────────────
WONG = ['#000000', '#E69F00', '#56B4E9', '#009E73',
        '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
C_IDLE    = WONG[2]   # blue
C_CONTACT = WONG[6]   # vermillion
C_NAN     = WONG[4]   # yellow
C_VALID   = WONG[3]   # green

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_a1_s01.csv')
FIG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'figures')

FEATURE_COLS = [
    'dist_raw', 'dist_local', 'v_n', 'a_n', 'sigma_d', 'v_t',
    'approach_theta', 'shadow_score', 'flow_mag', 'brightness_contact',
    'dist2d_palm_0', 'dist2d_palm_5', 'dist2d_palm_9',
    'dist2d_palm_13', 'dist2d_palm_17', 'hull_overlap_ratio',
]

FEATURE_LABELS = {
    'dist_raw': 'dist_raw',
    'dist_local': 'dist_local',
    'v_n': 'v_n',
    'a_n': 'a_n',
    'sigma_d': 'σ_d',
    'v_t': 'v_t',
    'approach_theta': 'approach_θ',
    'shadow_score': 'shadow_score',
    'flow_mag': 'flow_mag',
    'brightness_contact': 'brightness',
    'dist2d_palm_0': 'dist2d_palm₀',
    'dist2d_palm_5': 'dist2d_palm₅',
    'dist2d_palm_9': 'dist2d_palm₉',
    'dist2d_palm_13': 'dist2d_palm₁₃',
    'dist2d_palm_17': 'dist2d_palm₁₇',
    'hull_overlap_ratio': 'hull_overlap',
}

RANDOM_STATE = 42
N_FOLDS      = 5


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def save_figure(fig, name, description=''):
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf_path = os.path.join(FIG_DIR, f'{name}.pdf')
    png_path = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, bbox_inches='tight', format='png', dpi=300)
    print(f"  [saved] {png_path}")
    if description:
        print(f"          {description}")
    plt.close(fig)


def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    return df


def compute_auroc_cv(X_col, y, n_splits=5):
    """Per-feature AUROC via StratifiedKFold CV."""
    valid = ~np.isnan(X_col)
    if valid.sum() < 20:
        return np.nan, np.nan
    X_v = X_col[valid].reshape(-1, 1)
    y_v = y[valid]
    if len(np.unique(y_v)) < 2:
        return np.nan, np.nan
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X_v, y_v):
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_v[tr], y_v[tr])
        prob = lr.predict_proba(X_v[te])[:, 1]
        fpr, tpr, _ = roc_curve(y_v[te], prob)
        aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs)), float(np.std(aucs))


def cohen_d(g1, g2):
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    pooled = np.sqrt((np.var(g1, ddof=1) * (len(g1) - 1) +
                      np.var(g2, ddof=1) * (len(g2) - 1)) /
                     (len(g1) + len(g2) - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled if pooled > 1e-12 else np.nan


def rank_biserial_r(g1, g2):
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    stat, _ = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return 1 - 2 * stat / (len(g1) * len(g2))


def bhattacharyya_overlap(g1, g2, n_pts=500):
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
    if len(g1) < 5 or len(g2) < 5:
        return np.nan
    lo = min(g1.min(), g2.min())
    hi = max(g1.max(), g2.max())
    if hi <= lo:
        return np.nan
    x = np.linspace(lo, hi, n_pts)
    try:
        kde1 = stats.gaussian_kde(g1)(x)
        kde2 = stats.gaussian_kde(g2)(x)
        kde1 /= kde1.sum()
        kde2 /= kde2.sum()
        bc = float(np.sum(np.sqrt(kde1 * kde2)))
        return float(np.clip(bc, 0, 1))
    except Exception:
        return np.nan


def mannwhitney_pval(g1, g2):
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
    if len(g1) < 3 or len(g2) < 3:
        return 1.0
    _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return float(p)


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ── Task 1: 数据概况图 ────────────────────────────────────────────────────────

def task1_data_overview(df):
    print("\n[Task 1] 数据概况图...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Exp-A Data Overview', fontsize=13, fontweight='bold', y=1.01)

    # (a) contact_label 时间轴
    ax = axes[0, 0]
    add_panel_label(ax, '(a)')
    frames = df['frame_id'].values
    labels = df['contact_label'].values
    ax.fill_between(frames, 0, 1, where=(labels == 0),
                    color=C_IDLE, alpha=0.6, label='IDLE', transform=ax.get_xaxis_transform())
    ax.fill_between(frames, 0, 1, where=(labels == 1),
                    color=C_CONTACT, alpha=0.6, label='CONTACT', transform=ax.get_xaxis_transform())
    ax.set_xlabel('Frame ID')
    ax.set_ylabel('Contact Label')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['IDLE', 'CONTACT'])
    ax.set_xlim(frames[0], frames[-1])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Contact label timeline', fontsize=11)

    # (b) 接触事件持续帧数分布
    ax = axes[0, 1]
    add_panel_label(ax, '(b)')
    contact_runs = []
    in_contact, start = False, 0
    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_contact:
            in_contact, start = True, i
        elif lbl == 0 and in_contact:
            contact_runs.append(i - start)
            in_contact = False
    if in_contact:
        contact_runs.append(len(labels) - start)
    contact_runs = np.array(contact_runs)
    median_dur = np.median(contact_runs)
    ax.hist(contact_runs, bins=20, color=C_CONTACT, edgecolor='white', alpha=0.85)
    ax.axvline(median_dur, color='black', linestyle='--', linewidth=1.5,
               label=f'Median = {median_dur:.0f} fr')
    ax.set_xlabel('Duration (frames)')
    ax.set_ylabel('Count')
    ax.set_title('Contact event duration', fontsize=11)
    ax.legend(fontsize=9)

    # (c) IDLE vs CONTACT 帧数对比
    ax = axes[1, 0]
    add_panel_label(ax, '(c)')
    n_idle    = int((labels == 0).sum())
    n_contact = int((labels == 1).sum())
    total     = n_idle + n_contact
    bars = ax.bar(['IDLE', 'CONTACT'], [n_idle, n_contact],
                  color=[C_IDLE, C_CONTACT], edgecolor='white', width=0.5)
    for bar, n in zip(bars, [n_idle, n_contact]):
        pct = 100 * n / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f'{n}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Frame count')
    ax.set_title('IDLE vs CONTACT frame counts', fontsize=11)
    ax.set_ylim(0, max(n_idle, n_contact) * 1.2)

    # (d) 各特征有效帧比例
    ax = axes[1, 1]
    add_panel_label(ax, '(d)')
    total_frames = len(df)
    valid_ratios = []
    nan_ratios   = []
    feat_names   = []
    for f in FEATURE_COLS:
        n_valid = df[f].notna().sum()
        valid_ratios.append(n_valid / total_frames)
        nan_ratios.append(1 - n_valid / total_frames)
        feat_names.append(FEATURE_LABELS[f])
    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, valid_ratios, color=C_VALID, alpha=0.8, label='Valid')
    ax.barh(y_pos, nan_ratios, left=valid_ratios, color=C_NAN, alpha=0.8, label='NaN')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=8.5)
    ax.set_xlabel('Proportion of frames')
    ax.set_title('Feature valid-frame ratio', fontsize=11)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.legend(loc='lower right', fontsize=9)

    fig.tight_layout(pad=1.5)
    save_figure(fig, 'task1_data_overview',
                'Data overview: timeline, duration, class balance, feature completeness')


# ── Task 2: 单特征判别力综合图 ─────────────────────────────────────────────────

def task2_feature_discriminability(df):
    print("\n[Task 2] 单特征判别力综合图...")
    y = df['contact_label'].values.astype(int)
    idle_mask    = y == 0
    contact_mask = y == 1

    records = []
    for feat in FEATURE_COLS:
        col = df[feat].values.astype(float)
        g_idle    = col[idle_mask]
        g_contact = col[contact_mask]
        auroc_mean, auroc_std = compute_auroc_cv(col, y)
        cd   = cohen_d(g_contact, g_idle)
        rrb  = rank_biserial_r(g_contact, g_idle)
        bhat = bhattacharyya_overlap(g_idle, g_contact)
        pval = mannwhitney_pval(g_idle, g_contact)
        records.append({
            'feat': feat,
            'label': FEATURE_LABELS[feat],
            'auroc': auroc_mean,
            'auroc_std': auroc_std,
            'cohens_d': abs(cd) if not np.isnan(cd) else np.nan,
            'rr': rrb,
            'bhatt': bhat,
            'pval': pval,
        })

    res = pd.DataFrame(records).sort_values('auroc', ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 6))
    fig.suptitle('Single-feature Discriminability Summary', fontsize=13, fontweight='bold')
    y_pos = np.arange(len(res))
    labels_sorted = res['label'].tolist()

    def hbar_plot(ax, values, errs, xlabel, ref_line=None, panel_lbl=''):
        colors = [WONG[3] if not np.isnan(v) and v >= 0.7
                  else WONG[2] if not np.isnan(v) and v >= 0.6
                  else WONG[6] for v in values]
        valid  = ~np.isnan(values)
        bars   = ax.barh(y_pos[valid], values[valid],
                         xerr=errs[valid] if errs is not None else None,
                         color=[colors[i] for i in range(len(colors)) if valid[i]],
                         error_kw={'elinewidth': 1, 'capsize': 2},
                         height=0.65, align='center')
        if ref_line is not None:
            ax.axvline(ref_line, color='gray', linestyle='--', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_sorted, fontsize=8.5)
        ax.set_xlabel(xlabel)
        if panel_lbl:
            add_panel_label(ax, panel_lbl, x=-0.2)

    # AUROC
    ax = axes[0]
    add_panel_label(ax, '(a)')
    aurocs = res['auroc'].values
    auroc_stds = res['auroc_std'].values
    valid = ~np.isnan(aurocs)
    bar_colors_auroc = [WONG[3] if v >= 0.8 else WONG[2] if v >= 0.65 else WONG[6]
                        for v in aurocs]
    ax.barh(y_pos[valid], aurocs[valid], xerr=auroc_stds[valid],
            color=[bar_colors_auroc[i] for i in range(len(bar_colors_auroc)) if valid[i]],
            error_kw={'elinewidth': 1, 'capsize': 2},
            height=0.65, align='center')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, label='chance')
    # Add significance stars
    for i, row in res.iterrows():
        stars = sig_stars(row['pval'])
        color = 'black' if stars != 'ns' else 'gray'
        ax.text(0.02, y_pos[i], stars, va='center', ha='left',
                fontsize=7, color=color, transform=ax.get_yaxis_transform())
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=8.5)
    ax.set_xlabel("AUROC (5-fold CV)")
    ax.set_title("AUROC", fontsize=11)
    ax.set_xlim(0.3, 1.05)
    ax.legend(fontsize=8)

    # Cohen's d
    ax = axes[1]
    add_panel_label(ax, '(b)')
    cd_vals = res['cohens_d'].values
    valid = ~np.isnan(cd_vals)
    cd_colors = [WONG[3] if v >= 0.8 else WONG[2] if v >= 0.5 else WONG[6]
                 for v in cd_vals]
    ax.barh(y_pos[valid], cd_vals[valid],
            color=[cd_colors[i] for i in range(len(cd_colors)) if valid[i]],
            height=0.65, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([''] * len(labels_sorted))
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("|Cohen's d|", fontsize=11)

    # Rank-biserial r
    ax = axes[2]
    add_panel_label(ax, '(c)')
    rr_vals = res['rr'].values
    valid = ~np.isnan(rr_vals)
    rr_colors = [WONG[6] if v < 0 else WONG[3] if abs(v) >= 0.5 else WONG[2]
                 for v in rr_vals]
    ax.barh(y_pos[valid], rr_vals[valid],
            color=[rr_colors[i] for i in range(len(rr_colors)) if valid[i]],
            height=0.65, align='center')
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([''] * len(labels_sorted))
    ax.set_xlabel('Rank-biserial r')
    ax.set_title('Rank-biserial r', fontsize=11)

    # Bhattacharyya overlap
    ax = axes[3]
    add_panel_label(ax, '(d)')
    bhat_vals = res['bhatt'].values
    valid = ~np.isnan(bhat_vals)
    bhat_colors = [WONG[3] if v < 0.4 else WONG[2] if v < 0.7 else WONG[6]
                   for v in bhat_vals]
    ax.barh(y_pos[valid], bhat_vals[valid],
            color=[bhat_colors[i] for i in range(len(bhat_colors)) if valid[i]],
            height=0.65, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([''] * len(labels_sorted))
    ax.set_xlabel('Bhattacharyya overlap')
    ax.set_title('Overlap coeff.', fontsize=11)
    ax.set_xlim(0, 1.1)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=WONG[3], label='Strong'),
        mpatches.Patch(color=WONG[2], label='Moderate'),
        mpatches.Patch(color=WONG[6], label='Weak'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(pad=1.5)
    save_figure(fig, 'task2_feature_discriminability',
                'Single-feature discriminability: AUROC, Cohen\'s d, rank-biserial r, Bhattacharyya overlap')
    return res


# ── Task 3: 分布 KDE 图 ────────────────────────────────────────────────────────

def task3_kde_top6(df, feat_ranking):
    print("\n[Task 3] KDE 分布图（AUROC Top 6）...")
    y = df['contact_label'].values.astype(int)
    top6 = feat_ranking.dropna(subset=['auroc']).head(6)['feat'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle('Feature Distributions: IDLE vs CONTACT (AUROC Top 6)',
                 fontsize=13, fontweight='bold')
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for ax, feat, plbl in zip(axes.flat, top6, panel_labels):
        col = df[feat].values.astype(float)
        g_idle    = col[y == 0]
        g_contact = col[y == 1]
        g_idle    = g_idle[~np.isnan(g_idle)]
        g_contact = g_contact[~np.isnan(g_contact)]

        auroc_val = feat_ranking.loc[feat_ranking['feat'] == feat, 'auroc'].values
        auroc_val = float(auroc_val[0]) if len(auroc_val) else np.nan
        cd_val    = feat_ranking.loc[feat_ranking['feat'] == feat, 'cohens_d'].values
        cd_val    = float(cd_val[0]) if len(cd_val) else np.nan

        if len(g_idle) < 5 or len(g_contact) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        lo = min(g_idle.min(), g_contact.min())
        hi = max(g_idle.max(), g_contact.max())
        x  = np.linspace(lo, hi, 400)
        try:
            kde_idle    = stats.gaussian_kde(g_idle)(x)
            kde_contact = stats.gaussian_kde(g_contact)(x)
            overlap     = np.minimum(kde_idle, kde_contact)
            ax.fill_between(x, overlap, alpha=0.4, color='#999999', label='Overlap')
            ax.plot(x, kde_idle,    color=C_IDLE,    linewidth=2, label='IDLE')
            ax.plot(x, kde_contact, color=C_CONTACT, linewidth=2, label='CONTACT')
            ax.fill_between(x, kde_idle,    alpha=0.15, color=C_IDLE)
            ax.fill_between(x, kde_contact, alpha=0.15, color=C_CONTACT)
        except Exception:
            ax.text(0.5, 0.5, 'KDE failed', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        add_panel_label(ax, plbl)
        ax.set_title(FEATURE_LABELS[feat], fontsize=11)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel('Density')
        info = f"AUROC={auroc_val:.3f}\n|d|={cd_val:.2f}" if not np.isnan(cd_val) else f"AUROC={auroc_val:.3f}"
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax.legend(fontsize=8, loc='upper left')

    fig.text(0.5, -0.01, 'Shaded region = overlap area (KDE). Error bars not shown for KDE plots.',
             ha='center', fontsize=9, style='italic')
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'task3_kde_top6',
                'KDE distributions for top-6 AUROC features, overlap filled in grey')


# ── Task 4: 时序对齐图 ────────────────────────────────────────────────────────

def task4_temporal_alignment(df):
    print("\n[Task 4] 时序对齐图（onset ± 20 帧）...")
    TARGET_FEATS = [
        'dist2d_palm_0', 'approach_theta', 'brightness_contact',
        'shadow_score', 'flow_mag', 'v_t',
    ]
    WINDOW = 20
    labels = df['contact_label'].values.astype(int)

    # Detect onset frames (0→1 transitions)
    onsets = []
    for i in range(1, len(labels)):
        if labels[i] == 1 and labels[i - 1] == 0:
            onsets.append(i)

    t_axis = np.arange(-WINDOW, WINDOW + 1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle('Temporal Alignment at Contact Onset (t = 0)',
                 fontsize=13, fontweight='bold')
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for ax, feat, plbl in zip(axes.flat, TARGET_FEATS, panel_labels):
        col = df[feat].values.astype(float)
        segments = []
        for onset in onsets:
            s = onset - WINDOW
            e = onset + WINDOW + 1
            if s < 0 or e > len(col):
                continue
            seg = col[s:e]
            segments.append(seg)

        if not segments:
            ax.text(0.5, 0.5, 'No onset found', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        mat   = np.array(segments, dtype=float)
        mean_ = np.nanmean(mat, axis=0)
        sem_  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0).clip(1))

        ax.plot(t_axis, mean_, color=WONG[5], linewidth=2)
        ax.fill_between(t_axis, mean_ - sem_, mean_ + sem_,
                        alpha=0.25, color=WONG[5])
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Onset (t=0)')
        add_panel_label(ax, plbl)
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlabel('Frame relative to onset')
        ax.set_ylabel(feat, fontsize=9)
        ax.legend(fontsize=8)

    fig.text(0.5, -0.01, f'N={len(segments)} onset events aligned. Shading = ±1 SEM.',
             ha='center', fontsize=9, style='italic')
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'task4_temporal_alignment',
                'Temporal alignment at contact onset ±20 frames, mean ± SEM shaded')


# ── Task 5: 特征组合消融 ROC ──────────────────────────────────────────────────

COMBOS = {
    'baseline':     ['dist_raw'],
    'geo_wrist':    ['dist2d_palm_0'],
    'geo_5pt':      ['dist2d_palm_0', 'dist2d_palm_5', 'dist2d_palm_9',
                     'dist2d_palm_13', 'dist2d_palm_17'],
    'kinematic':    ['dist_raw', 'v_n', 'sigma_d'],
    'geo+theta':    ['dist2d_palm_0', 'approach_theta'],
    'geo+optical':  ['dist2d_palm_0', 'shadow_score', 'flow_mag', 'brightness_contact'],
    'all_fusion':   FEATURE_COLS,
}

COMBO_COLORS = {k: c for k, c in zip(COMBOS.keys(),
                ['#000000', WONG[1], WONG[2], WONG[3], WONG[5], WONG[6], WONG[7]])}


def run_cv_combo(df, feats, n_splits=N_FOLDS):
    """Run StratifiedKFold LR for a feature combo, return ROC and PR curves."""
    sub = df[feats + ['contact_label']].dropna(subset=feats)
    X = sub[feats].values.astype(float)
    y = sub['contact_label'].values.astype(int)
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, 300)
    tprs, roc_aucs, pr_aucs = [], [], []
    all_prec, all_rec = [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                class_weight='balanced')
        lr.fit(Xtr, y[tr])
        prob = lr.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y[te], prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_aucs.append(auc(fpr, tpr))
        prec, rec, _ = precision_recall_curve(y[te], prob)
        pr_aucs.append(auc(rec, prec))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr':  std_tpr,
        'auroc_mean': np.mean(roc_aucs),
        'auroc_std':  np.std(roc_aucs),
        'prauc_mean': np.mean(pr_aucs),
        'prauc_std':  np.std(pr_aucs),
    }


def task5_ablation_roc(df):
    print("\n[Task 5] 特征组合消融 ROC...")
    results = {}
    for name, feats in COMBOS.items():
        res = run_cv_combo(df, feats)
        if res is not None:
            results[name] = res
            print(f"  {name:15s}  AUROC={res['auroc_mean']:.3f}±{res['auroc_std']:.3f}  "
                  f"PR-AUC={res['prauc_mean']:.3f}±{res['prauc_std']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Feature Combination Ablation', fontsize=13, fontweight='bold')

    # ROC curves
    ax = axes[0]
    add_panel_label(ax, '(a)')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Chance')
    for name, res in results.items():
        c = COMBO_COLORS[name]
        ax.plot(res['mean_fpr'], res['mean_tpr'],
                color=c, linewidth=1.8,
                label=f"{name} ({res['auroc_mean']:.3f})")
        ax.fill_between(res['mean_fpr'],
                         res['mean_tpr'] - res['std_tpr'],
                         res['mean_tpr'] + res['std_tpr'],
                         alpha=0.12, color=c)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (5-fold CV)', fontsize=11)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    # AUROC bar chart
    ax = axes[1]
    add_panel_label(ax, '(b)')
    names  = list(results.keys())
    aurocs = [results[n]['auroc_mean'] for n in names]
    auroc_stds = [results[n]['auroc_std'] for n in names]
    colors = [COMBO_COLORS[n] for n in names]
    bars   = ax.bar(names, aurocs, yerr=auroc_stds,
                    color=colors, edgecolor='white',
                    error_kw={'elinewidth': 1.2, 'capsize': 3})
    ax.axhline(0.85, color='gray', linestyle=':', linewidth=1.2, label='0.85 threshold')
    for bar, v, e in zip(bars, aurocs, auroc_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('AUROC')
    ax.set_title('AUROC (5-fold CV ± std)', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)

    # PR-AUC bar chart
    ax = axes[2]
    add_panel_label(ax, '(c)')
    praucs      = [results[n]['prauc_mean'] for n in names]
    prauc_stds  = [results[n]['prauc_std'] for n in names]
    bars        = ax.bar(names, praucs, yerr=prauc_stds,
                         color=colors, edgecolor='white',
                         error_kw={'elinewidth': 1.2, 'capsize': 3})
    ax.axhline(0.85, color='gray', linestyle=':', linewidth=1.2, label='0.85 threshold')
    for bar, v, e in zip(bars, praucs, prauc_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('PR-AUC')
    ax.set_title('PR-AUC (5-fold CV ± std)', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)

    fig.text(0.5, -0.03,
             'Error bars = ±1 std across folds. Dashed line = 0.85 decision threshold.',
             ha='center', fontsize=9, style='italic')
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'task5_ablation_roc',
                'Ablation study: ROC curves and AUROC/PR-AUC comparison across feature combos')
    return results


# ── Task 6: 错误分析 violin plot ─────────────────────────────────────────────

def task6_error_analysis(df, ablation_results):
    print("\n[Task 6] 错误分析 violin plot...")
    best_combo_name = 'all_fusion'
    best_feats      = COMBOS[best_combo_name]

    sub = df[best_feats + ['contact_label']].dropna(subset=best_feats)
    X   = sub[best_feats].values.astype(float)
    y   = sub['contact_label'].values.astype(int)

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds  = np.full(len(y), np.nan)

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr    = scaler.fit_transform(X[tr])
        Xte    = scaler.transform(X[te])
        lr     = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                    class_weight='balanced')
        lr.fit(Xtr, y[tr])
        preds[te] = lr.predict(Xte)

    idx_correct = np.where(preds == y)[0]
    idx_error   = np.where(preds != y)[0]
    idx_fp      = np.where((preds == 1) & (y == 0))[0]
    idx_fn      = np.where((preds == 0) & (y == 1))[0]

    print(f"  Correct: {len(idx_correct)}  Error: {len(idx_error)}  "
          f"FP: {len(idx_fp)}  FN: {len(idx_fn)}")

    ANALYSIS_FEATS = ['dist2d_palm_0', 'approach_theta', 'brightness_contact']
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    fig.suptitle(
        f'Error Analysis: {best_combo_name} — Correct vs Error Frames (FP+FN)',
        fontsize=13, fontweight='bold'
    )
    panel_labels = ['(a)', '(b)', '(c)']

    for ax, feat, plbl in zip(axes, ANALYSIS_FEATS, panel_labels):
        feat_idx = best_feats.index(feat)
        val_correct = X[idx_correct, feat_idx]
        val_fp      = X[idx_fp,      feat_idx]
        val_fn      = X[idx_fn,      feat_idx]

        groups   = []
        grp_vals = []
        for grp_label, vals in [('Correct', val_correct),
                                  ('FP', val_fp),
                                  ('FN', val_fn)]:
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                groups.append(grp_label)
                grp_vals.append(vals)

        data_for_violin = pd.DataFrame({
            'value': np.concatenate(grp_vals),
            'group': np.concatenate([[g] * len(v) for g, v in zip(groups, grp_vals)])
        })

        group_order = [g for g in ['Correct', 'FP', 'FN'] if g in groups]
        palette = {'Correct': WONG[3], 'FP': WONG[6], 'FN': WONG[1]}

        sns.violinplot(data=data_for_violin, x='group', y='value',
                       order=group_order, palette=palette,
                       inner='box', ax=ax, cut=0, linewidth=1)
        add_panel_label(ax, plbl)
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel(feat, fontsize=9)

        # Annotate n counts
        for xi, grp in enumerate(group_order):
            n = len(data_for_violin[data_for_violin['group'] == grp])
            ymax = data_for_violin[data_for_violin['group'] == grp]['value'].max()
            ax.text(xi, ymax, f'n={n}', ha='center', va='bottom', fontsize=8)

    fig.text(0.5, -0.03,
             'FP = false positives (predicted CONTACT, true IDLE). '
             'FN = false negatives (predicted IDLE, true CONTACT). '
             'Box = IQR, whisker = 1.5×IQR.',
             ha='center', fontsize=9, style='italic', wrap=True)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'task6_error_analysis',
                'Error analysis violin plots: correct vs FP vs FN frames on key features')


# ── Task 7: 填0 vs NaN 处理方式 AUROC 对比验证 ───────────────────────────────

APPEARANCE_FEATS = ['shadow_score', 'flow_mag', 'brightness_contact']

# 检测丢失的代理：write_lm 未检测到时，lm_8_x/y 均为 0（与 triple-zero、dist2d NaN 完全一致）
def _detect_lost_mask(df):
    return (df['lm_8_x'] == 0) & (df['lm_8_y'] == 0)


def _auroc_single_feat(col_values, y, n_splits=N_FOLDS):
    """Per-feature AUROC，返回 (mean, std, n_valid)；col可含NaN，自动过滤。"""
    valid = ~np.isnan(col_values)
    n_valid = int(valid.sum())
    if n_valid < 20 or len(np.unique(y[valid])) < 2:
        return np.nan, np.nan, n_valid
    X_v = col_values[valid].reshape(-1, 1)
    y_v = y[valid]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X_v, y_v):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_v[tr])
        Xte = scaler.transform(X_v[te])
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(Xtr, y_v[tr])
        prob = lr.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_v[te], prob)
        aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs)), float(np.std(aucs)), n_valid


def task7_zero_vs_nan(df):
    print("\n[Task 7] 填0 vs NaN 处理方式 AUROC 对比验证...")

    y_all  = df['contact_label'].values.astype(int)
    lost   = _detect_lost_mask(df).values          # True = 检测丢失
    n_lost = int(lost.sum())
    n_valid_frames = int((~lost).sum())

    print(f"  检测丢失帧: {n_lost}  有效帧: {n_valid_frames}  共: {len(df)}")

    records = []
    roc_curves = {}   # feat -> {zero: {...}, nan_: {...}}

    for feat in APPEARANCE_FEATS:
        col_raw = df[feat].values.astype(float)

        # ── 方式1：填0（当前做法，直接使用原始列）──────────────────────────
        auroc_zero, std_zero, n_zero = _auroc_single_feat(col_raw, y_all)

        # ── 方式2：填NaN，只在有效帧子集评估 ─────────────────────────────
        col_nan = col_raw.copy()
        col_nan[lost] = np.nan
        # y 无需改变，_auroc_single_feat 内部自动 dropna
        auroc_nan, std_nan, n_nan = _auroc_single_feat(col_nan, y_all)

        delta = auroc_nan - auroc_zero if not (np.isnan(auroc_nan) or np.isnan(auroc_zero)) else np.nan

        print(f"  {feat:25s}  zero={auroc_zero:.3f}±{std_zero:.3f} (n={n_zero})  "
              f"NaN={auroc_nan:.3f}±{std_nan:.3f} (n={n_nan})  Δ={delta:+.3f}")

        records.append({
            'feat': feat,
            'label': FEATURE_LABELS[feat],
            'auroc_zero': auroc_zero, 'std_zero': std_zero, 'n_zero': n_zero,
            'auroc_nan':  auroc_nan,  'std_nan':  std_nan,  'n_nan':  n_nan,
            'delta': delta,
        })

        # 保存完整 ROC 曲线数据（all frames & valid-only）用于绘图
        mean_fpr = np.linspace(0, 1, 300)

        def _fold_roc(col, y):
            valid = ~np.isnan(col)
            Xv, yv = col[valid].reshape(-1, 1), y[valid]
            if len(np.unique(yv)) < 2:
                return None
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            tprs, aucs_ = [], []
            for tr, te in skf.split(Xv, yv):
                sc = StandardScaler()
                lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                lr.fit(sc.fit_transform(Xv[tr]), yv[tr])
                prob = lr.predict_proba(sc.transform(Xv[te]))[:, 1]
                fpr, tpr, _ = roc_curve(yv[te], prob)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs_.append(auc(fpr, tpr))
            return {'mean_tpr': np.mean(tprs, axis=0),
                    'std_tpr':  np.std(tprs,  axis=0),
                    'auroc':    np.mean(aucs_)}

        roc_curves[feat] = {
            'zero': _fold_roc(col_raw, y_all),
            'nan_': _fold_roc(col_nan, y_all),
        }

    res = pd.DataFrame(records)

    # ── 绘图：3列（每特征一列） × 2行（ROC曲线 + AUROC条形） ───────────────
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        'Validation: Zero-fill vs NaN-mask for Appearance Features\n'
        '(shadow_score, flow_mag, brightness_contact)',
        fontsize=13, fontweight='bold'
    )

    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    panel_labels = [['(a)', '(b)', '(c)'],
                    ['(d)', '(e)', '(f)'],
                    ['(g)', '(h)', '(i)']]

    C_ZERO = WONG[2]   # blue  → 填0
    C_NAN  = WONG[6]   # red   → NaN

    for col_i, feat in enumerate(APPEARANCE_FEATS):
        label  = FEATURE_LABELS[feat]
        rec    = res[res['feat'] == feat].iloc[0]
        curves = roc_curves[feat]

        # Row 0: ROC 曲线对比
        ax_roc = fig.add_subplot(gs[0, col_i])
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Chance')
        for key, color, lbl in [('zero', C_ZERO, 'Zero-fill'),
                                  ('nan_', C_NAN,  'NaN-mask')]:
            c = curves[key]
            if c is None:
                continue
            ax_roc.plot(mean_fpr := np.linspace(0, 1, 300),
                        c['mean_tpr'], color=color, linewidth=1.8,
                        label=f"{lbl} ({c['auroc']:.3f})")
            ax_roc.fill_between(np.linspace(0, 1, 300),
                                c['mean_tpr'] - c['std_tpr'],
                                c['mean_tpr'] + c['std_tpr'],
                                alpha=0.18, color=color)
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
        ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
        ax_roc.set_title(label, fontsize=11)
        ax_roc.legend(fontsize=7.5, loc='lower right')
        add_panel_label(ax_roc, panel_labels[0][col_i])

        # Row 1: AUROC 条形对比（双柱，带误差棒）
        ax_bar = fig.add_subplot(gs[1, col_i])
        methods = ['Zero-fill\n(all frames)', 'NaN-mask\n(valid only)']
        aurocs_ = [rec['auroc_zero'], rec['auroc_nan']]
        stds_   = [rec['std_zero'],   rec['std_nan']]
        ns_     = [rec['n_zero'],     rec['n_nan']]
        bars_   = ax_bar.bar(methods, aurocs_, yerr=stds_,
                             color=[C_ZERO, C_NAN], edgecolor='white', width=0.5,
                             error_kw={'elinewidth': 1.2, 'capsize': 4})
        ax_bar.axhline(0.5, color='gray', linestyle='--', linewidth=1)
        for bar_, v_, e_, n_ in zip(bars_, aurocs_, stds_, ns_):
            if np.isnan(v_): continue
            ax_bar.text(bar_.get_x() + bar_.get_width() / 2,
                        v_ + e_ + 0.015,
                        f'{v_:.3f}\n(n={n_})',
                        ha='center', va='bottom', fontsize=8)
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_ylabel('AUROC')
        ax_bar.set_title(f'AUROC comparison', fontsize=10)
        add_panel_label(ax_bar, panel_labels[1][col_i])

        # Δ annotation
        delta_ = rec['delta']
        if not np.isnan(delta_):
            clr = WONG[3] if delta_ > 0 else WONG[6]
            ax_bar.text(0.97, 0.05, f'Δ = {delta_:+.3f}',
                        transform=ax_bar.transAxes, ha='right', va='bottom',
                        fontsize=9, color=clr, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85))

    # Row 2: 汇总 Δ AUROC 条形图（横向，三特征并排）
    ax_sum = fig.add_subplot(gs[2, :])
    add_panel_label(ax_sum, '(g)', x=-0.03)
    feat_labels = [FEATURE_LABELS[f] for f in APPEARANCE_FEATS]
    deltas      = [res[res['feat'] == f]['delta'].values[0] for f in APPEARANCE_FEATS]
    bar_colors  = [WONG[3] if d > 0 else WONG[6] for d in deltas]
    bars_sum    = ax_sum.bar(feat_labels, deltas, color=bar_colors, edgecolor='white', width=0.4)
    ax_sum.axhline(0, color='black', linewidth=0.8)
    ax_sum.set_ylabel('Δ AUROC  (NaN-mask − Zero-fill)')
    ax_sum.set_title(
        'AUROC gain from correcting detection-lost frames: NaN-mask vs Zero-fill',
        fontsize=11
    )
    for bar_, d_ in zip(bars_sum, deltas):
        if np.isnan(d_): continue
        va = 'bottom' if d_ >= 0 else 'top'
        offset = 0.003 if d_ >= 0 else -0.003
        ax_sum.text(bar_.get_x() + bar_.get_width() / 2,
                    d_ + offset, f'{d_:+.3f}',
                    ha='center', va=va, fontsize=10, fontweight='bold')

    fig.text(0.5, -0.02,
             f'Detection-lost proxy: lm_8_x = lm_8_y = 0 (n={n_lost} frames, {100*n_lost/len(df):.1f}% of total). '
             f'Zero-fill evaluates all {len(df)} frames; NaN-mask evaluates only {n_valid_frames} valid frames. '
             f'5-fold StratifiedKFold, random_state=42.',
             ha='center', fontsize=8.5, style='italic')

    save_figure(fig, 'task7_zero_vs_nan',
                'Validation: AUROC comparison between zero-fill and NaN-mask for appearance features')
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Exp-A Visualization Analysis  (publication quality)")
    print(f"  Data : {os.path.abspath(DATA_PATH)}")
    print(f"  Figs : {os.path.abspath(FIG_DIR)}")
    print("=" * 60)

    df = load_data()
    print(f"\n  Loaded {len(df)} frames | "
          f"IDLE={int((df['contact_label']==0).sum())}  "
          f"CONTACT={int((df['contact_label']==1).sum())}")

    task1_data_overview(df)
    feat_ranking = task2_feature_discriminability(df)
    task3_kde_top6(df, feat_ranking)
    task4_temporal_alignment(df)
    ablation_results = task5_ablation_roc(df)
    task6_error_analysis(df, ablation_results)
    task7_zero_vs_nan(df)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {os.path.abspath(FIG_DIR)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
