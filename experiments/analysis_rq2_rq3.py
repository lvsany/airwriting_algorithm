#!/usr/bin/env python3
"""
RQ2 & RQ3: Feature combination ablation for contact detection.

RQ2 – which features contribute most in controlled tap (Exp-A)?
RQ3 – do those findings transfer to writing (Exp-B)?

Usage:
    python experiments/analysis_rq2_rq3.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── global aesthetics (publication-ready, print-safe) ────────────────────────
plt.rcParams.update({
    'font.family':         'sans-serif',
    'font.sans-serif':     ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':           8,
    'axes.titlesize':      10,
    'axes.labelsize':      9,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     7.5,
    'legend.framealpha':   0.92,
    'legend.edgecolor':    '#CCCCCC',
    'legend.borderpad':    0.35,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.spines.left':    True,
    'axes.spines.bottom':  True,
    'axes.linewidth':      0.8,
    'axes.grid':           False,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'xtick.major.size':    3.0,
    'ytick.major.size':    3.0,
    'axes.axisbelow':      True,
    'figure.dpi':          150,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
})

# ── Wong (2011) colorblind-safe palette ───────────────────────────────────────
WONG = {
    'depth':       '#0072B2',   # blue
    'kinematic':   '#009E73',   # bluish-green
    'directional': '#D55E00',   # vermilion
    'appearance':  '#E69F00',   # orange
    'geometric':   '#CC79A7',   # reddish-purple
    'fusion':      '#56B4E9',   # sky-blue
}

COMBO_CAT = {
    'd_only':        'depth',
    'd+v_n':         'kinematic',
    'd+sigma_d':     'kinematic',
    'd+v_n+sigma_d': 'kinematic',
    'd+a_n':         'kinematic',
    'd+theta':       'directional',
    'd+shadow':      'appearance',
    'd+flow':        'appearance',
    'd+brightness':  'appearance',
    'd+dist2d':      'geometric',
    'd+overlap':     'geometric',
    'fusion':        'fusion',
}

# display order top→bottom (Fusion first = best performers at top)
CAT_GROUPS = [
    ('Fusion',      'fusion',      ['fusion']),
    ('Geometric',   'geometric',   ['d+dist2d', 'd+overlap']),
    ('Appearance',  'appearance',  ['d+flow', 'd+brightness', 'd+shadow']),
    ('Directional', 'directional', ['d+theta']),
    ('Kinematic',   'kinematic',   ['d+v_n+sigma_d', 'd+sigma_d', 'd+v_n', 'd+a_n']),
    ('Depth',       'depth',       ['d_only']),
]

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_A  = os.path.join(BASE, 'data')
DATA_B  = os.path.join(BASE, 'data_b')
FIG_DIR = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── feature definitions ────────────────────────────────────────────────────────
D = ['dist_raw', 'dist_local']

COMBOS = {
    'd_only':        D,
    'd+v_n':         D + ['v_n'],
    'd+sigma_d':     D + ['sigma_d'],
    'd+v_n+sigma_d': D + ['v_n', 'sigma_d'],
    'd+a_n':         D + ['a_n'],
    'd+theta':       D + ['approach_theta'],
    'd+shadow':      D + ['shadow_score'],
    'd+flow':        D + ['flow_mag'],
    'd+brightness':  D + ['brightness_contact'],
    'd+dist2d':      D + ['dist2d_palm_0', 'dist2d_palm_5', 'dist2d_palm_9',
                           'dist2d_palm_13', 'dist2d_palm_17'],
    'd+overlap':     D + ['hull_overlap_ratio'],
    'fusion':        D + ['v_n', 'a_n', 'sigma_d', 'v_t', 'approach_theta',
                           'shadow_score', 'flow_mag', 'brightness_contact',
                           'dist2d_palm_0', 'dist2d_palm_5', 'dist2d_palm_9',
                           'dist2d_palm_13', 'dist2d_palm_17',
                           'hull_overlap_ratio'],
}

CLIP_COLS = ['v_t', 'dist2d_palm_0', 'dist2d_palm_5', 'dist2d_palm_9',
             'dist2d_palm_13', 'dist2d_palm_17']

TEMPORAL_GROUPS = [
    ('Depth',        ['dist_raw', 'dist_local']),
    ('Directional',  ['approach_theta']),
    ('Kinematics',   ['v_n', 'a_n', 'sigma_d', 'v_t']),
    ('Appearance',   ['shadow_score', 'flow_mag', 'brightness_contact']),
    ('Geometric',    ['dist2d_palm_0', 'hull_overlap_ratio']),
]

WINDOW = 15

# ── data ──────────────────────────────────────────────────────────────────────
def _clip(df):
    df = df.copy()
    for col in CLIP_COLS:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)
    return df


def load_exp_a():
    out = {}
    for f in sorted(os.listdir(DATA_A)):
        if f.endswith('.csv'):
            sid = f.replace('exp_a1_', '').replace('.csv', '')
            out[sid] = _clip(pd.read_csv(os.path.join(DATA_A, f)))
    return out


def load_exp_b():
    out = {}
    for f in sorted(os.listdir(DATA_B)):
        if f.endswith('_labeled.csv'):
            sid = f.split('_')[2]
            out[sid] = _clip(pd.read_csv(os.path.join(DATA_B, f)))
    return out


# ── AUC ───────────────────────────────────────────────────────────────────────
def _pipe():
    return Pipeline([('sc', RobustScaler()),
                     ('lr', LogisticRegression(max_iter=1000, random_state=42))])


def cv_auc(df, feats, n_splits=5):
    sub = df[feats + ['contact_label']].dropna()
    if sub['contact_label'].nunique() < 2 or len(sub) < n_splits * 2:
        return np.nan
    X, y = sub[feats].values, sub['contact_label'].values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_score(_pipe(), X, y, cv=cv, scoring='roc_auc').mean()


def ablation(subjects):
    return {c: [cv_auc(df, f) for df in subjects.values()]
            for c, f in COMBOS.items()}


# ── Wilcoxon ──────────────────────────────────────────────────────────────────
KEY_PAIRS = [
    ('d+v_n+sigma_d', 'd+dist2d'),
    ('d+dist2d',      'fusion'),
    ('d_only',        'fusion'),
]


def wilcoxon_tests(res):
    rows = []
    for c1, c2 in KEY_PAIRS:
        d = np.array(res[c1]) - np.array(res[c2])
        d = d[~np.isnan(d)]
        row = dict(pair=f'{c1} vs {c2}', n=len(d),
                   mean_c1=np.nanmean(res[c1]), mean_c2=np.nanmean(res[c2]),
                   mean_diff=d.mean() if len(d) else np.nan,
                   stat=np.nan, p=np.nan)
        if len(d) >= 2:
            try:
                row['stat'], row['p'] = stats.wilcoxon(d)
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


# ── temporal curves (per-time-point aggregation) ──────────────────────────────
# Kinematic features store 0.0 (not NaN) when no hand is detected.
# We use dist_raw.isna() as a proxy to mask those sentinel zeros.
_SENTINEL_ZERO_FEATS = {'v_n', 'a_n', 'sigma_d', 'v_t'}


def _temporal_curves(df):
    """Return per-time-offset mean/std across all contact onsets."""
    df = df.copy()
    no_detect = df['dist_raw'].isna()
    for col in _SENTINEL_ZERO_FEATS:
        if col in df.columns:
            df.loc[no_detect, col] = np.nan

    all_feats = [f for _, fl in TEMPORAL_GROUPS for f in fl]
    label  = df['contact_label'].values
    onsets = np.where((label[:-1] == 0) & (label[1:] == 1))[0] + 1
    t_arr  = np.arange(-WINDOW, WINDOW + 1)

    curves = {}
    for feat in all_feats:
        col_vals = df[feat].values if feat in df.columns else None
        by_t = [[] for _ in t_arr]
        if col_vals is not None:
            for onset in onsets:
                for ti, dt in enumerate(t_arr):
                    idx = onset + dt
                    if 0 <= idx < len(col_vals):
                        v = col_vals[idx]
                        if not (np.isnan(v) or np.isinf(v)):
                            by_t[ti].append(v)
        means = np.array([np.mean(x) if x else np.nan for x in by_t])
        stds  = np.array([np.std(x)  if len(x) > 1 else 0.0 for x in by_t])
        n_min = min((len(x) for x in by_t), default=0)
        curves[feat] = (t_arr, means, stds, n_min)
    return curves


# ── transfer ──────────────────────────────────────────────────────────────────
def transfer_auc(subj_a, subj_b):
    results = {c: [] for c in COMBOS}
    for sid in sorted(set(subj_a) & set(subj_b)):
        for combo, feats in COMBOS.items():
            tr = subj_a[sid][feats + ['contact_label']].dropna()
            te = subj_b[sid][feats + ['contact_label']].dropna()
            if tr['contact_label'].nunique() < 2 or te['contact_label'].nunique() < 2:
                results[combo].append(np.nan)
                continue
            p = _pipe()
            p.fit(tr[feats].values, tr['contact_label'].values)
            results[combo].append(
                roc_auc_score(te['contact_label'].values,
                              p.predict_proba(te[feats].values)[:, 1]))
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLICATION FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'{stem}.{ext}'))
    plt.close(fig)
    print(f'  → {stem}.png / .pdf')


# ── shared y-position builder for grouped combo plots ─────────────────────────
def _build_rows(res_ref, sort_descending=True):
    """Return list of (cat_label, cat_key, combo, y_pos) with group gaps."""
    GAP, STEP = 0.60, 1.0
    rows, y = [], 0.0
    for cat_label, cat_key, combos in CAT_GROUPS:
        sign = -1 if sort_descending else 1
        sc = sorted(combos, key=lambda c: sign * np.nanmean(res_ref[c]))
        for c in sc:
            rows.append((cat_label, cat_key, c, y))
            y += STEP
        y += GAP
    return rows


def _add_group_labels(ax, rows):
    """Colored category labels left of y-axis (kept for backward compat)."""
    group_ys = {}
    for cat_label, cat_key, c, yi in rows:
        key = (cat_label, cat_key)
        group_ys.setdefault(key, [yi, yi])
        group_ys[key][1] = yi
    for (cat_label, cat_key), (y_lo, y_hi) in group_ys.items():
        ax.text(-0.005, (y_lo + y_hi) / 2, cat_label,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=7.5,
                color=WONG[cat_key], fontweight='bold')


def _group_labels_on_sep(ax, rows, x_start):
    """
    Group labels inside the plot area, never co-located with y-tick labels.

    Labels for groups 2..N are placed just above each inter-group separator
    line at x_start (left edge of data area).  The first group's label goes
    just below its first row (in the bottom margin).
    """
    # First group: below its first bar row, in the bottom margin
    ax.text(x_start, rows[0][3] - 0.40, rows[0][0],
            ha='left', va='top', fontsize=7.0,
            color=WONG[rows[0][1]], fontweight='bold', alpha=0.88, zorder=5)
    for i in range(len(rows) - 1):
        if rows[i][0] != rows[i + 1][0]:
            sep = (rows[i][3] + rows[i + 1][3]) / 2
            ax.axhline(sep, color='#D0D0D0', lw=0.9, zorder=1)
            ax.text(x_start, sep + 0.06, rows[i + 1][0],
                    ha='left', va='bottom', fontsize=7.0,
                    color=WONG[rows[i + 1][1]], fontweight='bold',
                    alpha=0.88, zorder=5)


def _add_group_separators(axes_list, rows):
    for i, (cat_label, _, __, yi) in enumerate(rows[:-1]):
        if rows[i + 1][0] != cat_label:
            sep = (yi + rows[i + 1][3]) / 2
            for ax in axes_list:
                ax.axhline(sep, color='#E4E4E4', lw=0.7, zorder=1)


def _set_y_axis(ax, rows, show_labels=True):
    y_vals = [r[3] for r in rows]
    ax.set_yticks(y_vals)
    ax.set_yticklabels([r[2] for r in rows] if show_labels else
                       ['' for _ in rows], fontsize=8.0)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.set_ylim(y_vals[0] - 0.7, y_vals[-1] + 0.7)
    ax.spines['left'].set_visible(False)


# ── Fig 1: horizontal bar chart — RQ2 Exp-A ablation ─────────────────────────
def fig_rq2_expa(res_a):
    rows   = _build_rows(res_a, sort_descending=True)
    y_vals = [r[3] for r in rows]

    fig_h  = max(4.5, len(rows) * 0.45 + 1.0)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    BAR_H   = 0.36

    for cat_label, cat_key, c, yi in rows:
        col    = WONG[cat_key]
        mean_v = np.nanmean(res_a[c])
        std_v  = np.nanstd(res_a[c])

        ax.barh(yi, mean_v, height=BAR_H, color=col, alpha=0.82, zorder=3)
        if std_v > 0:
            ax.errorbar(mean_v, yi, xerr=std_v, fmt='none',
                        ecolor='#444444', capsize=2.5, elinewidth=0.9, zorder=4)
        ax.text(mean_v + 0.012, yi, f'{mean_v:.2f}',
                ha='left', va='center', fontsize=7.0, color='#333333')

    # group labels inside the plot, above each separator — never co-located
    # with y-tick labels (which are outside the axes to the left)
    _group_labels_on_sep(ax, rows, x_start=0.01)

    ax.axvline(0.50, color='#999999', ls='--', lw=0.9, zorder=1)
    ax.axvline(0.85, color='#CC3333', ls='--', lw=0.9, zorder=1, alpha=0.8)
    ax.text(0.502, 0.999, 'Chance',    transform=ax.get_xaxis_transform(),
            ha='left', va='bottom', fontsize=6.5, color='#999999')
    ax.text(0.852, 0.999, 'Threshold', transform=ax.get_xaxis_transform(),
            ha='left', va='bottom', fontsize=6.5, color='#CC3333')

    ax.set_yticks(y_vals)
    ax.set_yticklabels([r[2] for r in rows], fontsize=8)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(y_vals[0] - 0.7, y_vals[-1] + 0.7)
    ax.spines['left'].set_visible(False)

    # x starts at 0 so bar lengths faithfully represent magnitude differences
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('AUC  (mean ± std across subjects)', fontsize=9)
    ax.set_title('RQ2 — Feature Combination Ablation  ·  Exp-A (Tap)',
                 fontsize=10, pad=10)
    fig.tight_layout()
    _save(fig, 'fig1_rq2_expa')


# ── Fig 2: selective 3×2 temporal panel ───────────────────────────────────────
# 6 features chosen for narrative value: 2 strong / 2 moderate / 2 weak
_FIG2 = [
    # (feature, human-readable y-label, combo for AUC, Wong color key)
    ('dist2d_palm_0',      '2D Fingertip Distance',  'd+dist2d',    'geometric'),
    ('approach_theta',     'Approach Angle',          'd+theta',     'directional'),
    ('flow_mag',           'Optical Flow',            'd+flow',      'appearance'),
    ('brightness_contact', 'Contact Brightness',      'd+brightness','appearance'),
    ('dist_raw',           'Depth Distance',          'd_only',      'depth'),
    ('v_n',                'Normal Velocity',         'd+v_n',       'kinematic'),
]
_ROW_LABELS = ['Strong signal', 'Moderate signal', 'Weak / no signal']


def fig_rq2_temporal(df_s01, res_a):
    curves = _temporal_curves(df_s01)

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 5.8))

    for idx, (feat, ylabel, combo, cat_key) in enumerate(_FIG2):
        row_i, col_i = divmod(idx, ncols)
        ax  = axes[row_i, col_i]
        col = WONG[cat_key]

        t_arr, means, stds, n_min = curves.get(
            feat, (np.arange(-WINDOW, WINDOW + 1), None, None, 0))

        if means is not None:
            valid = ~np.isnan(means)
            if valid.any():
                ax.plot(t_arr[valid], means[valid], lw=2.0, color=col)
                ax.fill_between(t_arr[valid],
                                (means - stds)[valid],
                                (means + stds)[valid],
                                alpha=0.25, color=col)
                lo = (means - stds)[valid].min()
                hi = (means + stds)[valid].max()
                pad = max((hi - lo) * 0.30, 1e-6)
                ax.set_ylim(lo - pad, hi + pad)
            else:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='#AAAAAA')

        # contact onset + phase shading
        ax.axvline(0, color='#CC3333', lw=1.3, ls='--', alpha=0.85)
        ax.axvspan(0, WINDOW, alpha=0.055, color='#CC3333')

        # AUC badge — 2 decimal places, upper right
        auc_v = np.nanmean(res_a[combo])
        ax.text(0.97, 0.96, f'AUC = {auc_v:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7.5, color='#222222',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#C8C8C8', alpha=0.92, lw=0.6))

        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(-WINDOW, WINDOW)

        # x-axis label only on bottom row
        if row_i == nrows - 1:
            ax.set_xlabel('Frames relative to contact onset', fontsize=8.5)
        else:
            ax.tick_params(labelbottom=False)

        # row label: upper-left corner of leftmost subplot, light gray
        if col_i == 0:
            ax.text(0.03, 0.97, _ROW_LABELS[row_i],
                    transform=ax.transAxes,
                    ha='left', va='top', fontsize=7.0,
                    color='#BBBBBB', fontstyle='italic')

        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')

    fig.suptitle('Feature Dynamics Around Contact Onset  ·  Exp-A, s01',
                 fontsize=10, y=1.02)
    fig.text(0.5, -0.01,
             'Red dashed line = contact onset (t = 0)  ·  shaded = contact phase',
             ha='center', fontsize=7, color='#888888')
    fig.tight_layout(h_pad=0.55, w_pad=1.0)
    _save(fig, 'fig2_rq2_temporal')


# ── Fig 3: slope chart (left 70%) + lollipop ΔAUC (right 30%) ────────────────
def fig_rq3_ab(res_a, res_b):
    rows   = _build_rows(res_a, sort_descending=True)
    y_vals = [r[3] for r in rows]

    fig_h  = max(4.5, len(rows) * 0.50 + 1.2)
    fig    = plt.figure(figsize=(8.0, fig_h))
    gs     = GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1.0], wspace=0.04)
    ax_m   = fig.add_subplot(gs[0])
    ax_d   = fig.add_subplot(gs[1], sharey=ax_m)

    DELTA_THRESH = 0.03

    for cat_label, cat_key, c, yi in rows:
        col   = WONG[cat_key]
        a_m   = np.nanmean(res_a[c])
        b_m   = np.nanmean(res_b[c])
        delta = b_m - a_m
        lc    = ('#009E73' if delta > DELTA_THRESH else
                 '#D55E00' if delta < -DELTA_THRESH else '#BBBBBB')

        # slope: Exp-A dot → connecting line → Exp-B dot
        # line color encodes ΔAUC direction; dot color encodes category
        ax_m.plot([a_m, b_m], [yi, yi], '-', color=lc, lw=1.6,
                  alpha=0.85, zorder=2, solid_capstyle='round')
        ax_m.scatter(a_m, yi, marker='o', s=54, color=col,
                     zorder=4, linewidths=0.6, edgecolors='white')
        ax_m.scatter(b_m, yi, marker='D', s=38, facecolors='white',
                     edgecolors=col, linewidths=1.4, zorder=4)

        # lollipop in right panel
        ax_d.plot([0, delta], [yi, yi], '-', color=lc, lw=1.2,
                  alpha=0.85, zorder=2)
        ax_d.scatter(delta, yi, marker='o', s=36, color=lc, zorder=3)

    # group labels inside the plot, above each separator
    _group_labels_on_sep(ax_m, rows, x_start=0.285)
    # mirror separators to ΔAUC panel
    for i in range(len(rows) - 1):
        if rows[i][0] != rows[i + 1][0]:
            sep = (rows[i][3] + rows[i + 1][3]) / 2
            ax_d.axhline(sep, color='#D0D0D0', lw=0.9, zorder=1)

    ax_m.set_yticks(y_vals)
    ax_m.set_yticklabels([r[2] for r in rows], fontsize=8)
    ax_m.tick_params(axis='y', length=0)
    ax_m.set_ylim(y_vals[0] - 0.7, y_vals[-1] + 0.7)
    ax_m.spines['left'].set_visible(False)

    ax_m.axvline(0.50, color='#999999', ls='--', lw=0.9)
    ax_m.axvline(0.85, color='#CC3333', ls='--', lw=0.9, alpha=0.6)
    ax_m.set_xlim(0.28, 1.06)
    ax_m.set_xlabel('AUC  (5-fold CV mean)', fontsize=9)
    ax_m.set_title('RQ3 — Exp-A (Tap) vs Exp-B (Writing)', fontsize=10, pad=8)

    h_a  = ax_m.scatter([], [], marker='o', s=52, color='#777777',
                         label='Exp-A  (tap)')
    h_b  = ax_m.scatter([], [], marker='D', s=38, facecolors='white',
                         edgecolors='#777777', linewidths=1.4,
                         label='Exp-B  (writing)')
    h_up = plt.Line2D([0],[0], ls='-', lw=1.6, color='#009E73',
                       label='delta > +0.03')
    h_dn = plt.Line2D([0],[0], ls='-', lw=1.6, color='#D55E00',
                       label='delta < -0.03')
    ax_m.legend(handles=[h_a, h_b, h_up, h_dn],
                loc='lower right', fontsize=7.0, ncol=2,
                handletextpad=0.4, columnspacing=0.5)

    ax_d.axvline(0, color='#AAAAAA', lw=0.9)
    ax_d.set_xlim(-0.28, 0.45)
    ax_d.set_xlabel('delta AUC (B - A)', fontsize=9)
    ax_d.set_title('delta AUC', fontsize=10, pad=8)
    ax_d.spines['left'].set_visible(False)
    ax_d.tick_params(axis='y', labelleft=False, length=0)

    fig.tight_layout()
    _save(fig, 'fig3_rq3_expa_vs_expb')


# ── Fig 4: 2×2 small multiples slopegraph — in-dist vs transfer ───────────────
def fig_rq3_transfer(res_b_id, res_tr):
    from matplotlib.lines import Line2D

    FOCUS  = ['d_only', 'd+theta', 'd+dist2d', 'fusion']
    # Use exact combo names (consistent with Fig 1/2) — avoids "only" misnomer
    # and Unicode arrows that break PDF font rendering
    TITLES = {
        'd_only':   'd_only',
        'd+theta':  'd+theta',
        'd+dist2d': 'd+dist2d',
        'fusion':   'fusion',
    }
    S_COLS   = ['#0072B2', '#D55E00']   # s01=blue, s02=vermilion (both Wong)
    X_LABELS = ['In-dist\n(B)', 'A-to-B\nTransfer']

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8), sharey=True)
    axes = axes.flatten()

    for ax_i, combo in enumerate(FOCUS):
        ax = axes[ax_i]
        cat_key = COMBO_CAT[combo]
        id_vals = res_b_id[combo]
        tr_vals = res_tr[combo]

        n = min(len(id_vals), len(tr_vals))
        for si in range(n):
            iv = id_vals[si]
            tv = tr_vals[si]
            if np.isnan(iv) or np.isnan(tv):
                continue
            col   = S_COLS[si]
            delta = tv - iv
            ax.plot([0, 1], [iv, tv], '-o', color=col,
                    lw=1.6, markersize=6, zorder=3)
            # Δ annotation mid-segment, offset to avoid overlap
            y_off = 0.025 * (1 if si == 0 else -1)
            ax.text(0.5, (iv + tv) / 2 + y_off, f'{delta:+.2f}',
                    ha='center', va='center', fontsize=7.0,
                    color=col, fontweight='bold')

        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0.3, 1.02)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(X_LABELS, fontsize=8)
        ax.set_title(TITLES[combo], fontsize=9, color=WONG[cat_key], pad=4)
        ax.axhline(0.5, color='#BBBBBB', ls='--', lw=0.8, zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax_i % 2 == 0:
            ax.set_ylabel('AUC', fontsize=8.5)
        else:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', labelleft=False, length=0)

    h_s = [Line2D([0],[0], ls='-', marker='o', color=c, markersize=5,
                  label=f's0{i+1}')
           for i, c in enumerate(S_COLS)]
    fig.legend(handles=h_s, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.03), frameon=True)

    fig.suptitle('RQ3 — Cross-Scene Transfer  ·  B in-dist vs A-to-B',
                 fontsize=10)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    _save(fig, 'fig4_rq3_transfer')


# ── summary table ─────────────────────────────────────────────────────────────
def summary_table(res_a, res_b, res_tr):
    rows = []
    for c in COMBOS:
        a_m, a_s = np.nanmean(res_a[c]), np.nanstd(res_a[c])
        b_m, b_s = np.nanmean(res_b[c]), np.nanstd(res_b[c])
        t_m, t_s = np.nanmean(res_tr[c]), np.nanstd(res_tr[c])
        rows.append({
            'combo':          c,
            'A_mean':         round(a_m, 4),
            'A_std':          round(a_s, 4),
            'B_mean':         round(b_m, 4),
            'B_std':          round(b_s, 4),
            'dAUC_BminusA':   round(b_m - a_m, 4),
            'Transfer_mean':  round(t_m, 4),
            'Transfer_std':   round(t_s, 4),
            'dAUC_TminusB':   round(t_m - b_m, 4),
        })
    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  RQ2 & RQ3 Feature Ablation Analysis')
    print('=' * 60)

    print('\n[1/6] Loading data …')
    subj_a = load_exp_a()
    subj_b = load_exp_b()
    print(f'  Exp-A: {sorted(subj_a)}')
    print(f'  Exp-B: {sorted(subj_b)}')

    print('\n[2/6] Exp-A ablation (5-fold CV, LogReg+RobustScaler) …')
    res_a = ablation(subj_a)
    for c, aucs in res_a.items():
        vals = ['%.4f' % v if not np.isnan(v) else 'NaN' for v in aucs]
        print(f'  {c:<18} {vals}  mean={np.nanmean(aucs):.4f}')

    print('\n[3/6] Wilcoxon signed-rank tests (Exp-A) …')
    wx = wilcoxon_tests(res_a)
    print(wx.to_string(index=False))
    print('  Note: n=2 — descriptive only.')

    print('\n[4/6] Exp-B ablation …')
    res_b = ablation(subj_b)
    for c, aucs in res_b.items():
        vals = ['%.4f' % v if not np.isnan(v) else 'NaN' for v in aucs]
        print(f'  {c:<18} {vals}  mean={np.nanmean(aucs):.4f}')

    print('\n[5/6] Cross-scene transfer (train A / test B) …')
    res_tr = transfer_auc(subj_a, subj_b)
    for c, aucs in res_tr.items():
        vals = ['%.4f' % v if not np.isnan(v) else 'NaN' for v in aucs]
        print(f'  {c:<18} {vals}  mean={np.nanmean(aucs):.4f}')

    print('\n[6/6] Generating publication figures …')
    fig_rq2_expa(res_a)
    fig_rq2_temporal(subj_a['s01'], res_a)
    fig_rq3_ab(res_a, res_b)
    fig_rq3_transfer(res_b, res_tr)

    df_sum = summary_table(res_a, res_b, res_tr)
    df_sum.to_csv(os.path.join(FIG_DIR, 'rq2_rq3_results.csv'), index=False)
    wx.to_csv(os.path.join(FIG_DIR, 'rq2_wilcoxon.csv'), index=False)

    print('\n=== Summary ===')
    print(df_sum.to_string(index=False))
    print(f'\nAll outputs → {FIG_DIR}/')


if __name__ == '__main__':
    main()
