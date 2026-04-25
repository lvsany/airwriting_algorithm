# Exp-A 使用教程

纯接触检测实验，分三步：**采集数据（A1）→ 特征分析（A2）→ ROC 对比（A3）**。

---

## 环境准备

项目根目录下安装依赖：

```bash
pip install opencv-python mediapipe numpy pandas matplotlib scipy scikit-learn
```

所有命令均在**项目根目录**（`airwriting_algorithm/`）下执行。

---

## 实验流程总览

```
Step 1  准备材料（贴纸 / 确认摄像头）
Step 2  A1 采集  →  生成 exp_a1_{subject}.csv
Step 3  A2 分析  →  生成时序曲线图 + 箱线图
Step 4  A3 ROC   →  生成 ROC 图 + 决策点报告
```

---

## Step 1：材料准备

### 标注方式一：荧光贴纸（推荐，全自动）

1. 购买荧光便利贴或荧光胶带，裁成约 **1cm × 1cm** 小块
2. 贴在**画布手（左手）手掌中心**位置，贴纸朝上
3. 确保在当前光照下贴纸颜色鲜艳（程序会在开始时自动标定参考面积）
4. 每次 tap 都点触贴纸位置

推荐颜色与参数对应：

| 贴纸颜色 | `--sticker-color` 参数 |
|--------|----------------------|
| 荧光绿  | `green`（默认）       |
| 荧光黄  | `yellow`             |
| 荧光粉  | `pink`               |
| 荧光蓝  | `blue`               |
| 黑色    | `black`              |

### 标注方式二：键盘空格键（备选）

无需贴纸，采集时：
- **按住空格键** = 手指接触手掌
- **松开空格键** = 手指抬起

误差约 < 2 帧，适合无贴纸材料时使用。

---

## Step 2：A1 数据采集

### 基本用法

```bash
# 荧光贴纸标注（推荐）
python -m experiments.exp_a.exp_a_runner --mode collect --subject s01

# 键盘标注
python -m experiments.exp_a.exp_a_runner --mode collect --subject s01 --label-mode keyboard

# 指定贴纸颜色
python -m experiments.exp_a.exp_a_runner --mode collect --subject s01 --sticker-color yellow
```

### 采集界面说明

程序启动后弹出摄像头窗口，界面分三个区域：

```
┌─────────────────────────────────────────┐
│ [标注状态]  Sticker: 1243px (0.92) IDLE  │  ← 顶部：贴纸面积和当前标注状态
│ d=8.3mm  vn=-0.12  sd=0.45             │  ← 中部：实时特征值
│ shadow=12.4  flow=0.03                  │
│                                         │
│   [手部骨架可视化]                        │  ← 中部：MediaPipe 关键点
│         ●  接触投影点（红色圆点）         │
│                                         │
│ Frame 1024  Subject: s01               │  ← 底部：帧数和受试者 ID
└─────────────────────────────────────────┘
```

**贴纸标注模式**的启动流程：
1. 将手掌平放，贴纸朝向摄像头，**保持静止 2 秒**
2. 屏幕显示 `Calibrating... 28/30`，等待标定完成
3. 标定完成后显示参考面积，如 `标定完成 贴纸参考面积: 1350 px²`
4. 开始做 tap 动作（点触贴纸 → 抬起，反复 50-100 次）
5. 按 `Q` 或 `ESC` 结束

### 采集规范

每次采集建议包含以下变化，增加数据多样性：

| 变量 | 建议值 |
|-----|-------|
| 接触速度 | 慢速（约 1 次/2秒）、正常（约 1 次/秒）、快速（约 2 次/秒）各 20 次 |
| 接触力度 | 轻触（勉强接触）、正常、重压 各若干次 |
| 手掌角度 | 正对摄像头、略偏 15-30° 各若干次 |

### 输出文件

```
experiments/data/exp_a1_s01.csv
```

CSV 列说明：

| 列名 | 含义 |
|-----|-----|
| `frame_id` | 帧序号 |
| `timestamp` | Unix 时间戳（秒） |
| `contact_label` | 标注：1=接触，0=未接触 |
| `dist_raw` | 指尖到手掌平面距离（mm），手不在范围内时为空 |
| `v_n` | 法向速度：距离对时间的差分（mm/帧） |
| `sigma_d` | 距离滑窗标准差（mm） |
| `v_t` | 切向速度（mm/s） |
| `shadow_score` | 局部阴影梯度（Laplacian 方差） |
| `flow_mag` | 局部光学流幅值 |
| `lm_{0-20}_{x,y,z}` | MediaPipe 21 个关键点归一化坐标 |

---

## Step 3：A2 特征统计分析

```bash
python -m experiments.exp_a.exp_a_runner --mode analyze --subject s01
```

### 输出图表

**图1：特征时序曲线**（`exp_a2_s01_timeseries.png`）

每个特征展示接触事件前后 ±15 帧的均值（实线）± 标准差（阴影），红色虚线为接触起始时刻。

```
期望结论：
  d        接触前下降，接触后在低值稳定
  v_n      接触前出现负峰（趋近），接触后回归 ≈ 0
  sigma_d  接触时明显低于悬停时（接触稳定，悬停抖动）
  shadow   接触时升高（指腹压变形产生阴影）
  flow     接触时趋近 0（接触区域无相对运动）
```

**图2：箱线图 + t 检验**（`exp_a2_s01_boxplot.png`）

每个特征在 IDLE / CONTACT 状态下的分布，标注 p 值和显著性（`***` `**` `*` `ns`）。

### 判读标准

| 特征 | 期望结论 | 若不显著 |
|-----|---------|---------|
| `v_n` | p < 0.001，CONTACT 均值 ≈ 0 | 检查距离计算是否正确 |
| `sigma_d` | p < 0.05，CONTACT 方差更小 | 增加采集数量 |
| `shadow` | 可能 ns（纯 tap 场景光照变化小） | 正常，到书写场景再验证 |
| `flow` | 可能 ns（tap 后静止，流场本就接近 0） | 正常 |

---

## Step 4：A3 ROC 曲线对比

```bash
python -m experiments.exp_a.exp_a_runner --mode roc --subject s01
```

### 测试的 8 种特征组合

| # | 特征组合 | 说明 |
|---|---------|-----|
| 1 | `d` only | 当前系统 baseline |
| 2 | `d + v_n` | 加法向速度 |
| 3 | `d + σ_d` | 加稳定性特征 |
| 4 | `d + v_n + σ_d` | **运动学主方案** |
| 5 | `d + shadow` | 外观特征（阴影）单独 |
| 6 | `d + flow` | 外观特征（光学流）单独 |
| 7 | `d + shadow + flow` | 外观联合 |
| 8 | `d + v_n + σ_d + shadow + flow` | 全特征融合上界 |

分类器：逻辑回归 + 5-fold 交叉验证，报告 `mean AUC ± std`。

### 输出图表

- `exp_a3_s01_roc.png`：8 条 ROC 曲线叠加图（运动学主方案加粗显示）
- `exp_a3_s01_auc_bar.png`：AUC 柱状图（带误差棒，红色虚线为 0.85 门槛）

### 终端决策点报告

```
── 决策点报告 ──────────────────────────────────────
  baseline (d only)          AUC = 0.7312
  kinematic (d+v_n+σ_d)      AUC = 0.9134  ✅ ≥ 0.85，进入阶段二
  fusion vs kinematic        ΔAUC = +0.0187  ⚠️  外观特征贡献有限
────────────────────────────────────────────────────
```

**决策规则**：

| 结果 | 下一步 |
|-----|-------|
| 运动学 AUC ≥ 0.85 | ✅ 进入 Exp-B（书写场景实验） |
| 运动学 AUC < 0.85 | ❌ 重新评估：可能是 MediaPipe z 深度噪声过大，考虑改用双目 |
| 外观 ΔAUC ≥ 0.03 | 融合外观特征到最终方案 |
| 外观 ΔAUC < 0.03 | 外观特征作为消融实验对照，不进入主方案 |

---

## 一键全流程

采集完数据后，直接运行全流程：

```bash
python -m experiments.exp_a.exp_a_runner --mode all --subject s01
```

多受试者批量分析（需要先分别采集）：

```bash
for s in s01 s02 s03; do
    python -m experiments.exp_a.exp_a_runner --mode analyze --subject $s
    python -m experiments.exp_a.exp_a_runner --mode roc --subject $s
done
```

---

## 常见问题

**Q：贴纸标定后一直显示 CONTACT，没有 IDLE？**
检查是否有其他绿色物体在镜头内，或环境光线中绿色分量过强。换用 `--sticker-color pink` 或 `yellow`。

**Q：`dist_raw` 全部为空，所有特征都是 0？**
手没有被识别为书写手/画布手。确认右手（书写手）和左手（画布手）都在镜头内，并且左手掌心朝上。

**Q：ROC 报告 "接触帧过少（<20），无法进行可靠评估"？**
增加采集量，确保至少有 20 个以上有效接触帧。检查标注是否正确（贴纸是否被遮挡触发了 contact_label=1）。

**Q：A2 时序曲线中 v_n 在接触后不趋近 0？**
可能是手掌在接触后还在移动（非纯 tap 动作）。让受试者做更规范的"点触-静止-抬起"动作。
