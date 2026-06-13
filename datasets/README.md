# Air-Writing 数据集说明

## 完整流程概览

```
第一步：用自有方法采集参考数据（摄像头模式，自动录制原始视频）
  python datasets/test.py --user U01 --method own_framework
  → Exp3/own_framework/exp3_U01.json   ← 接触轨迹
  → Exp3/video/exp3_U01_raw_<ts>.mp4  ← 原始录像

第二步：在服务器上用其他方法对同一段视频重跑（回放模式，无需摄像头）
  python datasets/test.py --user U01 --method palmpad
  → Exp3/palmpad/exp3_U01.json

第三步：识别评估（将各方法的笔画送入 VLM）
  python datasets/recognize.py datasets/Exp3/<method>/exp3_U01.json

第四步：跨用户汇总
  python datasets/analyze_results.py --dir datasets/Exp3/<method>
```

---

## 目录结构

```
datasets/
├── contact_detectors/          # 可插拔接触检测方法（每种方法一个文件）
│   ├── base.py                 # 抽象接口 ContactDetectorBase
│   ├── own_framework.py        # 我们自己的方法（封装 DualHandDetector）
│   ├── palmpad.py              # PalmPad 方法（CHI 2025）
│   └── __init__.py             # 注册表 + build_detector() 工厂函数
│
├── Exp3/
│   ├── video/                  # 原始录像（摄像头模式自动保存于此）
│   │   ├── exp3_U01_raw_<ts>.mp4
│   │   └── ...
│   ├── own_framework/          # 自有方法的接触轨迹 JSON + 识别结果
│   │   ├── exp3_U01.json
│   │   ├── exp3_U01_results.json
│   │   └── ...
│   ├── palmpad/                # PalmPad 方法的接触轨迹 JSON（回放模式生成）
│   │   └── ...
│   └── <future_method>/        # 未来新方法的数据（自动创建）
│
├── palmpad_checkpoints/
│   └── best.pt                 # PalmPad 接触检测模型权重
│
├── prestudy/                   # 预实验脚本
│   ├── collect_tap.py
│   ├── collect_write.py
│   ├── runner.py
│   └── analyze.py
│
├── data_prestudy/              # 预实验数据
│   ├── tap/
│   ├── write/
│   └── prestudy_results.json
│
├── test.py                     # Exp3 数据采集主脚本
├── recognize.py                # 识别评估
├── analyze_results.py          # 跨用户汇总
├── exp3_eval_palmpad.py        # PalmPad 离线视频重评
└── words.txt                   # 候选词表（MacKenzie & Soukoreff）
```

---

## 一、接触检测方法接口

所有接触检测方法共享统一接口，位于 `contact_detectors/base.py`：

| 方法 | 说明 |
|------|------|
| `process(frame) → bool` | 处理当前帧，返回 `is_writing` |
| `get_screen_position() → (x, y)` | 书写手食指指尖的屏幕像素坐标 |
| `get_writing_position() → (u, v) \| None` | 掌面局部坐标系中的指尖坐标（用于轨迹记录） |
| `consume_still_hold_event() → bool` | 静止保持事件（接触且 ~1 s 不移动时触发，用于自动跳题） |
| `hover_result` | 悬停校准进度（`needs_calibration=False` 时忽略） |
| `reset()` | 重置内部状态（重新校准时调用） |

**已注册方法：**

| 名称 | 文件 | 说明 |
|------|------|------|
| `own_framework` | `own_framework.py` | 基于掌面坐标系 + 悬停锚点的几何接触检测 |
| `palmpad` | `palmpad.py` | ResNet18 + LSTM 视觉接触检测（CHI 2025） |

**添加新方法：**
1. 在 `contact_detectors/` 下新建 `<name>.py`，继承 `ContactDetectorBase`
2. 在 `__init__.py` 的 `REGISTRY` 字典中添加一行
3. 如需 CLI 参数，在 `test.py` 的 `main()` 里添加对应的 `argparse` 条目

---

## 二、数据采集（`test.py`）

`test.py` 支持两种工作模式，由 `--method` 自动选择：

| 模式 | 触发条件 | 视频输入 | 是否录制 |
|------|----------|----------|----------|
| **摄像头模式** | `--method own_framework` | 实时摄像头 | 是，保存原始视频 |
| **回放模式** | `--method <其他>` | 已录制的 own_framework 视频 | 否 |

回放模式的核心优势：**所有对比方法处理完全相同的视频帧**，确保比较公平。  
服务器无摄像头也可正常运行 PalmPad 等方法。

### 环境要求

**通用：** Python 3.9+，OpenCV，NumPy

**摄像头模式（own_framework）额外要求：**
- `src/hand_track/dual_hand_detector.py` 可正常导入
- 连接摄像头（优先索引 1，自动回退到 0）

**回放模式（palmpad 等）额外要求：**
- `torch`、`torchvision`（PalmPad 推理）
- `palmpad_checkpoints/best.pt`（模型权重）
- MediaPipe ≥ 0.10（Tasks API；首次运行自动下载 `hand_landmarker.task`）
- 已完成 `own_framework` 采集（需要参考视频和试次序列）

### 运行方式

```bash
# ── 第一步：摄像头采集（自有方法，需连接摄像头）──────────────────────
python datasets/test.py --user U01 --method own_framework

# 从特定阶段继续（例如跳到 LEVEL2）
python datasets/test.py --user U01 --method own_framework --start LEVEL2

# ── 第二步：回放评估（服务器，无摄像头）─────────────────────────────
# 自动查找 own_framework 最新视频和试次序列
python datasets/test.py --user U01 --method palmpad

# 指定权重路径
python datasets/test.py --user U01 --method palmpad \
    --checkpoint datasets/palmpad_checkpoints/best.pt

# 手动指定参考视频和试次 JSON（覆盖自动检测）
python datasets/test.py --user U01 --method palmpad \
    --video  datasets/Exp3/own_framework/exp3_U01_raw_1718000000.mp4 \
    --trials datasets/Exp3/own_framework/exp3_U01.json
```

数据保存至 `datasets/Exp3/<method_name>/exp3_<user>.json`。

### 采集/回放流程

**摄像头模式（own_framework）：**

| 阶段 | 说明 |
|------|------|
| **CALIB** | 悬停校准。双手静止，等待检测器建立基线，进度条满后自动进入下一阶段。 |
| **PRACTICE** | 自由练习。静止 ~1 s 自动清空画布，按 `SPACE` 进入 LEVEL1。 |
| **LEVEL1** | 单字符（A–Z 或 0–9）。随机采样 10 个字符，每个写 2 次，共 20 试次。 |
| **LEVEL2** | 短词（3–5 字符），从 `words.txt` 随机采样 15 个。 |
| **LEVEL3** | 长词（6 字符及以上），从 `words.txt` 随机采样 5 个。 |

**回放模式（palmpad 等）：**

- 直接从 **LEVEL1** 开始，跳过 CALIB 和 PRACTICE
- 试次目标顺序从 `own_framework/exp3_<user>.json` 加载，与原始采集完全一致
- 视频自动定位到第一条 LEVEL1 笔画前 3 s 处开始播放
- HUD 右上角显示 `[REPLAY·<method>]` 标签及回放进度 `帧号/总帧数 (%)`

书写确认（两种模式均适用）：静止 ~1 s 自动确认，或按 `SPACE` 手动确认。  
每次确认后**增量写入磁盘**（断点续采：同一 `--user --method` 组合自动加载已有数据）。

### 键盘控制

| 按键 | 功能 |
|------|------|
| `SPACE` | 手动确认当前试次，进入下一个 |
| `C` | 清空当前试次的笔画（重写） |
| `R` | 重置当前阶段（从第一个试次重新开始，删除本阶段已保存数据） |
| `N` | 跳过当前阶段 |
| `H` | 触发重新校准（回到 CALIB 阶段） |
| `Q` / `ESC` | 退出采集 |

---

## 三、数据格式

`exp3_<user>.json` 为 JSON 数组，每个元素对应一次书写试次：

```json
{
  "user_id": "U01",
  "method":  "palmpad",
  "level":   "LEVEL2",
  "timestamp": 1779604296.16,
  "target":  "world",
  "strokes": [
    [
      { "x": 556, "y": 291, "u": 0.157, "v": 0.293, "t": 1779604291.75, "f": 1024 },
      ...
    ]
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | str | 被试编号 |
| `method` | str | 采集时所用接触检测方法 |
| `level` | str | `LEVEL1` / `LEVEL2` / `LEVEL3` |
| `timestamp` | float | 试次完成时的 Unix 时间戳 |
| `target` | str | 目标字符或单词 |
| `strokes` | list[list] | 笔画列表，每条笔画是一段连续落笔轨迹 |

每个轨迹点：

| 字段 | 说明 |
|------|------|
| `x`, `y` | 屏幕像素坐标 |
| `u`, `v` | 掌面局部坐标系坐标（EMA 平滑，用于渲染和识别） |
| `t` | 该点的 Unix 时间戳（秒） |
| `f` | 帧编号 |

---

## 四、PalmPad 离线视频评估（`exp3_eval_palmpad.py`）

> **注意：** 此脚本的功能已被 `test.py` 的回放模式（`--method palmpad`）完整替代，  
> 并与其他方法共享统一接口和输出格式。新实验建议直接使用 `test.py`：
> ```bash
> python datasets/test.py --user U01 --method palmpad
> ```
> `exp3_eval_palmpad.py` 保留用于独立批量评估或向后兼容。

对 `test.py` 录制的原始视频进行离线重播，用 PalmPad 模型重新做接触检测，  
生成与在线采集格式一致的笔画数据，便于对比两种方法在同一录像上的差异。

### 环境要求

```
Python 3.9+，mediapipe >= 0.10（Tasks API），torch，torchvision，opencv-python
```

> MediaPipe 0.10 起移除了旧版 `mp.solutions` 接口，改用基于模型文件的 Tasks API。  
> 首次运行时脚本自动从 Google 服务器下载 `hand_landmarker.task` 并缓存到  
> `datasets/hand_landmarker.task`，之后无需重复下载。

### 文件路径约定

| 路径 | 说明 |
|------|------|
| `datasets/palmpad_checkpoints/best.pt` | PalmPad 权重 |
| `datasets/Exp3/video/exp3_U0*_raw_*.mp4` | 原始视频 |
| `datasets/Exp3_Results/` | 评估输出（自动创建） |
| `datasets/hand_landmarker.task` | MediaPipe 手部关键点模型 |

### 运行方式

```bash
python datasets/exp3_eval_palmpad.py
```

### 处理流程

```
Exp3/video/exp3_U0*_raw_*.mp4
    └─► MediaPipe 逐帧手部追踪     提取食指指尖（lm8）、掌心（lm9）
    └─► PalmPad 模型推理           RGB Crop × 2 + 局部光流 → 接触 / 非接触
    └─► 笔画切分                   抬笔时切分；静止 ~1 s 触发 still-hold 分组
    └─► 写入 JSON                  Exp3_Results/eval_results_<user>_<ts>.json
```

---

## 五、识别评估（`recognize.py`）

### 运行方式

```bash
# 评估全部 Level
python datasets/recognize.py datasets/Exp3/own_framework/exp3_U01.json

# 只评估 LEVEL2
python datasets/recognize.py datasets/Exp3/palmpad/exp3_U01.json --level LEVEL2

# 同时保存渲染图
python datasets/recognize.py datasets/Exp3/own_framework/exp3_U01.json \
    --save-images datasets/Exp3/rendered_imgs
```

### 处理流程

```
strokes (u, v)
    └─► render_strokes()    渲染为 256×256 白底黑字 PNG
                              · u/v 取补翻转坐标
                              · 图像整体旋转 -45°（修正掌面坐标系偏转）
                              · 红点标记每段笔画起点
    └─► Qwen3-VL 本地推理   LEVEL1：识别单个大写字母或数字
                             LEVEL2/3：从 words.txt 候选词表中匹配最近词
    └─► 结果比对             大小写不敏感，计算词级与字符级准确率
```

### 输出

```
datasets/Exp3/own_framework/exp3_U01_results.json
```

---

## 六、跨用户汇总统计（`analyze_results.py`）

```bash
# 默认扫描 datasets/Exp3/own_framework/
python datasets/analyze_results.py

# 指定目录
python datasets/analyze_results.py --dir datasets/Exp3/palmpad
```

| 维度 | 指标 | 说明 |
|------|------|------|
| LEVEL1 | Letter Accuracy | 字母（A–Z）识别准确率 |
| LEVEL1 | Digit Accuracy | 数字（0–9）识别准确率 |
| LEVEL2/3 | Word Accuracy (WA) | 整词完全匹配率 |
| LEVEL2/3 | Character Error Rate (CER) | `CER = EditDist(pred, target) / len(target)` |

生成 `analysis_summary.json` 和 `analysis_summary.csv`（可直接导入 Excel / pandas / R）。

---

## 七、预实验（Pre-study）

预实验用于验证各特征对接触检测的区分能力，分两种场景采集数据：

- **Tap（受控点触）**：用贴纸辅助定位，精确标注接触时刻，主要用于 RQ1/RQ2
- **Write（连续书写）**：空格键实时标注，模拟真实书写场景，主要用于 RQ3

### 采集

```bash
python -m datasets.prestudy.runner --task tap   --subject s01 --sticker-color black
python -m datasets.prestudy.runner --task write --subject s01 --lighting normal --speed normal
```

### 分析

```bash
python -m datasets.prestudy.analyze
# 或指定数据路径
python -m datasets.prestudy.analyze \
    --tap-dir data_prestudy/tap --write-dir data_prestudy/write
```

输出：终端打印 RQ1/RQ2/RQ3 AUROC 表格 + `data_prestudy/prestudy_results.json`

---

## 八、预实验特征术语说明

分析脚本使用 5 折分层交叉验证（Logistic Regression + StandardScaler），  
评估指标为 AUROC（ROC 曲线下面积），统计检验采用 DeLong 1988 方法。

### 单特征

| 特征名 | 中文术语 | 说明 |
|--------|----------|------|
| `dist_raw` | **原始三维距离** | 书写手食指指尖到支撑手掌面的原始三维欧氏距离，未经坐标系归一化 |
| `dist_local` | **掌面法向距离**（局部坐标系 n 分量） | 指尖在掌面局部坐标系中的法向分量，即垂直于掌面的有符号距离；接触时趋近于零 |
| `v_n` | **法向速度** | 指尖在掌面法向方向上的速度分量；接触前急剧减小趋近于零 |
| `a_n` | **法向加速度** | 指尖在掌面法向方向上的加速度分量；接触瞬间因受力反向而出现正向峰值 |
| `sigma_d` | **近期距离标准差** | 最近若干帧法向距离的标准差，反映接触稳定性；稳定接触时值较小 |
| `v_t` | **切向速度** | 指尖在掌面切线方向上的速度分量；书写时持续非零，悬停时趋近于零 |
| `approach_theta` | **接近角** | 指尖三维运动方向与掌面法向量的夹角（°）；0° 表示垂直靠近掌面，90° 表示平行滑行 |
| `shadow_score` | **接触阴影得分** | 指尖 32×32 ROI 的拉普拉斯方差；接触时局部阴影使纹理减弱，方差减小 |
| `flow_mag` | **光流幅值** | 指尖区域光学流场的平均幅值；接触时运动受约束，幅值下降 |
| `brightness_contact` | **接触区亮度** | 指尖 32×32 ROI 的灰度均值；指尖受压变形时皮肤局部变亮（血液受压退散） |
| `geo_wrist` | **腕部几何距离**（`dist2d_palm_0`） | 书写手食指指尖到支撑手手腕关键点（lm0）的 2D 像素欧氏距离 |

### 组合特征集

| 特征集名 | 中文术语 | 组成 |
|----------|----------|------|
| `kinematic` | **运动学特征组** | 法向速度 + 法向加速度 + 近期距离标准差 + 切向速度（`v_n, a_n, sigma_d, v_t`） |
| `appearance` | **外观特征组** | 接触阴影得分 + 光流幅值 + 接触区亮度（`shadow_score, flow_mag, brightness_contact`） |
| `geo+theta` | **几何距离 + 接近角** | 腕部几何距离 + 接近角（`dist2d_palm_0, approach_theta`） |
| `geo+theta+vt` | **几何距离 + 接近角 + 切向速度** | 在 `geo+theta` 基础上加入切向速度，捕捉书写动作的切线运动 |
| `geo+appearance` | **几何距离 + 外观特征** | 腕部几何距离 + 完整外观特征组 |
| `geo+theta+appear` | **几何距离 + 接近角 + 外观特征**（完整融合） | `geo+theta` + 完整外观特征组，融合几何、运动方向与视觉线索 |

### 分析场景与研究问题

| 研究问题 | 场景 | 评估方式 | 特征集 |
|----------|------|----------|--------|
| **RQ1** 基础特征区分力 | Tap | 5-fold CV AUROC | `dist_raw`, `kinematic`, 各外观单特征 |
| **RQ2** 最优几何组合 | Tap | 5-fold CV AUROC | `geo_wrist` → `geo+theta` → `geo+theta+vt` → 融合 |
| **RQ3a** 书写场景域内 | Write | 5-fold CV AUROC | `geo+theta` 起各融合组合 |
| **RQ3b** 零样本跨域 | Tap→Write | 直接迁移 AUROC | 同 RQ3a |
| **DeLong 检验** | Write | OOF 预测分 | 逐对组合显著性检验 |

---

## 九、候选词表（`words.txt`）

来源：MacKenzie & Soukoreff phrase set 中的全部唯一词，共约 1164 个。

- LEVEL2 筛选：长度 3–5 字符
- LEVEL3 筛选：长度 ≥ 6 字符
- 识别时将完整词表注入 prompt，引导模型做约束识别而非自由生成
