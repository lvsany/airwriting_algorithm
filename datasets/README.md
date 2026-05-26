# Air-Writing 数据集说明

## 完整流程概览

```
采集 (test.py)  →  识别评估 (recognize.py)  →  汇总统计 (analyze_results.py)
```

## 目录结构

```
datasets/
├── datasets/
│   └── Exp3/
│       ├── exp3_user_01.json            # 笔画轨迹数据
│       ├── exp3_user_01_raw.mp4         # 采集时的原始视频
│       ├── exp3_user_01_results.json    # 识别评估结果（recognize.py 生成）
│       ├── analysis_summary.json        # 跨用户汇总（analyze_results.py 生成）
│       └── analysis_summary.csv        # 同上，CSV 格式
├── rendered_preview/                    # 渲染预览图（可选，--save-images 时生成）
├── test.py                              # 数据采集脚本
├── recognize.py                         # 识别评估脚本
├── analyze_results.py                   # 跨用户汇总统计脚本
└── words.txt                            # 候选词表（MacKenzie & Soukoreff phrase set）
```

---

## 一、数据采集（`test.py`）

### 环境要求

- Python 3.9+
- OpenCV、NumPy、MediaPipe
- 连接摄像头（优先摄像头索引 1，否则自动回退到 0）
- 运行前确保 `src/hand_track/dual_hand_detector.py` 可正常导入

### 运行方式

```bash
python datasets/test.py --user U01
# 指定从某阶段开始（跳过前置阶段）
python datasets/test.py --user U01 --start LEVEL2
```

输出文件保存在 `datasets/Exp3/exp3_{user_id}.json`，视频保存至同目录。

### 采集流程

采集分五个阶段，自动顺序推进：

| 阶段 | 说明 |
|------|------|
| **CALIB** | 悬停校准。双手保持静止，等待检测器建立基线，进度条满后自动进入下一阶段。 |
| **PRACTICE** | 自由练习。可随意书写，静止 0.5 s 自动清空画布。按 `SPACE` 进入 LEVEL1。 |
| **LEVEL1** | 单字符（字母 A–Z 或数字 0–9）。随机采样 10 个字符，每个写 2 次，共 20 试次。 |
| **LEVEL2** | 中等长度单词（5–6 字符），从 `words.txt` 随机采样 15 个。 |
| **LEVEL3** | 长单词（7 字符及以上），从 `words.txt` 随机采样 5 个。 |

**书写确认方式：**
- 静止 0.5 s → 自动确认并推进到下一试次
- 按 `SPACE` → 手动确认

### 键盘控制

| 按键 | 功能 |
|------|------|
| `SPACE` | 手动确认当前试次，进入下一个 |
| `C` | 清空当前试次的笔画（重写） |
| `R` | 重置当前阶段（从第一个试次重新开始） |
| `N` | 跳过当前阶段 |
| `H` | 触发重新校准 |
| `Q` / `ESC` | 退出采集 |

数据在每次确认后**增量写入**磁盘，断电或异常退出不会丢失已完成的试次。

---

## 二、数据格式

`exp3_{user_id}.json` 是一个 JSON 数组，每个元素对应一次书写试次：

```json
{
  "user_id": "user_01",
  "level": "LEVEL2",
  "timestamp": 1779604296.16,
  "target": "world",
  "strokes": [
    [
      { "x": 556, "y": 291, "u": 0.157, "v": 0.293, "t": 1779604291.75 },
      { "x": 551, "y": 293, "u": 0.152, "v": 0.290, "t": 1779604291.83 },
      ...
    ],
    ...
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | str | 被试编号 |
| `level` | str | `LEVEL1` / `LEVEL2` / `LEVEL3` |
| `timestamp` | float | 试次完成时的 Unix 时间戳 |
| `target` | str | 目标字符或单词 |
| `strokes` | list[list] | 笔画列表，每条笔画是一段连续落笔轨迹 |

每个轨迹点字段：

| 字段 | 说明 |
|------|------|
| `x`, `y` | 屏幕像素坐标 |
| `u`, `v` | 掌面归一化坐标（用于渲染，u/v 均取 0–1 范围） |
| `t` | 该点的 Unix 时间戳（秒） |

---

## 三、识别评估（`recognize.py`）

### 运行方式

```bash
# 评估全部 Level
python datasets/recognize.py datasets/datasets/Exp3/exp3_user_01.json

# 只评估某个 Level
python datasets/recognize.py datasets/datasets/Exp3/exp3_user_01.json --level LEVEL2

# 同时保存渲染图供人工检查
python datasets/recognize.py datasets/datasets/Exp3/exp3_user_01.json --save-images datasets/rendered_preview
```

### 处理流程

```
strokes (u, v)
    └─► render_strokes()   将笔画渲染为 256×256 白底黑字 PNG
                            • u、v 均取补（坐标翻转）
                            • 图像整体旋转 -45°（纠正掌面坐标系偏转）
                            • 红点标记每段笔画起点
    └─► Qwen3-VL 本地推理    将图像直接送入本地模型
                            • LEVEL1：识别单个大写字母或数字
                            • LEVEL2/3：从 words.txt 候选词表中选择最匹配的单词
    └─► 结果比对            大小写不敏感，计算词级与字符级准确率
```

### 输出

控制台实时打印每条试次的预测结果，评估完成后输出准确率汇总，并将结果保存至：

```
datasets/datasets/Exp3/exp3_user_01_results.json
```

结果文件格式：

```json
{
  "model": "/home/shared_models/workspace/qwen/Qwen3-VL-8B-Instruct",
  "summary": {
    "LEVEL1": {
      "word_acc": 0.80,
      "char_acc": 0.80,
      "records": [
        { "target": "M", "pred": "M", "correct": true },
        ...
      ]
    },
    "LEVEL2": { "word_acc": 0.31, "char_acc": 0.41, "records": [...] },
    "LEVEL3": { "word_acc": 0.00, "char_acc": 0.16, "records": [...] }
  }
}
```

---

## 四、候选词表（`words.txt`）

来源：MacKenzie & Soukoreff phrase set 中的全部唯一词，共约 1164 个。

- LEVEL2 从中筛选长度 5–6 的单词
- LEVEL3 从中筛选长度 ≥ 7 的单词
- 识别时将完整词表注入 prompt，引导模型做约束识别而非自由生成

---

## 五、跨用户汇总统计（`analyze_results.py`）

### 运行方式

```bash
# 默认扫描 datasets/datasets/Exp3/ 目录
python datasets/analyze_results.py

# 指定其他目录
python datasets/analyze_results.py --dir datasets/datasets/Exp4
```

每次新增用户数据并完成 `recognize.py` 评估后，重跑此脚本即可更新汇总。

### 统计指标

| 维度 | 指标 | 说明 |
|------|------|------|
| LEVEL1 | Letter Accuracy | 字母（A–Z）识别准确率 |
| LEVEL1 | Digit Accuracy | 数字（0–9）识别准确率 |
| LEVEL2/3 | Word Accuracy (WA) | 整词完全匹配率 |
| LEVEL2/3 | Character Error Rate (CER) | 字符错误率，`CER = EditDist(pred, target) / len(target)` |

### 输出文件

脚本运行后在同一目录下生成两个文件：

**`analysis_summary.json`**

```json
{
  "per_user": [
    {
      "user": "exp3_user_01",
      "L1_letter_acc": 0.625, "L1_letter_n": 16,
      "L1_digit_acc":  0.75,  "L1_digit_n":  4,
      "LEVEL2_wa": 0.31, "LEVEL2_cer": 0.548, "LEVEL2_n": 29,
      "LEVEL3_wa": 0.0,  "LEVEL3_cer": 0.757, "LEVEL3_n": 5
    },
    ...
  ],
  "overall": {
    "LEVEL1": { "letter_acc": 0.594, "letter_n": 32, "digit_acc": 0.625, "digit_n": 8 },
    "LEVEL2": { "wa": 0.25, "cer": 0.588, "n": 44 },
    "LEVEL3": { "wa": 0.10, "cer": 0.728, "n": 10 }
  }
}
```

**`analysis_summary.csv`**

逐行为每位用户，最后一行为 `OVERALL`，列结构如下：

```
user | L1_letter_acc | L1_letter_n | L1_digit_acc | L1_digit_n | LEVEL2_wa | LEVEL2_cer | LEVEL2_n | LEVEL3_wa | LEVEL3_cer | LEVEL3_n
```

可直接导入 Excel、pandas 或 R 用于绘图与统计检验。
