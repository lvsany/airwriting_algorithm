# Exp-B 使用教程

书写场景实验采用两阶段流程：**第一阶段采集（B1）→ 第二阶段补标注（B2）**。  
当前仓库已实现 **B1 + B2**：B1 采集视频与特征，B2 回放标注 `contact_label`。

---

## 环境准备

在项目根目录安装依赖：

```bash
pip install opencv-python mediapipe numpy pandas
```

所有命令默认在 `airwriting_algorithm/` 根目录执行。

---

## 流程总览

```
Step 1  准备采集环境（摄像头 / 光照 / 双手入镜）
Step 2  运行 B1 采集脚本（交互输入受试者与场景配置）
Step 3  按 q 结束采集，保存 mp4 + features.csv + meta.json
Step 4  用 B2 标注脚本补写 contact_label
Step 5  用 B3 脚本完成统计分析与出图
```

---

## Step 1：采集前准备

1. 保证画面里能稳定看到双手（书写手 + 画板手）
2. 尽量减少背景干扰与强反光
3. 根据试验计划确定本次配置：
   - 光照：`normal / low / side`
   - 速度：`slow / normal / fast`

---

## Step 2：运行 B1 采集

```bash
python experiments/exp_b/b1_collect.py
```

也可以进入目录后运行：

```bash
cd experiments/exp_b
python b1_collect.py
```

启动后会依次询问：

```text
受试者编号（如 s02，直接回车使用上次记录）:
光照条件 [normal / low / side]:
书写速度 [slow / normal / fast]:
摄像头编号（默认 0，直接回车跳过）:
提示词（空格/逗号分隔，直接回车使用默认常用词）:
```

输入后会显示输出文件路径，按 Enter 开始采集。

---

## Step 3：采集界面与结束

采集窗口为全屏 OpenCV 画面，左上角实时显示：

1. `REC ●`、已采集时长与帧号
2. `sid · lighting · speed`
3. `MediaPipe: ✓ 双手检测`（未检测到时为红色 `✗`）
4. `d / vn / sd` 与 `shadow / flow`（风格与 Exp-A 采集界面一致）
5. `dist2d_palm_0` 与 `approach_theta`（检测成功时显示）
6. `Target [i/N]: word` 当前要写的单词

画面中会显示双手骨架与书写手指尖红点，便于对齐 Exp-A 的调试观察方式。  
说明：OpenCV 内置字体对中文支持有限，因此窗口叠字使用英文（终端交互仍是中文）。

按键说明：

- `n`：切换到下一个提示词
- `p`：切换到上一个提示词
- `q`：结束采集并保存

按 `q` 后脚本会自动：

1. 保存视频
2. 保存逐帧特征 CSV
3. 保存元数据 JSON
4. 在终端打印采集摘要（总帧数、时长、实际帧率、双手检测率等）

---

## Step 4：运行 B2 标注

```bash
python experiments/exp_b/b2_label.py
```

脚本会自动列出 `experiments/data_b/` 下可标注的 `exp_b1_*_features.csv`，选择后进入回放标注界面。

标注界面按键：

- `space`：切换当前标签（0/1）
- `0` / `1`：直接设置当前标签
- `p`：播放/暂停
- `j` / `l`：上一帧 / 下一帧
- `a` / `d`：后退10帧 / 前进10帧
- `[` / `]`：减速 / 加速（0.25x, 0.5x, 1.0x, 1.5x, 2.0x）
- `c`：清空当前帧标签
- `o`：切换覆盖模式（ON 时播放会覆盖已有标签）
- `s`：保存中间结果
- `q`：保存并退出

标签约定：

- `1` = CONTACT
- `0` = IDLE

输出文件：

- `exp_b1_{sid}_{lighting}_{speed}_features_labeled.csv`
- `exp_b1_{sid}_{lighting}_{speed}_features_labeled_meta.json`

---

## Step 5：运行 B3 完整分析

```bash
python experiments/exp_b/b3_analyze.py
```

输出目录：

- `experiments/data_b/figures/`

主要输出图：

- `task_b1_overview.pdf/.png`
- `task_b2_discriminability.pdf/.png`
- `task_b3_kde_compare.pdf/.png`
- `task_b4_temporal_alignment.pdf/.png`
- `task_b5_ablation.pdf/.png`
- `task_b6_transfer.pdf/.png`
- `task_b7_loso.pdf/.png`（受试者数不足会跳过）
- `task_b8_grouped.pdf/.png`

---

## 输出文件说明

输出目录：`experiments/data_b/`（不存在会自动创建）

命名规则：

- 视频：`exp_b1_{sid}_{lighting}_{speed}.mp4`
- 特征：`exp_b1_{sid}_{lighting}_{speed}_features.csv`
- 元数据：`exp_b1_{sid}_{lighting}_{speed}_meta.json`
- 标注后特征：`exp_b1_{sid}_{lighting}_{speed}_features_labeled.csv`

CSV 与 Exp-A 字段**严格兼容**，字段顺序一致；差异点是：

- `contact_label` 在 B1 全部为空字符串 `''`
- B2 将 `contact_label` 填为 `0/1`
- 检测丢失帧的相关特征字段依旧保持空字符串（不是 0）

---

## 常见问题

**1）窗口打开但第三行长期显示“未检测到双手”**  
检查双手是否同时入镜，并确保画板手掌心可见、无遮挡。

**2）采集后很多特征为空**  
这通常表示该帧双手检测或关键点失败；B1 设计上会保留空值，不做 0 填充。

**3）摄像头打不开**  
修改启动时输入的摄像头编号（如 0/1/2）后重试。

**4）程序马上结束且总帧数=0**  
这是“摄像头能打开但首帧读取失败”。常见于相机被占用或当前编号不是可读视频流。优先改用 `0`，并关闭 Zoom/Meet/微信等占用摄像头的软件后重试。
