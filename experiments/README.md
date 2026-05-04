# experiments/

实验代码目录，包含 Exp-A（纯接触）与 Exp-B（书写场景）两条流程。

```
experiments/
├── exp_a/                  # 阶段一：纯接触检测实验
│   ├── exp_a_runner.py     # 入口，统一调用 A1/A2/A3
│   ├── a1_collect.py       # Exp-A1: 数据采集 + 标注
│   ├── a2_analyze.py       # Exp-A2: 特征统计分析（时序曲线 + 箱线图）
│   ├── a3_roc.py           # Exp-A3: ROC 曲线 + AUC 对比 + 决策点
│   └── TUTORIAL.md         # Exp-A 教程
├── exp_b/                  # 阶段二：书写场景实验
│   ├── b1_collect.py       # Exp-B1: 第一阶段采集（仅视频+特征，contact_label 留空）
│   ├── b2_label.py         # Exp-B2: 第二阶段回放标注（填写 contact_label）
│   ├── b3_analyze.py       # Exp-B3: 完整分析（Task B1~B8）
│   └── TUTORIAL.md         # Exp-B 教程
├── data/                   # Exp-A 数据目录（gitignore）
└── data_b/                 # Exp-B 数据目录（自动创建）
```

## 快速开始

```bash
cd airwriting_algorithm

# 采集数据（荧光贴纸标注）
python -m experiments.exp_a.exp_a_runner --mode collect --subject s01 --label-mode sticker --sticker-color green

# 采集数据（键盘标注）
python -m experiments.exp_a.exp_a_runner --mode collect --subject s01 --label-mode keyboard

# 分析特征
python -m experiments.exp_a.exp_a_runner --mode analyze --subject s01

# ROC 对比
python -m experiments.exp_a.exp_a_runner --mode roc --subject s01

# 全流程一键运行
python -m experiments.exp_a.exp_a_runner --mode all --subject s01

# Exp-B1 书写场景采集（交互式）
python experiments/exp_b/b1_collect.py

# Exp-B2 回放标注（交互式）
python experiments/exp_b/b2_label.py

# Exp-B3 完整分析（输出 figures）
python experiments/exp_b/b3_analyze.py
```

## 教程入口

- Exp-A：`experiments/exp_a/TUTORIAL.md`
- Exp-B：`experiments/exp_b/TUTORIAL.md`

## 标注方式说明

| 实验 | 推荐标注方式 | 原因 |
|-----|-----------|------|
| Exp-A1（纯 tap） | `sticker` | tap 位置固定，贴纸可全程覆盖，全自动 |
| Exp-B1（书写第一阶段） | 无（`contact_label` 先留空） | 第一阶段只做视频与特征采集，标签由后续脚本补充 |

## 依赖

```bash
pip install opencv-python mediapipe numpy pandas matplotlib scipy scikit-learn
```
