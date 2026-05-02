# experiments/

调研实验代码目录，对应 PLAN.md 第三节。

```
experiments/
├── exp_a/                  # 阶段一：纯接触检测实验
│   ├── exp_a_runner.py     # 入口，统一调用 A1/A2/A3
│   ├── a1_collect.py       # Exp-A1: 数据采集 + 标注
│   ├── a2_analyze.py       # Exp-A2: 特征统计分析（时序曲线 + 箱线图）
│   └── a3_roc.py           # Exp-A3: ROC 曲线 + AUC 对比 + 决策点
├── exp_b/                  # 阶段二：书写场景实验（待实现）
└── data/                   # 实验数据与图表（gitignore）
    ├── exp_a1_s01.csv
    └── figures/
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
```

## 标注方式选择

| 实验 | 推荐标注方式 | 原因 |
|-----|-----------|------|
| Exp-A1（纯 tap） | `sticker` | tap 位置固定，贴纸可全程覆盖，全自动 |
| Exp-B1（书写） | `keyboard` | 书写位置不固定，贴纸无法覆盖全掌 |

## 依赖

```
pip install opencv-python mediapipe numpy pandas matplotlib scipy scikit-learn pynput
```
