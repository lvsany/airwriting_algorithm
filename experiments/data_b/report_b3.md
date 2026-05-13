Exp-B 阶段三分析报告

生成时间：2026-05-04

1. 数据与运行条件

Exp-B 输入文件：
- `experiments/data_b/exp_b1_s01_normal_normal_features_labeled.csv`

Exp-B 数据统计：
- 总帧数：1390
- IDLE：916（65.9%）
- CONTACT：474（34.1%）
- 接触事件数：10
- 接触事件中位持续帧数：49
- 接触事件平均持续帧数：47.4

Exp-B 关键特征有效率：
- `dist2d_palm_0`：68.71%
- `approach_theta`：67.77%
- `v_t`：67.77%
- `flow_mag`：47.05%
- `brightness_contact`：68.56%
- `sigma_d`：67.77%

Exp-A 参照：
- 组合结果使用脚本内常量 `EXPA_RESULTS`
- 单特征结果使用脚本内常量 `EXPA_SINGLE`
- 运行时加载的 Exp-A CSV：`data/data_0000/exp_a1_s01.csv`

2. 图表


3. 核心问题结论

Q1：Exp-A 最优方案 `geo+theta` 在书写场景是否仍有效。

结论：在当前 Exp-B 数据中不成立。  
证据：
- Exp-A：`geo+theta` AUROC = 0.964
- Exp-B：`geo+theta` AUROC = 0.668 ± 0.016
- 变化量：ΔAUROC = -0.296

Q2：书写特征 `v_t` 是否带来增益。

结论：当前数据上未观察到显著增益。  
证据：
- 单特征 `v_t`：AUROC = 0.508（接近随机）
- `geo+vt`：AUROC = 0.676 ± 0.019
- `geo+theta+vt`：AUROC = 0.677 ± 0.023
- 与 `geo+theta` 对比 DeLong：p = 0.0565（ns）

Q3：迁移后退化的特征。

结论：几何核心特征和角度特征出现退化。  
按单特征 AUROC 变化（Exp-B - Exp-A）：
- `dist2d_palm_0`：-0.270
- `approach_theta`：-0.269
- `sigma_d`：-0.213
- `dist2d_palm_17`：-0.144

4. 组合消融与迁移结果

Exp-B 组合结果（AUROC）：

| 组合 | n_valid | AUROC(B) | PR-AUC(B) | ΔAUROC(vs A) |
|---|---:|---:|---:|---:|
| baseline | 955 | 0.726 ± 0.042 | 0.746 ± 0.035 | +0.242 |
| geo_wrist | 955 | 0.673 ± 0.024 | 0.686 ± 0.013 | -0.270 |
| geo_5pt | 955 | 0.842 ± 0.029 | 0.753 ± 0.027 | -0.115 |
| kinematic | 942 | 0.738 ± 0.028 | 0.716 ± 0.027 | +0.258 |
| geo+theta | 942 | 0.668 ± 0.016 | 0.687 ± 0.011 | -0.296 |
| geo+optical | 654 | 0.920 ± 0.018 | 0.832 ± 0.036 | -0.029 |
| all_fusion | 648 | 0.976 ± 0.007 | 0.956 ± 0.011 | -0.001 |
| geo+vt | 942 | 0.676 ± 0.019 | 0.688 ± 0.019 | N/A |
| geo+theta+vt | 942 | 0.677 ± 0.023 | 0.689 ± 0.022 | N/A |
| geo+optical+vt | 654 | 0.923 ± 0.020 | 0.820 ± 0.039 | N/A |

DeLong 检验（Exp-B）：
- geo+theta vs geo_wrist：p = 0.2211（ns）
- geo+theta vs geo+theta+vt：p = 0.0565（ns）
- geo+theta vs all_fusion：p = 0.0000（***）

跨场景迁移（Task B6）：

| 组合 | A->B zero-shot AUROC | B 内部5折 AUROC |
|---|---:|---:|
| geo_wrist | 0.673 | 0.673 ± 0.024 |
| geo+theta | 0.672 | 0.668 ± 0.016 |
| all_fusion | 0.786 | 0.976 ± 0.007 |

`all_fusion` 在 zero-shot 与域内训练之间存在明显差距（0.786 -> 0.976）。

5. 分组分析结果

Task B8 结果：
- speed：仅 `normal` 组有数据，AUROC = 0.668 ± 0.016
- lighting：仅 `normal` 组有数据，AUROC = 0.668 ± 0.016

`slow/fast` 与 `low/side` 当前无数据，未进行有效比较。

6. 结论

当前单受试者书写数据下，Exp-A 的 `geo+theta` 方案未保持原有性能。  
`v_t` 在当前样本中未显示显著增益。  
全特征融合 `all_fusion` 在 Exp-B 取得最高 AUROC。  
跨场景 zero-shot 与域内训练存在差距，主要体现在 `all_fusion`。
