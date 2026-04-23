# 项目日程表 (Project Schedule)

模式: 开发与实验并行 (Development & Experimentation)  
起止: 2026-02-01 至 2026-02-28  
备注: 缩短开发周期至 1 周，增加实验与论文写作时间。

---

### Phase 1: 系统开发 (System Development)

Total Time: 02-01 (周日) — 02-07 (周六)

#### Block A: 核心功能实现 (Core Implementation)

时间: 02-01 — 02-03 (3 Days)

- 目标: 完成双人手检测、坐标投影和接触判定。
- 并行任务:
  1. Engine: MediaPipe 双手配置与 ID 锁定。
  2. Math: dynamic_canvas.py 实现平滑拟合与坐标变换。
  3. Interaction: 实现接触检测与状态机。
- 交付物: 可用的手掌书写原型系统。

#### Block B: 算法与工具完善 (Algorithm & Tools)

时间: 02-04 — 02-07 (4 Days)

- 目标: 完成 SRIF 滤波、无限画布功能及实验工具。
- 并行任务:
  1. Algo: 实现 SRIF/Kalman 预测处理遮挡。
  2. UI: 实现 Sliding Window 交互。
  3. Tools: 开发 ExperimentLogger 支持数据录制。
- 交付物: 包含抗遮挡功能的系统及实验数据采集工具。

---

### Phase 2: 系统优化 (Spring Festival & System Optimization)

Total Time: 02-08 (周日) — 02-15 (周日)

- 任务: 系统优化: 02-08 — 02-15 完成代码优化、文档整理，可以写写论文，设计一下实验方案

---

### Phase 3: 实验验证 (Experimentation)

Total Time: 02-20 (周五) — 02-26 (周四)
春节后集中进行实验，增加样本量，进行详细数据采集。

#### Exp Cycle 1: 系统测试与参数调整 (System Testing)

时间: 02-20 — 02-22 (3 Days)

- 任务:
  1. 消融数据: 采集 50 组遮挡数据验证算法。
  2. 边界测试: 测试系统在低光照和快速移动下的表现。
  3. 参数固定: 确定所有算法参数。

#### Exp Cycle 2: 用户研究 (User Study)

时间: 02-23 — 02-26 (4 Days)

- 任务:
  1. 样本采集: 招募 10-12 名受试者。
  2. 对照实验: 执行 Air/Palm/Tablet 三组对比实验。
  3. 问卷收集: 收集 NASA-TLX 和 UEQ 问卷。

---

### Phase 4: 论文写作 (Paper Writing)

Total Time: 02-27 (周五) — 02-28 (周六) + 后续
