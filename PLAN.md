# 手掌书写系统 —— 后续研究计划

## 一、研究定位与核心问题

### 1.1 系统现状

当前系统（Block A）已实现：

| 模块 | 实现状态 | 局限 |
|------|---------|------|
| 双手检测与角色分配 | ✅ 完成 | 依赖位置规则，运动模式分配未激活 |
| RANSAC 手掌平面拟合 | ✅ 完成 | 仅用3点(0,5,17)，侧面角度退化 |
| 接触检测（距离阈值 + Otsu自适应） | ✅ 完成 | **核心瓶颈**：MediaPipe z轴非度量深度 |
| 轨迹平滑与分割 | ✅ 完成 | — |
| 文字识别接口 | ⚠️ 未激活 | — |

### 1.2 核心研究问题

> 在手掌书写场景下，如何从**单目 RGB 视频**中，仅依赖手部关键点运动学特征与局部外观特征，**实时**检测手指与手掌（软组织曲面）的接触状态？

**为什么这是难题**：MediaPipe 的 z 坐标是归一化相对深度（以手腕为参考的比例估计），在指尖距手掌 < 15mm 的极近距离下，深度误差量级与接触信号量级相当，纯距离阈值方法不可靠。

### 1.3 研究价值定位

社会可接受性驱动：相比空中书写，手掌书写具备：
- **低疲劳**：手臂有物理支撑，肌肉持续张力降低
- **低可见度**：动作幅度小，适合会议、公共场所等需要低调输入的场景
- **隐私保护**：书写内容不暴露给旁观者
- **触觉反馈**：物理接触提供本体感觉，改善书写精度

---

## 二、接触检测技术路线

### 2.1 当前系统缺陷分析

```
现有链路：
MediaPipe z(t) → 距离 d(t) → Otsu 阈值 → WRITING / IDLE

问题：
1. z 轴误差 ~5-15mm，接触距离阈值也在 5-15mm → 信噪比 ≈ 1
2. Otsu 假设双峰分布，单一场景下可能退化为单峰
3. 无法区分"手指静止悬停在手掌上方"与"手指轻触手掌"
```

### 2.2 技术方向 A：运动时序特征（主线）

**物理直觉**：手指接触软组织时，法向速度骤降为 0，而切向书写速度继续存在，形成独特的运动签名。

#### 特征工程

| 特征符号 | 含义 | 计算方式 | 接触时行为 |
|---------|-----|---------|-----------|
| `d(t)` | 指尖到平面距离 | `PalmCoordinateSystem.get_distance_to_plane()` | 趋近并稳定 |
| `v_n(t) = Δd/Δt` | 法向速度 | 距离一阶差分 | 从负值→趋近0 |
| `a_n(t) = Δv_n/Δt` | 法向加速度 | 距离二阶差分 | 接触瞬间出现负峰值 |
| `v_t(t)` | 切向速度（2D像素） | 指尖在手掌坐标系xy平面的速度 | 接触后仍 > 0（书写运动） |
| `σ_d(t)` | 距离滑窗方差 | 3帧滚动标准差 | 接触时低，悬停时高（抖动） |
| `θ(t)` | 手指-法线夹角 | MCP→指尖向量与平面法线的夹角 | 接触时趋近90° |

#### 判决规则（初步）

```
WRITING ← d ≤ τ_d  AND  |v_n| < τ_vn  AND  v_t > τ_vt
```

相比纯距离阈值，增加"法向速度稳定"条件可消除"经过式触碰"误报。

#### 实现位置

`src/hand_track/contact_state_machine.py` → 扩展 `_next_state()` 方法

```python
def _next_state(self, d: float, v_t: float) -> ContactState:
    v_n = self._calc_normal_velocity()   # Δd/Δt
    a_n = self._calc_normal_accel()      # Δv_n/Δt
    
    dist_ok      = d <= self.writing_threshold
    stable       = abs(v_n) < self.vn_threshold    # 不再趋近
    has_motion   = v_t > self.vt_threshold          # 在书写
    
    if dist_ok and stable:
        return ContactState.WRITING
    return ContactState.IDLE
```

### 2.3 技术方向 B：局部外观特征（辅助/消融）

**物理直觉**：手指接触手掌时，指腹受压形变在接触区域产生局部阴影；接触点光学流场趋近于零（相对手掌运动）。

#### 特征提取

```python
# src/hand_track/appearance_contact_detector.py（新建）
class AppearanceContactDetector:
    def extract(self, frame, contact_point_2d, prev_frame, radius=18):
        x, y = contact_point_2d
        roi = frame[y-radius:y+radius, x-radius:x+radius]
        
        # 特征1：局部梯度能量（接触产生阴影边缘）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        shadow_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 特征2：局部光学流（接触区域相对手掌运动≈0）
        prev_roi = prev_frame[y-radius:y+radius, x-radius:x+radius]
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_mag = np.mean(np.linalg.norm(flow, axis=2))
        
        return {'shadow': shadow_score, 'flow': flow_mag}
```

**接触投影点计算**：在 `DualHandDetector.process()` 中，已有 `palm_sys.project_to_plane(tip_3d)` → 转换为像素坐标即为 ROI 中心。

### 2.4 融合策略

```
阶段一（快速验证）：规则融合
    score = w1 * dist_norm + w2 * (1 - vn_norm) + w3 * shadow_norm
    WRITING if score > threshold

阶段二（若规则不够）：轻量分类器
    特征向量: [d, v_n, a_n, v_t, σ_d, shadow, flow]  →  SVM / 逻辑回归
    要求：推理延迟 < 1ms（CPU上）

阶段三（HCI论文级别）：时序模型
    输入: 滑窗内 10 帧特征序列  →  轻量LSTM  →  接触概率
```

---

## 三、调研实验路线（先验证，再实现）

实验分两阶段递进：**先用纯接触实验验证信号有效性**，再进入书写场景验证完整系统

### 阶段一：纯接触检测实验（不涉及书写）

#### Exp-A1：纯 Tap 数据收集与标注

**目标**：在受控条件下验证 `v_n`、`σ_d` 等核心信号是否可区分接触与非接触

**采集方式**：
```
受试者用食指在手掌上反复做"点触-抬起"动作（tap），不做书写
→ 每次 tap 持续约 0.5-1 秒，重复 50-100 次
→ 可控变量：接触速度（慢/快）、接触力度（轻/重）、手掌位置（中心/边缘）
```

**标注方式一（自动，推荐用于 Exp-A1）：荧光贴纸颜色检测**：
```
在手掌固定位置贴荧光色贴纸（~1cm²）
→ 受试者每次 tap 均点触贴纸位置（纯 tap 实验中位置可控）
→ 指尖遮挡贴纸时 HSV 颜色面积骤降 → 自动生成 contact_label
→ 实现：scripts/auto_label.py 逐帧检测贴纸颜色面积，阈值触发标注
→ 误差 < 1 帧，完全自动，无需人工介入
注意：此方法仅适用于 Exp-A1（tap 位置固定），
      Exp-B1（书写全掌）中手指不固定在贴纸位置，需改用方式二
```

**标注方式二（半自动，用于 Exp-B1 书写场景）：键盘自标注**：
```
受试者在接触的同时按住键盘空格键，抬起时松开
→ 键盘事件时间戳直接作为 contact_label
→ 误差 < 2 帧，无需手动逐帧标注
→ 实现：在 feature_logger.py 中监听键盘事件并写入 CSV
```

**标注格式**（CSV）：
```
frame_id, timestamp, contact_label, dist_raw, landmark_21x3, ...
```

**采集场景**：
- 光照：室内标准光
- 受试者：先用自己数据验证，结论正向后扩展至 3-5 人

#### Exp-A2：纯接触特征统计分析（运动学 + 外观双线）

在 Exp-A1 数据上**同时**提取运动学特征和外观特征，绘制接触事件前后 ±10 帧的时序曲线：

```python
# scripts/analyze_features.py（新建）
for event in contact_events:
    window = features[event-10 : event+10]
    # 运动学特征
    plot(window['d'], window['v_n'], window['sigma_d'])
    # 外观特征（同帧提取）
    plot(window['shadow_score'], window['flow_mag'])
```

**运动学特征期望结论**：
- `v_n` 在接触前 ~3 帧急降，接触后维持 ≈ 0
- `σ_d` 接触时显著低于悬停时（接触稳定，悬停抖动）
- `dist` 趋势可用但绝对值有噪声（验证当前系统局限）

**外观特征期望结论**：
- `shadow_score`（局部梯度能量）接触时升高——指腹压变形产生阴影边缘
- `flow_mag`（光学流幅值）接触时趋近 0——接触区域相对手掌无相对运动
- 若两者在 Exp-A1 中变化不显著，说明外观特征对纯 tap 场景贡献有限，需到书写场景再验证

#### Exp-A3：纯接触场景的 ROC 曲线（运动学 + 外观全对比）

| 特征组合 | 验证目标 |
|---------|---------|
| 仅 `d`（current baseline） | 当前系统上界 |
| `d + v_n` | 法向速度核心贡献 |
| `d + σ_d` | 稳定性特征贡献 |
| `d + v_n + σ_d` | 运动学主方案 |
| `d + shadow` | 外观特征（阴影）单独贡献 |
| `d + flow` | 外观特征（光学流）单独贡献 |
| `d + shadow + flow` | 外观特征联合贡献 |
| `d + v_n + σ_d + shadow + flow` | 运动学 + 外观融合上界 |

**两个独立决策点**：
- 运动学主方案 AUC ≥ 0.85 → 阶段二运动学路线成立
- 外观特征是否带来 AUC 显著提升（>0.03）→ 决定融合策略是否值得

---

### 阶段二：书写场景实验（基于阶段一结论）

#### Exp-B1：书写数据收集

**与纯接触实验的关键区别**：书写时接触后手指有侧向移动（`v_t > 0`），且指腹与手掌的相对运动使光学流特征行为发生变化——书写时接触区域的 `flow_mag` 不再为 0，需重新评估外观特征在书写场景的有效性。

**采集方式**：
```
受试者在手掌上书写单字和连续词（CASIA-HWDB 常用字）
→ 标注方式：键盘空格键自标注
→ 采集量：每人约 20-30 分钟书写视频
→ 受试者：5-10 人
```

**采集场景**：
- 光照：室内标准光、低光（<100 lx）、侧光
- 书写速度：慢速、正常速、快速

#### Exp-B2：书写场景全特征 ROC 与消融实验

在 Exp-B1 数据上重跑 Exp-A3 的全部特征组合对比，重点关注：

| 特征组合 | 新增验证目标 |
|---------|------------|
| `d + v_n + σ_d + v_t` | `v_t` 在书写场景的贡献（区分悬停接触 vs 书写接触） |
| `d + v_n + σ_d + shadow` | 阴影特征在书写场景是否仍有效（手部遮挡可能影响光照） |
| `d + v_n + σ_d + flow` | 光学流在书写场景是否退化（书写时指腹相对手掌有横向移动） |
| `d + v_n + σ_d + shadow + flow + v_t` | 全特征融合上界 |

**跨场景对比**：将 Exp-A3 和 Exp-B2 的同一特征组合 AUC 并排展示，观察哪些特征从 tap 场景到书写场景出现退化（尤其是 `flow_mag`）。

### Exp-C：延迟与实时性评估

在 60fps 视频上分别测量两类特征的处理延迟：
- 运动学特征（`v_n`、`σ_d`、`v_t`）：目标 < 1ms/frame（纯数值计算）
- 外观特征（`shadow`、`flow`）：目标 < 5ms/frame（含 ROI 裁剪 + 光学流计算）
- 端到端总延迟：目标 < 5ms/frame

---

## 四、实现优先级与工作量估计

### Phase 1：特征扩展（约 2-3 天）

- [ ] `ContactStateMachine`: 新增 `_calc_normal_velocity()`, `_calc_normal_accel()`
- [ ] `DualHandDetector`: 传入 `v_t`（切向速度）到状态机
- [ ] 新增 `feature_logger.py`：将每帧特征记录到 CSV，供 Exp-B/C 使用
- [ ] `config.yaml`: 新增 `vn_threshold`, `vt_threshold` 参数

### Phase 2a：纯接触实验（约 2-3 天）

- [ ] `scripts/feature_logger.py`：新增键盘事件监听，空格键标注接触时刻
- [ ] 录制纯 tap 数据（自己先录，约 30 分钟）
- [ ] `scripts/analyze_features.py`：绘制特征时序曲线与 ROC 曲线（Exp-A2/A3）
- [ ] **决策点**：AUC ≥ 0.85 → 进入 Phase 2b；否则重新评估技术路线

### Phase 2b：书写场景实验（约 3-5 天，仅在 Phase 2a 结论正向后执行）

- [ ] 录制书写数据（5-10 人，含标注）
- [ ] 重跑 ROC 实验，验证 `v_t` 和外观特征的额外贡献（Exp-B2）

### Phase 3：接触检测改进（约 2-3 天，基于实验结论）

- [ ] 根据 Exp-C 结论选择最优特征组合
- [ ] 改造 `_next_state()` 为多特征判决
- [ ] 若需要外观特征：新建 `appearance_contact_detector.py`
- [ ] 若需要分类器：训练 SVM，导出为 joblib，推理集成到状态机

### Phase 4：遮挡鲁棒性（约 3-5 天）

**问题**：书写手可能遮挡画布手关键点，导致平面拟合失败

**方案**：
- 关键点丢失时，用上一帧平面参数 + 卡尔曼预测 plane normal 的运动趋势
- `PalmPlaneTracker` 增加平滑历史缓存，丢失时降级为 last-known

### Phase 5：用户实验与评估（约 1-2 周）

见第五节。

---

## 五、JCST 论文结构与实验要求

JCST 为计算机技术类期刊，滚动投稿，评审侧重**技术贡献 + 定量验证**，不要求大规模用户实验。核心实验为接触检测准确性评估与系统性能测试。

### 5.1 论文正文结构（目标 ~18 页）

```
1. Introduction（~1.5 页）
   - 手掌书写的动机：低疲劳、低可见度、隐私保护
   - 单目 RGB 接触检测的技术挑战（z 深度不可靠）
   - 贡献声明（3 点）

2. Related Work（~2 页）
   - 手部追踪（MediaPipe、WiLoR）
   - 接触检测（PressureVision++、生物阻抗方法）
   - 空中书写与手掌书写系统

3. System Overview（~1 页）
   - 系统架构图（双手检测 → 平面拟合 → 接触检测 → 轨迹处理）
   - 数据流与模块关系

4. Palm Coordinate System（~2 页）
   - RANSAC 平面拟合（参考点选取、鲁棒性分析）
   - 手掌局部坐标系构建与动态补偿

5. Contact Detection Method（~3 页）  ← 核心贡献
   - 现有方法局限：MediaPipe z 深度的噪声分析
   - 多特征框架：d、v_n、σ_d 的物理含义与计算
   - 多条件判决规则与自适应阈值（Otsu）
   - 与纯距离阈值方法的对比分析

6. Experiments（~4 页）
   见 5.2 节

7. Discussion（~1 页）
   - 方法局限：遮挡场景、极端光照
   - 与双目方案的对比

8. Conclusion（~0.5 页）
```

### 5.2 实验设计（JCST 标准）

#### Exp-1：接触检测准确性（核心，必须）

**数据**：Exp-A1 + Exp-B1 的标注数据（5-8 人）

**指标**：
- Precision / Recall / F1（接触帧级别）
- 误触发率 FPR：IDLE 状态下单位时间误判次数
- 接触起止时延（ms）：从实际接触到系统判定的延迟

**对比 baseline**：

| 方法 | P | R | F1 | FPR |
|-----|---|---|----|-----|
| 纯距离阈值（固定） | — | — | — | — |
| 距离 + Otsu 自适应（当前系统） | — | — | — | — |
| **本文：d + v_n + σ_d（多特征）** | — | — | — | — |

#### Exp-2：消融实验（核心，必须）

逐特征移除，验证每个特征的贡献：

| 特征组合 | F1 | 说明 |
|---------|-----|------|
| `d` only | — | baseline |
| `d + v_n` | — | 加法向速度 |
| `d + σ_d` | — | 加稳定性 |
| `d + v_n + σ_d` | — | 完整方案 |

#### Exp-3：系统性能测试（必须）

- 端到端延迟：60fps 下每帧处理时间（目标 < 5ms）
- 不同光照条件下的 F1 退化（标准光 / 低光 / 侧光）
- 不同书写速度下的 F1（慢 / 正常 / 快）

#### Exp-4：小规模用户书写演示（可选，增强说服力）

**不做统计检验**，仅作为系统可用性的定性说明：
- 5-8 人书写 CASIA-HWDB 常用字 30 个
- 报告平均 CER 与书写流畅度主观评分（1-5 分）
- 作用：证明接触检测改进对实际书写任务有正向影响

### 5.3 贡献声明（三点）

```
1. 提出基于运动学多特征（d + v_n + σ_d）的手掌接触检测框架，
   解决单目 RGB 下 MediaPipe z 深度不可靠导致的误判问题

2. 构建手掌局部坐标系与动态补偿机制，
   支持手掌自由移动下的稳定书写坐标提取

3. 在手掌书写场景下系统验证上述方法，
   相比纯距离阈值基线 F1 提升 XX%，误触发率降低 XX%
```

---

## 六、遮挡与极端场景处理

### 6.1 已知失效场景

| 场景 | 当前行为 | 改进方案 |
|-----|---------|---------|
| 书写手从上方经过画布手 | 平面拟合失败，接触判断为 None | Last-known 平面 + 置信度衰减 |
| 侧面角度（>60°） | RANSAC 用3点退化 | 增加备用关键点 [1, 9, 13] |
| 快速书写（> 1m/s） | 深度抖动增大 | 降低 `temporal_smoothing_window` 对速度自适应 |
| 手掌皮肤过暗 / 强背光 | MediaPipe 置信度下降 | 外观特征降级，纯运动学接管 |

### 6.2 平面拟合鲁棒性改进

```python
# palm_coordinate_system.py 改进点
# 当前：仅用 [0, 5, 17] 3个点
# 改进：用所有掌骨根部关键点 [0, 1, 5, 9, 13, 17]，提高 RANSAC 可用点数
reference_landmarks: [0, 1, 5, 9, 13, 17]
```

---

## 七、投稿策略

### 目标期刊：JCST（Journal of Computer Science and Technology）

- **定位**：计算机技术领域综合性期刊，Springer 出版，中科院计算所主办
- **投稿方式**：滚动投稿，无固定截止日期
- **审稿周期**：通常 3-6 个月
- **页数要求**：正文 15-20 页（双栏格式约 8000-12000 字）
- **评审重点**：技术新颖性、实验充分性、系统实用性

### 为什么 JCST 适合这篇论文

1. **技术贡献匹配**：接触检测算法改进 + 手掌坐标系是计算机技术范畴，符合 JCST 定位
2. **不要求大规模用户实验**：5-8 人的技术评估实验足够，无需 N≥12 的统计检验
3. **滚动投稿**：允许按实际完成节奏投稿，无需赶固定 DDL
4. **中文研究社区**：编委会熟悉手写输入方向，审稿背景友好

### 备选期刊（若 JCST 退稿）

| 期刊 | 额外要求 |
|-----|---------|
| IEEE THMS | 需要更完整的用户实验 |
| ACM IMWUT | 需要普适计算场景扩展 |
| Pattern Recognition | 需要与更多识别方法对比 |

---

## 八、里程碑时间线（技术实现与论文写作并行）

策略：每完成一个技术模块，立即写对应论文章节；实验数据出来后直接填表。

```
Week 1（4/23 - 4/30）
  【实现】
  ├── ContactStateMachine 新增 v_n、σ_d 特征提取
  ├── feature_logger.py：逐帧记录特征 + 键盘事件标注接触时刻
  └── 录制纯 tap 数据（Exp-A1，自己 + 1-2 人，约 1 小时）
  【写论文】
  ├── Introduction 初稿（动机、挑战、贡献声明占位）
  └── System Overview + Palm Coordinate System 章节（基于已有代码）

Week 2（5/1 - 5/7）
  【实验】
  ├── analyze_features.py：特征时序曲线 + ROC（Exp-A2/A3）
  ├── 【决策点】AUC ≥ 0.85 → 继续；否则重新评估
  └── 录制书写数据（Exp-B1，3-5 人，含标注）
  【实现】
  └── 基于 ROC 结论改造 _next_state()（多特征判决）
  【写论文】
  └── Contact Detection Method 章节（方法描述 + 与 baseline 对比分析）

Week 3（5/8 - 5/14）
  【实验】
  ├── Exp-1：接触检测 Precision / Recall / F1（填论文表格）
  ├── Exp-2：消融实验（填论文表格）
  └── Exp-3：延迟测试 + 光照/速度鲁棒性
  【实现】
  └── 平面拟合关键点扩展 [0,1,5,9,13,17]（鲁棒性改进）
  【写论文】
  └── Experiments 章节（边跑实验边写结果）

Week 4（5/15 - 5/21）
  【实验】
  └── Exp-4：小规模书写演示（5-8 人，CER + 主观评分）
  【写论文】
  ├── Related Work 章节
  ├── Discussion + Conclusion
  └── Abstract 最终版（贡献数字填入）

Week 5（5/22 - 5/28）
  ├── 全文通读与修改
  ├── 图表整理（系统架构图、特征时序图、ROC 曲线、消融表格）
  └── 投稿
```

---

## 附录：关键文件索引

| 文件 | 职责 |
|-----|-----|
| `src/hand_track/contact_state_machine.py` | **核心改进点**：特征提取 + 状态判决 |
| `src/hand_track/dual_hand_detector.py` | 双手协调 + 接触检测调度 |
| `src/hand_track/palm_coordinate_system.py` | 手掌坐标系 + 平面拟合（RANSAC） |
| `src/config.yaml` | 所有阈值参数（接触检测、自适应Otsu） |
| `src/hand_track/appearance_contact_detector.py` | **待新建**：外观特征提取 |
| `scripts/auto_label.py` | **待新建**：颜色检测自动标注 |
| `scripts/analyze_features.py` | **待新建**：特征统计分析与ROC |
| `scripts/feature_logger.py` | **待新建**：运行时特征记录到CSV |