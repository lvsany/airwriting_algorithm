# 开发日志

2026-02-08

完成Block A核心功能。

实现内容：

palm_coordinate_system.py
建立手掌坐标系。RANSAC算法拟合平面，100次迭代，10mm阈值，使用手腕、食指根、小指根三点。3D点转换到手掌局部2D坐标。边界检测使用21个关节点。

dual_hand_detector.py
MediaPipe双手检测，max_num_hands=2。角色分配：书写手和画板手。根据config配置分配角色。食指尖坐标投影到手掌平面。边界内才记录轨迹。

contact_state_machine.py
5状态机：空闲→悬停→触碰→书写→抬起。
阈值：100mm空闲，30mm悬停，15mm触碰，25mm抬起。
3帧中值滤波防抖。速度阈值50px/s区分书写和触碰。

test_main.py
摄像头实时测试。半透明青色显示手掌区域。RGB坐标轴可视化。调试信息显示帧率、角色、状态、距离。ESC/Q退出，S保存，C清空。边界外红色警告。

config.yaml
新增palm_writing配置节。enabled开关、dominant_hand设置、plane_fitting参数、contact_detection阈值、boundary_detection边距。

修复问题：

1. AttributeError: prev_normal - 删除Block B残留代码
2. NameError: confidence - 改为固定返回1.0
3. MediaPipe标签反转 - 交换"Left"/"Right"映射
4. 边界外书写 - 添加is_within_palm_boundary检查
5. config.yaml重复键 - 删除重复的min_writing_speed和max_hover_speed

代码精简：

palm_coordinate_system.py: 230→111行（-52%）
contact_state_machine.py: 230→132行（-43%）
dual_hand_detector.py: 315→165行（-48%）
总计: 2000→1183行（-41%）

方法：删除docstring，缩短变量名，简化参数名，合并代码。

技术细节：

RANSAC: 100次迭代，3点拟合，10mm内点阈值
状态机: 100/30/15/25mm四级阈值，3帧滑动窗口
坐标系: 原点手腕，X轴手腕→食指根，Y轴垂直平面，4x4变换矩阵
配置: dominant_hand="right"表示右手书写左手画板

测试结果：
双手检测正常，平面拟合准确，状态转换稳定，边界检测有效，30fps以上。

当前限制：
position模式固定角色分配。无motion模式动态切换。无manual模式手动选择。无平面置信度评估。无轨迹平滑。

依赖库：
MediaPipe Hands, NumPy, OpenCV, Python 3.8+