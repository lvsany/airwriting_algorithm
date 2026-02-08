import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from hand_track.hand_writing_detector import HandWritingDetector

# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

# 从配置文件读取参数
trajectory_config = CONFIG['trajectory']
detector = HandWritingDetector()  # 使用配置文件中的默认参数
trajectory = []
MAX_TRAJECTORY_POINTS = trajectory_config['max_trajectory_points']
prev_index_pos = None
WRITING_MOTION_THRESHOLD = trajectory_config['writing_motion_threshold']
JITTER_THRESHOLD = trajectory_config['jitter_threshold']
history_points = []  # 用于存储历史轨迹点以计算速度和曲率

def smart_smooth(current_pos, prev_pos, history_points=history_points, max_window=None, min_window=None, fps=60):
    """自适应平滑函数，根据速度和曲率动态调整窗口大小和权重"""
    # 从配置文件读取平滑参数
    smoothing_config = CONFIG['trajectory']['smoothing']
    if max_window is None:
        max_window = smoothing_config['max_window']
    if min_window is None:
        min_window = smoothing_config['min_window']
    
    if prev_pos is None:
        history_points.clear()  # 重置历史点
        history_points.append(current_pos)
        return current_pos

    # 添加当前点到历史缓冲区
    history_points.append(current_pos)
    if len(history_points) > max_window:
        history_points.pop(0)

    # 计算速度（像素/秒）
    distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    speed = distance * fps  # 假设每帧时间为 1/fps 秒

    # 计算曲率（基于三点法）
    curvature = 0
    if len(history_points) >= 3:
        p1 = np.array(history_points[-3])
        p2 = np.array(history_points[-2])
        p3 = np.array(history_points[-1])
        v1 = p2 - p1
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 > 0 and norm_v2 > 0:
            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
            curvature = 1 - cos_theta  # 曲率越高，值越大（范围 [0, 2]）

    # 根据速度和曲率动态调整窗口大小
    speed_threshold_low = smoothing_config['speed_threshold_low']
    speed_threshold_high = smoothing_config['speed_threshold_high']
    curvature_threshold = smoothing_config['curvature_threshold']
    
    if speed > speed_threshold_high or curvature > curvature_threshold:
        window_size = min_window  # 快速或高曲率区域用小窗口
    elif speed < speed_threshold_low and curvature < curvature_threshold:
        window_size = max_window  # 慢速且低曲率区域用大窗口
    else:
        # 线性插值确定窗口大小
        speed_factor = (speed - speed_threshold_low) / (speed_threshold_high - speed_threshold_low)
        curvature_factor = curvature / curvature_threshold
        factor = max(speed_factor, curvature_factor)
        factor = np.clip(factor, 0, 1)
        window_size = int(min_window + (max_window - min_window) * (1 - factor))

    # 确保窗口大小不超过历史点数量
    window_size = min(window_size, len(history_points))
    # return current_pos
    # 计算加权移动平均
    if window_size > 1:
        points = np.array(history_points[-window_size:])
        # 权重根据时间衰减（最近的点权重更高）
        weights = np.exp(np.linspace(-1, 0, window_size))  # 指数衰减权重
        weights /= np.sum(weights)  # 归一化
        smoothed_x = int(np.sum(points[:, 0] * weights))
        smoothed_y = int(np.sum(points[:, 1] * weights))
        return (smoothed_x, smoothed_y)
    else:
        return current_pos

# from collections import deque

# # 可选：若你的工程里已有 scipy.signal，可用如下导入；
# # 若没有，也可以去掉 "双向" 那一步（但建议按当前工程依赖保留，之前你项目里已用 SciPy）

# from scipy.signal import savgol_filter
# _HAS_SG = True

# def smart_smooth(current_pos, prev_pos, history_points=history_points, max_window=None, min_window=None, fps=60):
#     """
#     自适应平滑（接口不变版本）：
#     - 角点/连笔保护：在高曲率或方向跃迁处，自动使用最小窗口或直接不过滤
#     - 短窗 Savitzky-Golay（默认二阶）提高“保峰/保拐点”能力
#     - 近似零相位：窗口内 forward-backward 两次 SG（需要 scipy），减少端点拖拽
#     - 速度/曲率/方向变化率三信号共同决定窗口大小与是否降级

#     返回：平滑后的 (x, y)，类型为 int（与原实现保持一致）
#     """
#     # ====== 配置读取（向后兼容） ======
#     smoothing_config = CONFIG['trajectory']['smoothing']
#     if max_window is None:
#         max_window = smoothing_config.get('max_window',  nine_or_odd(9))  # 9 是安全默认，见下方辅助
#     if min_window is None:
#         min_window = smoothing_config.get('min_window',  five_or_odd(5))  # 5 是安全默认

#     speed_threshold_low  = smoothing_config.get('speed_threshold_low',  100.0)  # px/s
#     speed_threshold_high = smoothing_config.get('speed_threshold_high', 400.0)  # px/s
#     curvature_threshold  = smoothing_config.get('curvature_threshold',  0.05)   # 与旧实现兼容

#     # 新增（可缺省，缺省时使用安全默认）
#     corner_kappa_high = smoothing_config.get('corner_kappa_high', 0.12)  # 角点曲率门（更敏感）
#     corner_omega_high = smoothing_config.get('corner_omega_high', 120.0) # 方向变化率(°/s)门
#     sg_poly           = int(smoothing_config.get('sg_poly', 2))          # SG 多项式阶数（2或3）
#     # 窗口需为奇数，且 >= sg_poly+2；下面会统一修正

#     # ====== 初始化历史 ======
#     if prev_pos is None:
#         history_points.clear()
#         history_points.append(tuple(current_pos))
#         return current_pos

#     history_points.append(tuple(current_pos))
#     # 防止历史无限增长
#     if len(history_points) > max_window:
#         history_points.pop(0)

#     # ====== 速度估计（px/s） ======
#     p_now  = np.array(current_pos, dtype=float)
#     p_prev = np.array(prev_pos,    dtype=float)
#     distance = np.linalg.norm(p_now - p_prev)
#     speed = distance * fps  # 假设帧间隔 1/fps

#     # ====== 方向变化率 / 曲率估计（稳健三点法 + 安全保护） ======
#     curvature = 0.0
#     omega_deg_per_s = 0.0   # 方向角速度（°/s），用来判定“拐点/大转向”
#     if len(history_points) >= 3:
#         p1 = np.array(history_points[-3], dtype=float)
#         p2 = np.array(history_points[-2], dtype=float)
#         p3 = p_now
#         v1 = p2 - p1
#         v2 = p3 - p2
#         n1 = np.linalg.norm(v1)
#         n2 = np.linalg.norm(v2)
#         if n1 > 1e-8 and n2 > 1e-8:
#             cos_theta = np.dot(v1, v2) / (n1 * n2)
#             cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
#             # 用 (1 - cos) 作为“曲率感”，与原实现保持风格一致
#             curvature = 1.0 - cos_theta
#             # 方向变化率：弧度/帧 -> 度/秒
#             theta = np.degrees(np.arccos(cos_theta))  # 本帧相邻两向量的夹角（度）
#             omega_deg_per_s = theta * fps

#     # ====== 角点门控（保护高曲率/大方向跃迁） ======
#     is_corner_like = (curvature >= corner_kappa_high) or (omega_deg_per_s >= corner_omega_high)

#     # ====== 基于速度/曲率的窗口自适应（连续区间线性插值） ======
#     # 注意：在角点/大转向时强制使用 min_window（或直接不过滤）
#     def _interp_window(speed, curvature):
#         if speed > speed_threshold_high or curvature > curvature_threshold:
#             w = min_window
#         elif speed < speed_threshold_low and curvature < curvature_threshold:
#             w = max_window
#         else:
#             # 线性插值（取两者中更"激进"的因子）
#             speed_factor = (speed - speed_threshold_low) / max(1e-6, (speed_threshold_high - speed_threshold_low))
#             curvature_factor = curvature / max(1e-6, curvature_threshold)
#             factor = float(np.clip(max(speed_factor, curvature_factor), 0.0, 1.0))
#             w = int(round(min_window + (max_window - min_window) * (1.0 - factor)))
#         return w

#     if is_corner_like:
#         window_size = min_window  # 角点：保细节，最小平滑
#     else:
#         window_size = _interp_window(speed, curvature)

#     # ====== 窗口形状修正：奇数 + 不小于 sg_poly+2 + 不超过历史长度 ======
#     def _fix_window(w):
#         w = int(max(w, sg_poly + 2))
#         if w % 2 == 0:
#             w += 1
#         w = min(w, len(history_points))
#         if w < 3:  # 太短不值得滤波
#             w = 1
#         return w

#     window_size = _fix_window(window_size)

#     # ====== 边界条件：窗口太小则直接回传 ======
#     if window_size <= 1 or not _HAS_SG:
#         # _HAS_SG 为 False 时，退化到“轻微指数平均”，但角点依然使用 min_window，从而较少抹角
#         if window_size <= 1:
#             return current_pos
#         pts = np.asarray(history_points[-window_size:], dtype=float)
#         # 轻度时间衰减权重（非常小的平滑，尽量不拖拽）
#         weights = np.linspace(0.6, 1.0, num=window_size)
#         weights = weights / np.sum(weights)
#         xy = np.sum(pts * weights[:, None], axis=0)
#         return (int(round(xy[0])), int(round(xy[1])))

#     # ====== Savitzky–Golay 短窗平滑（近似零相位：窗口内前后各一次） ======
#     pts = np.asarray(history_points[-window_size:], dtype=float)  # shape: (w, 2)

#     # 第一次 SG（中心窗），边界用插值模式
#     sg1 = savgol_filter(pts, window_length=window_size, polyorder=sg_poly, axis=0, mode='interp')
#     # 反向再来一次（forward-backward），获得近似零相位
#     sg2_rev = savgol_filter(sg1[::-1], window_length=window_size, polyorder=sg_poly, axis=0, mode='interp')
#     sg2 = sg2_rev[::-1]

#     # 角点附近再做一次“门控混合”：避免仍有轻微圆角
#     # 用当前点权重更高的线性混合原始与平滑（仅在检测到角点时）
#     smoothed_xy = sg2[-1]  # 取窗口末端（当前时刻）作为输出
#     if is_corner_like:
#         # alpha 越小，越靠近原始点（保护角点）
#         alpha = 0.25
#         smoothed_xy = alpha * smoothed_xy + (1 - alpha) * p_now

#     return (int(round(smoothed_xy[0])), int(round(smoothed_xy[1])))


# # ====== 小辅助：确保默认奇数窗口 ======
# def nine_or_odd(v):
#     v = int(v)
#     if v < 3: v = 3
#     if v % 2 == 0: v += 1
#     return v

# def five_or_odd(v):
#     v = int(v)
#     if v < 3: v = 3
#     if v % 2 == 0: v += 1
#     return v


# try:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         # frame = cv2.flip(frame, 1)
#         is_writing = detector.process(frame)
#         current_index_pos = detector.index_tip_position

#         if is_writing and current_index_pos != (0, 0):
#             if prev_index_pos is not None:
#                 smoothed_pos = smart_smooth(current_index_pos, prev_index_pos, history_points)
#                 prev_index_pos = smoothed_pos
#                 trajectory.append(smoothed_pos)
#                 # trajectory.append(current_index_pos)
#             else:
#                 trajectory.append(None)
#                 smoothed_pos = current_index_pos
#                 prev_index_pos = current_index_pos
#                 trajectory.append(smoothed_pos)

#             cv2.circle(frame, smoothed_pos, 10, (0, 0, 255), -1)
#         else:
#             prev_index_pos = None
#             history_points.clear() 

#         for i in range(1, len(trajectory)):
#             if trajectory[i-1] is not None and trajectory[i] is not None:
#                 cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

#         status_text = "Writing: YES" if is_writing else "Writing: NO"
#         status_color = (0, 255, 0) if is_writing else (0, 0, 255)
#         cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
#         cv2.putText(frame, f"Gesture Mode: {detector.gesture_mode}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow("Smart Finger Tracking", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('1'):
#             detector.gesture_mode = 1
#         elif key == ord('2'):
#             detector.gesture_mode = 2
#         elif key == ord('3'):
#             detector.gesture_mode = 3
#         elif key == ord('4'):
#             detector.gesture_mode = 4
#         elif key == ord('c'):
#             trajectory = []
#             prev_index_pos = None
#             history_points.clear()
#         elif key == ord('q'):
#             break

# finally:
#     cap.release()
#     cv2.destroyAllWindows()