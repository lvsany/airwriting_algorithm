"""
实时摄像头测试 - Block A 核心功能验证
实时显示双手检测、手掌平面拟合和接触检测状态
"""

import cv2
import numpy as np
import sys
import os
import yaml
import time
from pathlib import Path

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_track.dual_hand_detector import DualHandDetector
from hand_track.finger_tracking import HandWritingDetector, smart_smooth
from utils.online_trace_converter import OnlineTraceConverter

# 加载配置
with open("./config.yaml", 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)


def draw_palm_plane_visualization(frame, detector):
    """绘制手掌虚拟平面和坐标系"""
    if not hasattr(detector, 'palm_tracker'):
        return
    
    palm_system = detector.palm_tracker.get_current_system()
    if palm_system is None:
        return
    
    # 获取画布手关键点用于投影
    if detector.palm_lm is None:
        return
    
    h, w = frame.shape[:2]
    
    # 将3D坐标投影到2D屏幕（简化投影，实际应用透视变换）
    def project_3d_to_2d(point_3d):
        # MediaPipe坐标：x,y已归一化[0,1]，z是相对深度
        # 简单映射到屏幕坐标
        x = int(point_3d[0] * w)
        y = int(point_3d[1] * h)
        return (x, y)
    
    # 绘制手掌平面轮廓（使用手掌关键点）
    palm_landmarks = detector.palm_lm.landmark
    
    # 手掌四个角点：手腕(0)、食指根(5)、中指根(9)、小指根(17)
    corners_idx = [0, 5, 9, 17]
    palm_points_2d = []
    for idx in corners_idx:
        lm = palm_landmarks[idx]
        palm_points_2d.append(project_3d_to_2d([lm.x, lm.y, lm.z]))
    
    # 绘制半透明手掌区域
    overlay = frame.copy()
    palm_poly = np.array(palm_points_2d, dtype=np.int32)
    cv2.fillPoly(overlay, [palm_poly], (0, 255, 255))  # 青色
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # 绘制手掌边界
    cv2.polylines(frame, [palm_poly], True, (0, 200, 200), 2)
    
    # 绘制坐标系（从手腕发出的三个轴）
    origin_lm = palm_landmarks[0]
    origin_2d = project_3d_to_2d([origin_lm.x, origin_lm.y, origin_lm.z])
    
    # 计算轴的终点（放大显示）
    axis_length = 0.1  # 10cm在归一化坐标中
    
    # U轴（红色，手掌横向）
    u_end = palm_system.origin + palm_system.u_axis * axis_length
    u_end_2d = project_3d_to_2d([u_end[0], u_end[1], u_end[2]])
    cv2.arrowedLine(frame, origin_2d, u_end_2d, (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(frame, 'X', u_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # V轴（绿色，手掌纵向）
    v_end = palm_system.origin + palm_system.v_axis * axis_length
    v_end_2d = project_3d_to_2d([v_end[0], v_end[1], v_end[2]])
    cv2.arrowedLine(frame, origin_2d, v_end_2d, (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(frame, 'Y', v_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # W轴（蓝色，手掌法向）
    w_end = palm_system.origin + palm_system.w_axis * axis_length
    w_end_2d = project_3d_to_2d([w_end[0], w_end[1], w_end[2]])
    cv2.arrowedLine(frame, origin_2d, w_end_2d, (255, 0, 0), 3, tipLength=0.3)
    cv2.putText(frame, 'Z', w_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 标注原点
    cv2.circle(frame, origin_2d, 6, (255, 255, 0), -1)
    cv2.putText(frame, 'Origin', (origin_2d[0]+10, origin_2d[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


def draw_debug_info(frame, detector, is_writing, fps, palm_mode=True):
    """在帧上绘制调试信息"""
    h, w = frame.shape[:2]
    
    # 背景半透明面板
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if palm_mode:
        # 双手模式信息
        cv2.putText(frame, "Mode: Palm Writing", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 手部角色
        left_role = detector.left_role.value if detector.left_role else "None"
        right_role = detector.right_role.value if detector.right_role else "None"
        cv2.putText(frame, f"Left: {left_role}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        cv2.putText(frame, f"Right: {right_role}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        # 接触状态
        state = detector.contact_sm.state.value
        state_color = {
            'idle': (100, 100, 100),
            'hovering': (0, 255, 255),
            'touching': (0, 165, 255),
            'writing': (0, 255, 0),
            'lifted': (255, 0, 255)
        }
        color = state_color.get(state, (255, 255, 255))
        cv2.putText(frame, f"State: {state.upper()}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 距离到平面
        if detector.dist_palm is not None:
            cv2.putText(frame, f"Distance: {detector.dist_palm:.1f}mm", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        else:
            cv2.putText(frame, "Distance: Out of Boundary", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        # 单手模式信息
        cv2.putText(frame, "Mode: Air Writing", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        status = "WRITING" if is_writing else "IDLE"
        color = (0, 255, 0) if is_writing else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 操作提示
    cv2.putText(frame, "ESC: Exit | S: Save | C: Clear", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    """主函数"""
    print("=" * 60)
    print("Block A 实时测试 - 双手掌书写检测")
    print("=" * 60)
    
    # 初始化摄像头
    cap = cv2.VideoCapture("test.mp4")
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['video']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['video']['height'])
    cap.set(cv2.CAP_PROP_FPS, CONFIG['video']['fps'])
    
    # 初始化检测器
    palm_mode_enabled = CONFIG.get('palm_writing', {}).get('enabled', False)
    
    if palm_mode_enabled:
        detector = DualHandDetector()
        print("✓ 使用双手检测模式（手掌书写）")
    else:
        detector = HandWritingDetector(gesture_mode=CONFIG['hand_detection']['gesture_mode'])
        print("✓ 使用单手检测模式（空中书写）")
    
    # 初始化轨迹管理
    online_trace_converter = OnlineTraceConverter(smoothing_window=CONFIG['online_trace']['smoothing_window'])
    trajectory = []
    prev_index_pos = None
    history_points = []
    prev_is_writing = False
    
    # 创建画布用于绘制轨迹
    canvas = np.ones((CONFIG['video']['height'], CONFIG['video']['width'], 3), dtype=np.uint8) * 255
    
    # FPS计算
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0.0
    
    # 输出目录
    output_dir = "./results/test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n控制说明:")
    print("  ESC 或 Q - 退出程序")
    print("  S - 保存当前轨迹")
    print("  C - 清空轨迹")
    print("\n开始检测...\n")
    
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头帧")
                break
            
            # 计算时间戳
            timestamp = time.time() - start_time
            
            # 处理手势检测
            if palm_mode_enabled:
                is_writing = detector.process(frame, timestamp)
                current_index_pos = detector.get_screen_position()
            else:
                is_writing = detector.process(frame)
                current_index_pos = detector.index_tip_position
            
            # 检测状态变化（笔画分割）
            if prev_is_writing and not is_writing:
                # 抬笔
                valid_points = [pt for pt in trajectory if pt is not None]
                if len(valid_points) > 5:
                    online_trace_converter.end_trace()
                    print(f"✓ 笔画完成: {len(valid_points)} 个点")
                trajectory.clear()
                prev_index_pos = None
                history_points.clear()
            elif not prev_is_writing and is_writing:
                # 落笔
                trajectory.clear()
                prev_index_pos = None
                history_points.clear()
                print("✏ 开始新笔画")
            
            prev_is_writing = is_writing
            
            # 记录轨迹点
            if is_writing and current_index_pos != (0, 0):
                if prev_index_pos is not None:
                    # 平滑处理
                    smoothed_pos = smart_smooth(current_index_pos, prev_index_pos, history_points)
                    prev_index_pos = smoothed_pos
                    trajectory.append(smoothed_pos)
                    online_trace_converter.add_point(smoothed_pos[0], smoothed_pos[1])
                    
                    # 在画布上绘制轨迹
                    if len(trajectory) > 1 and trajectory[-2] is not None:
                        cv2.line(canvas, trajectory[-2], smoothed_pos, (0, 0, 0), 3)
                else:
                    trajectory.append(None)
                    smoothed_pos = current_index_pos
                    prev_index_pos = current_index_pos
                    trajectory.append(smoothed_pos)
                    online_trace_converter.add_point(smoothed_pos[0], smoothed_pos[1])
                
                # 在视频帧上显示当前书写点
                cv2.circle(frame, smoothed_pos, 8, (0, 0, 255), -1)
            else:
                prev_index_pos = None
                history_points.clear()
            
            # 计算FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0
            
            # 绘制手掌虚拟平面（仅双手模式）
            if palm_mode_enabled:
                draw_palm_plane_visualization(frame, detector)
            
            # 绘制调试信息
            draw_debug_info(frame, detector, is_writing, fps, palm_mode_enabled)
            
            # 混合显示画布和视频帧
            canvas_resized = cv2.resize(canvas, (frame.shape[1] // 3, frame.shape[0] // 3))
            frame[10:10+canvas_resized.shape[0], frame.shape[1]-10-canvas_resized.shape[1]:frame.shape[1]-10] = canvas_resized
            
            # 显示
            cv2.imshow('Block A Test - Palm Writing', frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC 或 Q
                print("\n退出程序...")
                break
            elif key == ord('s'):  # S - 保存
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                canvas_path = os.path.join(output_dir, f"trajectory_{timestamp_str}.png")
                cv2.imwrite(canvas_path, canvas)
                print(f"✓ 轨迹已保存: {canvas_path}")
            elif key == ord('c'):  # C - 清空
                canvas = np.ones_like(canvas) * 255
                online_trace_converter = OnlineTraceConverter(smoothing_window=CONFIG['online_trace']['smoothing_window'])
                trajectory.clear()
                print("✓ 轨迹已清空")
    
    except KeyboardInterrupt:
        print("\n\n用户中断...")
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 输出统计信息
        all_traces = online_trace_converter.get_all_traces()
        total_points = sum(len(t) for t in all_traces)
        print("\n" + "=" * 60)
        print("测试完成统计:")
        print(f"  总笔画数: {len(all_traces)}")
        print(f"  总轨迹点数: {total_points}")
        print(f"  输出目录: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
