import cv2
import numpy as np
import sys
import os
import glob
from pathlib import Path

from seg.trajectory_features import TrajectoryFeatures


# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_track.finger_tracking import HandWritingDetector, smart_smooth
from utils.online_trace_converter import OnlineTraceConverter
from projection.projection_visualizer import ProjectionVisualizer
import yaml

# 加载配置文件】
with open("./config.yaml", 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

# 视频源配置
VIDEO_DIR = "data/test_videos"  # 视频目录

# 画布配置
LARGE_CANVAS = tuple(CONFIG['visualization']['canvas_sizes']['large'])

def process_single_video(video_path: str, output_dir: str, projection_visualizer: ProjectionVisualizer) -> bool:
    """
    处理单个视频：获取轨迹 -> 平滑 -> 基于投影密度绘制字符阴影图
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        projection_visualizer: 投影可视化器实例

    Returns:
        是否成功处理
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"正在处理: {video_name}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频源是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频源 {video_path}")
        cap.release()
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['video']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['video']['height'])
    cap.set(cv2.CAP_PROP_FPS, CONFIG['video']['fps'])
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"视频信息: {total_frames} 帧, {fps:.1f} FPS, {duration:.1f} 秒")
    
    # 初始化手势检测器
    detector = HandWritingDetector(gesture_mode=CONFIG['hand_detection']['gesture_mode'])
    
    # 初始化在线轨迹转换器（用于平滑和轨迹管理）
    online_trace_converter = OnlineTraceConverter(smoothing_window=CONFIG['online_trace']['smoothing_window'])
    
    # 轨迹跟踪变量
    trajectory = []
    prev_index_pos = None
    history_points = []
    prev_is_writing = False
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"处理进度: {frame_count}/{total_frames} ({frame_count*100/total_frames:.1f}%)")
        
        # 处理手势检测
        is_writing = detector.process(frame)
        current_index_pos = detector.index_tip_position
        
        # 检测书写状态变化，实现自动分词
        if prev_is_writing and not is_writing:
            # 抬笔，轨迹段结束
            valid_points = [pt for pt in trajectory if pt is not None]
            if len(valid_points) > 5 and not all(pt == valid_points[0] for pt in valid_points):
                # 结束当前轨迹段
                online_trace_converter.end_trace()
            trajectory.clear()
            prev_index_pos = None
            history_points.clear()
        elif not prev_is_writing and is_writing:
            # 落笔，新轨迹段开始
            trajectory.clear()
            prev_index_pos = None
            history_points.clear()
        
        prev_is_writing = is_writing
        
        # 书写状态下记录轨迹点
        if is_writing and current_index_pos != (0, 0):
            if prev_index_pos is not None:
                # 使用smart_smooth进行平滑
                smoothed_pos = smart_smooth(current_index_pos, prev_index_pos, history_points)
                prev_index_pos = smoothed_pos
                trajectory.append(smoothed_pos)
                # 添加到在线轨迹转换器
                online_trace_converter.add_point(smoothed_pos[0], smoothed_pos[1])
            else:
                trajectory.append(None)
                smoothed_pos = current_index_pos
                prev_index_pos = current_index_pos
                trajectory.append(smoothed_pos)
                # 添加到在线轨迹转换器
                online_trace_converter.add_point(smoothed_pos[0], smoothed_pos[1])
            
            # 在视频帧上显示当前书写点
            cv2.circle(frame, smoothed_pos, 12, (0, 0, 255), -1)
        else:
            prev_index_pos = None
            history_points.clear()
        
        # 批量处理模式：不显示窗口，直接处理所有帧
    
    # 处理完成
    cap.release()
    
    print("\n正在生成投影字符阴影图...")
    
    # 获取所有轨迹
    completed_traces = online_trace_converter.get_all_traces()
    current_trace = online_trace_converter.get_current_trace()
    
    if current_trace and len(current_trace) > 5:
        completed_traces.append(current_trace)
    
    if not completed_traces:
        print(f"⚠ {video_name}: 没有检测到轨迹数据")
        return False
    
    print(f"✓ 检测到 {len(completed_traces)} 条轨迹, {sum(len(t) for t in completed_traces)} 个轨迹点")
    
    # 生成输出文件名
    output_path = os.path.join(output_dir, f"{video_name}_projection.png")

    # 使用velocity_visualizer生成投影字符阴影图（使用均匀采样）
    try:
        # 设置采样间隔为10像素，每隔10像素采样一个轨迹点
        sample_interval = 0.0001  # 像素
        projection_img = projection_visualizer.generate_trajectory_projection(
            completed_traces,
            canvas_size=LARGE_CANVAS,
            save_path=output_path,
            sample_interval=sample_interval
        )
        print(f"✓ 投影图已保存: {output_path} (采样间隔={sample_interval}px)")
        return True
    except Exception as e:
        print(f"✗ 生成投影图失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_process_videos():
    """
    批量处理所有视频文件
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 视频目录（相对于脚本目录）
    video_dir = os.path.join(script_dir, VIDEO_DIR)
    
    # 输出目录
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    
    if not video_files:
        print(f"错误：在 {video_dir} 目录下没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 初始化投影可视化器（共享实例）
    projection_visualizer = ProjectionVisualizer(figsize=(15, 10))

    # 统计信息
    success_count = 0
    fail_count = 0
    
    # 批量处理
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] 处理视频...")
        
        try:
            success = process_single_video(video_path, output_dir, projection_visualizer)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            fail_count += 1
    
    # 输出总结
    print("\n" + "=" * 60)
    print("批量处理完成!")
    print(f"成功: {success_count}/{len(video_files)}")
    print(f"失败: {fail_count}/{len(video_files)}")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    batch_process_videos()