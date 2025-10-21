from attr import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import seaborn as sns
from utils.online_trace_converter import TracePoint
from seg.type import PcaPack



class ProjectionVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化速度可视化器
        
        Args:
            figsize: 图像大小
        """
        self.figsize = figsize
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self._last_pcapack = None

    def _fig_to_array(self, fig) -> np.ndarray:
        """
        将matplotlib figure转换为numpy数组，避免reshape错误
        
        Args:
            fig: matplotlib figure对象
            
        Returns:
            图像数组
        """
        try:
            fig.canvas.draw()
            from PIL import Image
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            pil_image = Image.open(buf)
            img_array = np.array(pil_image)

            # Ensure RGB
            if img_array.ndim == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            elif img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)

            buf.close()
            return img_array
        except Exception as e:
            print(f"Figure to array conversion error: {e}")
            return np.zeros((400, 600, 3), dtype=np.uint8)

    def _uniform_sample_trajectory(self, trace: List[TracePoint], sample_interval: float = 10.0) -> List[TracePoint]:
        """
        对轨迹进行均匀采样，每隔一定长度采样一个点
        
        Args:
            trace: 原始轨迹点列表
            sample_interval: 采样间隔（像素），默认10像素
            
        Returns:
            均匀采样后的轨迹点列表
        """
        if len(trace) < 2:
            return trace
        
        sampled_points = [trace[0]]  # 总是包含起始点
        accumulated_length = 0.0
        
        for i in range(1, len(trace)):
            # 计算当前点与上一个点的距离
            dx = trace[i].x - trace[i-1].x
            dy = trace[i].y - trace[i-1].y
            segment_length = np.sqrt(dx**2 + dy**2)
            
            accumulated_length += segment_length
            
            # 如果累计长度超过采样间隔，采样当前点
            # 修复：保留超出部分的长度，而不是重置为0
            if accumulated_length >= sample_interval:
                sampled_points.append(trace[i])
                # 保留超出采样间隔的部分，用于下一次判断
                accumulated_length -= sample_interval
        
        # 总是包含终点（如果它还没被采样）
        if len(sampled_points) == 0 or sampled_points[-1] != trace[-1]:
            sampled_points.append(trace[-1])
        
        return sampled_points

    def generate_trajectory_projection(self, traces: List[List[TracePoint]], 
                                     canvas_size: Tuple[int, int] = (1280, 720),
                                     save_path: Optional[str] = None,
                                     sample_interval: float = 1) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate trajectory projection analysis with uniform sampling
        
        Args:
            traces: List of trajectory points
            canvas_size: Canvas size (width, height)
            save_path: Save path, None if not saving
            sample_interval: Sampling interval in pixels (default 10.0)
            
        Returns:
            Tuple containing:
            - Projection analysis image as numpy array
            - List of boundary line information dictionaries, each containing:
                - index: Boundary index
                - boundary_idx: Bin index of the boundary
                - start: Start point coordinates (x, y)
                - end: End point coordinates (x, y)
                - center: Center point coordinates (x, y)
        """
        try:
            from sklearn.decomposition import PCA
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            print("Warning: scikit-learn and scipy required for PCA")
            print("Run: pip install scikit-learn scipy")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'scikit-learn & scipy required\npip install scikit-learn scipy', 
                   ha='center', va='center', fontsize=16)
            img_array = self._fig_to_array(fig)
            plt.close(fig)
            return img_array, []
        
        # Create single professional figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # *** 步骤1：对每条轨迹进行均匀采样 ***
        sampled_traces = []
        for trace in traces:
            sampled_trace = self._uniform_sample_trajectory(trace, sample_interval)
            sampled_traces.append(sampled_trace)
            print(f"轨迹采样: 原始点数={len(trace)}, 采样后点数={len(sampled_trace)}, 采样间隔={sample_interval}px")
        
        # Collect all sampled trajectory points for PCA
        all_points = []
        trace_colors = plt.cm.tab10(np.linspace(0, 1, max(len(traces), 1)))
        
        for sampled_trace in sampled_traces:
            for point in sampled_trace:
                all_points.append([point.x, point.y])
        
        if len(all_points) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for PCA', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            img_array = self._fig_to_array(fig)
            plt.close(fig)
            return img_array, []
        
        all_points = np.array(all_points)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(all_points)
        
        # Get principal components
        principal_direction = pca.components_[0]
        center_point = np.mean(all_points, axis=0)
        
        # Calculate projection line endpoints
        line_length = max(canvas_size) * 1.2
        line_start = center_point - principal_direction * line_length / 2
        line_end = center_point + principal_direction * line_length / 2
        
        # 1. Draw original trajectories with professional styling
        for trace_idx, trace in enumerate(traces):
            if len(trace) < 2:
                continue
            x_coords = [point.x for point in trace]
            y_coords = [point.y for point in trace]
            
            # Draw trajectory line
            ax.plot(x_coords, y_coords, color=trace_colors[trace_idx], 
                   linewidth=3, alpha=0.85, label=f'Trajectory {trace_idx + 1}',
                   zorder=3)
            
            # Mark start and end points
            ax.scatter(x_coords[0], y_coords[0], s=150, c=trace_colors[trace_idx],
                      marker='o', edgecolors='white', linewidths=2.5, 
                      zorder=5, label='Start' if trace_idx == 0 else '')
            ax.scatter(x_coords[-1], y_coords[-1], s=150, c=trace_colors[trace_idx],
                      marker='s', edgecolors='white', linewidths=2.5,
                      zorder=5, label='End' if trace_idx == 0 else '')
        
        # 2. Draw PCA principal axis
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 
               'k--', linewidth=2.5, alpha=0.6, label='Principal Axis (PC1)',
               zorder=2)
        
        # 3. Calculate projections using sampled points
        projection_positions = []
        sampled_point_coords = []  # 记录采样点坐标，用于后续区域检测
        
        for sampled_trace in sampled_traces:
            for point in sampled_trace:
                point_vec = np.array([point.x, point.y]) - center_point
                projection_length = np.dot(point_vec, principal_direction)
                projection_point = center_point + projection_length * principal_direction
                projection_positions.append(projection_length)
                sampled_point_coords.append([point.x, point.y])
                
                # Optionally draw some projection lines (sampled)
                if np.random.random() < 0.15:  # Show 15% of projection lines
                    ax.plot([point.x, projection_point[0]], 
                           [point.y, projection_point[1]], 
                           color='gray', alpha=0.15, linewidth=0.8, 
                           zorder=1, linestyle=':')
        
        # 初始化分界线列表
        boundary_lines = []
        
        # 4. Calculate density distribution (based on sampled points)
        if projection_positions:
            min_pos = min(projection_positions)
            max_pos = max(projection_positions)
            
            # Create fine-grained bins
            n_bins = max(30, int((max_pos - min_pos) / 15))
            bin_edges = np.linspace(min_pos, max_pos, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # *** 使用均匀采样后的点进行直方图统计 ***
            # 每个bin中的点数代表该区域的采样点数量
            bin_counts, _ = np.histogram(projection_positions, bins=bin_edges)
            
            # 直接显示采样点数
            display_counts = bin_counts.astype(float)
            max_count = max(display_counts) if len(display_counts) > 0 else 1
            
            print(f"\n投影统计: 总采样点数={len(projection_positions)}, bins数={n_bins}")
            print(f"采样点分布: 最大bin={int(max_count)}个点, 平均bin={np.mean(display_counts):.1f}个点")
            
            # 5. Draw density curve BELOW the principal axis (pointing UPWARD)
            # Calculate perpendicular direction
            perpendicular = np.array([-principal_direction[1], principal_direction[0]])
            
            # Determine offset distance (move the baseline further down)
            offset_distance = 200  # pixels below the principal axis (increased from 120)
            
            # Create density curve points
            density_curve_x = []
            density_curve_y = []
            baseline_x = []
            baseline_y = []
            
            for pos, point_count in zip(bin_centers, display_counts):
                # Point on principal axis
                axis_point = center_point + pos * principal_direction
                
                # Baseline point (offset from axis, moved down)
                baseline_point = axis_point + perpendicular * offset_distance
                baseline_x.append(baseline_point[0])
                baseline_y.append(baseline_point[1])
                
                # Density curve point (UPWARD from baseline toward the axis)
                # 高度与点数成正比，最大高度为 120 像素
                density_height = (point_count / max_count) * 120 if max_count > 0 else 0
                curve_point = baseline_point - perpendicular * density_height  # Negative to point upward
                density_curve_x.append(curve_point[0])
                density_curve_y.append(curve_point[1])
            
            # Draw baseline (parallel to principal axis)
            ax.plot(baseline_x, baseline_y, 'k-', linewidth=2, alpha=0.5, 
                   linestyle='-.', label='Density Baseline', zorder=2)
            
            # Draw density curve - 基于均匀采样点的投影密度
            ax.plot(density_curve_x, density_curve_y, 'b-', linewidth=3, 
                   alpha=0.8, label=f'Sampled Point Count (Δ={sample_interval}px)', zorder=4)
            
            # Fill area between baseline and density curve
            ax.fill_between(baseline_x, baseline_y, density_curve_y,
                           color='lightblue', alpha=0.4, zorder=2,
                           label='Sampled Point Distribution')
            
            # 在密度曲线上标注采样点数
            label_interval = max(1, len(bin_counts) // 10)  # 最多标注10个位置
            for i in range(0, len(bin_counts), label_interval):
                if bin_counts[i] > 0:  # 只标注有点的位置
                    ax.text(density_curve_x[i], density_curve_y[i] - 15, 
                           f'{int(bin_counts[i])}',
                           fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', 
                                   alpha=0.7,
                                   edgecolor='orange'),
                           zorder=7)
            
            # === 新算法：基于二维投影的字符区域检测 ===
            # 步骤1：在主方向上找到投影点数≥1的连续区域（字符宽度）
            # 步骤2：对每个区域内的点在法线方向投影，确定字符高度
            
            print("\n=== 字符区域检测（二维投影法）===")
            
            # 1. 找到主方向上投影点数≥1的连续区域
            character_width_regions = []  # 存储字符的宽度区域 [(start_idx, end_idx, point_indices), ...]
            in_region = False
            region_start = 0
            region_point_indices = []  # 记录属于当前区域的所有点索引
            
            # 为每个bin记录属于它的点的索引
            bin_to_points = [[] for _ in range(len(bin_centers))]
            for point_idx, proj_pos in enumerate(projection_positions):
                # 找到这个点属于哪个bin
                bin_idx = np.searchsorted(bin_edges, proj_pos) - 1
                if 0 <= bin_idx < len(bin_centers):
                    bin_to_points[bin_idx].append(point_idx)
            
            # 遍历所有bin，找连续的有点的区域
            for i, count in enumerate(bin_counts):
                if count >= 1:  # 有投影点
                    if not in_region:
                        # 开始新区域
                        in_region = True
                        region_start = i
                        region_point_indices = bin_to_points[i].copy()
                    else:
                        # 继续当前区域
                        region_point_indices.extend(bin_to_points[i])
                else:
                    if in_region:
                        # 结束当前区域
                        character_width_regions.append((region_start, i - 1, region_point_indices))
                        in_region = False
                        region_point_indices = []
            
            # 处理最后一个区域
            if in_region:
                character_width_regions.append((region_start, len(bin_counts) - 1, region_point_indices))
            
            print(f"检测到 {len(character_width_regions)} 个连续投影区域")
            
            # 2. 合并过小的区域（可能是噪声）并限制为3-4个字符
            min_region_width = 2  # 最小区域宽度（bin数）
            filtered_regions = []
            for start, end, point_indices in character_width_regions:
                if end - start + 1 >= min_region_width and len(point_indices) >= 3:
                    filtered_regions.append((start, end, point_indices))
            
            print(f"过滤后剩余 {len(filtered_regions)} 个有效区域")
            
            # 3. 如果区域太多，合并相邻的小区域
            if len(filtered_regions) > 4:
                # 计算每个区域的"强度"（点数）
                region_strengths = [(end - start + 1) * len(point_indices) 
                                   for start, end, point_indices in filtered_regions]
                
                # 保留最强的4个区域，合并其他
                strong_indices = np.argsort(region_strengths)[-4:]
                strong_indices = sorted(strong_indices)
                filtered_regions = [filtered_regions[i] for i in strong_indices]
                print(f"保留最强的4个区域")
            
            # 4. 不再强制分割区域到特定数量，保持检测到的区域数量
            # 只有当单个区域特别大时（可能包含多个字符）才考虑分割
            if len(filtered_regions) == 1:
                start, end, point_indices = filtered_regions[0]
                region_width = end - start + 1
                # 如果唯一的区域特别宽（超过30个bin），可能包含多个字符，尝试分割
                if region_width > 30:
                    print(f"检测到单一大区域(宽度={region_width}个bin)，尝试分割...")
                    # 分割成2-3个子区域
                    num_splits = min(3, region_width // 15)  # 根据宽度自动决定分割数
                    new_regions = []
                    split_size = region_width // num_splits
                    
                    for i in range(num_splits):
                        split_start = start + i * split_size
                        split_end = start + (i + 1) * split_size - 1 if i < num_splits - 1 else end
                        split_points = [p for p in point_indices 
                                      if split_start <= np.searchsorted(bin_edges, projection_positions[p]) - 1 <= split_end]
                        if len(split_points) > 0:
                            new_regions.append((split_start, split_end, split_points))
                    
                    if len(new_regions) > 1:
                        filtered_regions = new_regions
                        print(f"分割后得到 {len(filtered_regions)} 个区域")
                    else:
                        print(f"分割失败，保持原区域")
                else:
                    print(f"单一区域宽度适中，不进行分割")
            else:
                print(f"保持 {len(filtered_regions)} 个检测到的区域")
            
            # 5. 对每个字符宽度区域，在法线方向投影以确定字符高度
            character_regions = []  # 最终的字符区域 [(start_idx, end_idx, bbox), ...]
            perpendicular = np.array([-principal_direction[1], principal_direction[0]])  # 法线方向
            
            for region_idx, (start_idx, end_idx, point_indices) in enumerate(filtered_regions):
                if len(point_indices) == 0:
                    continue
                
                # 使用采样后的点坐标（而不是原始轨迹点）
                region_points = [sampled_point_coords[idx] for idx in point_indices if idx < len(sampled_point_coords)]
                
                if len(region_points) == 0:
                    continue
                
                region_points = np.array(region_points)
                
                # 在法线方向投影
                perpendicular_projections = []
                for point in region_points:
                    point_vec = point - center_point
                    perp_proj = np.dot(point_vec, perpendicular)
                    perpendicular_projections.append(perp_proj)
                
                # 计算字符高度范围
                perp_min = np.min(perpendicular_projections)
                perp_max = np.max(perpendicular_projections)
                
                # 主方向范围
                main_min = bin_centers[start_idx]
                main_max = bin_centers[end_idx]
                
                # 构建2D边界框（在PCA坐标系中）
                bbox = {
                    'main_min': main_min,
                    'main_max': main_max,
                    'perp_min': perp_min,
                    'perp_max': perp_max,
                    'width': main_max - main_min,
                    'height': perp_max - perp_min,
                    'center_main': (main_min + main_max) / 2,
                    'center_perp': (perp_min + perp_max) / 2
                }
                
                character_regions.append((start_idx, end_idx, bbox))
                
                print(f"字符区域 {region_idx + 1}: "
                      f"宽度={bbox['width']:.1f}px, 高度={bbox['height']:.1f}px, "
                      f"采样点数={len(point_indices)}")
            
            # 6. 调整字符区域数量：如果超过4个，只保留最强的4个；否则保持原样
            if len(character_regions) > 4:
                # 超过4个，只保留最宽的4个（按宽度排序）
                print(f"⚠ 字符区域超过4个({len(character_regions)}个)，保留最宽的4个")
                widths = [bbox['width'] for _, _, bbox in character_regions]
                keep_indices = sorted(np.argsort(widths)[-4:])
                character_regions = [character_regions[i] for i in keep_indices]
            else:
                # 不足4个或恰好4个，保持原样，不使用等分法
                print(f"✓ 检测到 {len(character_regions)} 个字符区域（保持原样，不强制等分）")
            
            print(f"最终确定 {len(character_regions)} 个字符区域\n")
            
            # 绘制字符阴影区域（使用2D bbox信息）
            shadow_colors = plt.cm.Set3(np.linspace(0, 1, max(len(character_regions), 1)))
            
            for region_idx, (start_idx, end_idx, bbox) in enumerate(character_regions):
                # 获取该区域的baseline点和密度曲线点
                region_baseline_x = baseline_x[start_idx:end_idx+2]  # +2 to include boundary
                region_baseline_y = baseline_y[start_idx:end_idx+2]
                
                # 计算区域的垂直范围（从baseline向上延伸，覆盖整个轨迹）
                # 使用perpendicular方向，向上延伸足够的距离
                shadow_height = 400  # 阴影高度（覆盖轨迹）
                
                # 构建阴影多边形的顶点
                shadow_polygon_x = []
                shadow_polygon_y = []
                
                # 下边界（baseline下方延伸一点）
                for i in range(len(region_baseline_x)):
                    baseline_point = np.array([region_baseline_x[i], region_baseline_y[i]])
                    lower_point = baseline_point + perpendicular * 30  # 向下延伸30px
                    shadow_polygon_x.append(lower_point[0])
                    shadow_polygon_y.append(lower_point[1])
                
                # 上边界（向上延伸覆盖轨迹，逆序）
                for i in range(len(region_baseline_x) - 1, -1, -1):
                    baseline_point = np.array([region_baseline_x[i], region_baseline_y[i]])
                    upper_point = baseline_point - perpendicular * shadow_height  # 向上延伸
                    shadow_polygon_x.append(upper_point[0])
                    shadow_polygon_y.append(upper_point[1])
                
                # 绘制阴影多边形
                ax.fill(shadow_polygon_x, shadow_polygon_y,
                       color=shadow_colors[region_idx],
                       alpha=0.15,
                       zorder=1,
                       edgecolor=shadow_colors[region_idx],
                       linewidth=2,
                       linestyle='--')
                
                # 在阴影区域中心添加字符标签
                center_idx = (start_idx + end_idx) // 2
                center_x = baseline_x[center_idx]
                center_y = baseline_y[center_idx]
                label_point = np.array([center_x, center_y]) - perpendicular * (shadow_height / 2)
                
                
                
                # # 绘制字符的2D边界框（在主方向和法线方向）
                # if bbox['height'] > 0:  # 只有当法线投影有效时才绘制
                #     # 计算边界框的四个角点
                #     corners_pca = [
                #         (bbox['main_min'], bbox['perp_min']),
                #         (bbox['main_max'], bbox['perp_min']),
                #         (bbox['main_max'], bbox['perp_max']),
                #         (bbox['main_min'], bbox['perp_max']),
                #         (bbox['main_min'], bbox['perp_min'])  # 闭合
                #     ]
                    
                #     # 转换回原始坐标系
                #     bbox_x = []
                #     bbox_y = []
                #     for main_proj, perp_proj in corners_pca:
                #         point_in_original = center_point + main_proj * principal_direction + perp_proj * perpendicular
                #         bbox_x.append(point_in_original[0])
                #         bbox_y.append(point_in_original[1])
                    
                #     # 绘制边界框
                #     ax.plot(bbox_x, bbox_y, 
                #            color=shadow_colors[region_idx],
                #            linewidth=3,
                #            linestyle='-',
                #            alpha=0.8,
                #            zorder=7,
                #            label=f'Char {region_idx + 1} BBox' if region_idx == 0 else '')
            
            # 标记字符边界（区域之间的分割线）——使用区域中点绘制分割线
            for i in range(len(character_regions) - 1):
                _, end_idx1, bbox1 = character_regions[i]
                start_idx2, _, bbox2 = character_regions[i + 1]

                # 在两个区域的中间位置绘制分割线
                boundary_idx = (end_idx1 + start_idx2) // 2
                if boundary_idx < len(baseline_x):
                    boundary_x = baseline_x[boundary_idx]
                    boundary_y = baseline_y[boundary_idx]

                    # 绘制垂直于主轴的分割线
                    line_length = 450
                    line_start = np.array([boundary_x, boundary_y]) + perpendicular * 30
                    line_end = np.array([boundary_x, boundary_y]) - perpendicular * line_length

                    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]],
                           'r--', linewidth=2.5, alpha=0.7,
                           zorder=6,
                           label='Character Boundary' if i == 0 else '')
                    
                    # 保存分界线信息
                    boundary_lines.append({
                        'index': i,
                        'boundary_idx': boundary_idx,
                        'start': (line_start[0], line_start[1]),
                        'end': (line_end[0], line_end[1]),
                        'center': (boundary_x, boundary_y)
                    })
        
        # 6. Add statistics annotation
        if projection_positions:
            explained_var = pca.explained_variance_ratio_
            pc1_angle = np.degrees(np.arctan2(principal_direction[1], 
                                             principal_direction[0]))
            
           
        
        # Configure plot with adaptive axis limits
        # Collect all x and y coordinates for determining proper bounds
        all_x_coords = []
        all_y_coords = []
        
        # Add trajectory points
        for trace in traces:
            all_x_coords.extend([point.x for point in trace])
            all_y_coords.extend([point.y for point in trace])
        
        # Add density curve points if they exist
        if projection_positions and len(density_curve_x) > 0 and len(baseline_x) > 0:
            all_x_coords.extend(density_curve_x)
            all_x_coords.extend(baseline_x)
            all_y_coords.extend(density_curve_y)
            all_y_coords.extend(baseline_y)
        
        # Calculate adaptive bounds with margin
        if all_x_coords and all_y_coords:
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            
            # Add 10% margin on each side
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_max + y_margin, y_min - y_margin)  # Inverted for screen coordinates
        else:
            # Fallback to canvas size
            ax.set_xlim(0, canvas_size[0])
            ax.set_ylim(canvas_size[1], 0)
        
        ax.set_xlabel('X Coordinate (pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Trajectory PCA with Projection Density & Character Region Segmentation',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                 edgecolor='black', fancybox=True)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Trajectory projection analysis saved: {save_path}")
        
        img_array = self._fig_to_array(fig)
        plt.close(fig)

        self._last_pcapack = PcaPack(
            mean=center_point,
            pc1_axis=principal_direction,
            pc1_normal=perpendicular,
            density_bins=bin_centers if projection_positions else np.array([]),
            density_profile=display_counts if projection_positions else np.array([]),
            red_lines=np.array([ ( (line['start'][0] - center_point[0]) * principal_direction[0] +
                                     (line['start'][1] - center_point[1]) * principal_direction[1] )
                                  for line in boundary_lines ]) if projection_positions else np.array([])
        )
        
        # 返回图像数组和分界线信息
        return img_array, boundary_lines if projection_positions else (img_array, [])
    
    def get_last_pcapack(self) -> Optional[PcaPack]:
        """
        获取上一次计算的PCA数据包
        
        Returns:
            上一次的PcaPack对象，若无则为None
        """
        return self._last_pcapack

    def simple_projection_with_conn_segs(self,
                                         traces: List[List[TracePoint]],
                                         boundary_lines: List[dict],
                                         conn_segs_by_trace: dict = None,
                                         canvas_size: Tuple[int, int] = (1280, 720),
                                         save_path: Optional[str] = None) -> np.ndarray:
        """
        A lightweight projection visualization: draw trajectories and red boundary lines.
        Optionally overlay detected connected segments (dashed) provided per-trace.

        Args:
            traces: list of trajectories (each a list of TracePoint or Nx2 numpy array)
            boundary_lines: list of dicts as produced by `generate_trajectory_projection`
            conn_segs_by_trace: mapping trace_idx -> list of segments, where each segment
                has attributes `i`, `j` and optionally `score`/`geom`.
            canvas_size: output canvas size
            save_path: optional path to save the image

        Returns:
            rgb image as numpy array
        """
        fig, ax = plt.subplots(figsize=(canvas_size[0] / 100, canvas_size[1] / 100))

        # draw traces
        for t_idx, trace in enumerate(traces):
            if isinstance(trace, np.ndarray):
                pts = trace
            else:
                # TracePoint objects have .x and .y
                pts = np.array([[p.x, p.y] for p in trace]) if len(trace) > 0 else np.zeros((0, 2))
            if pts.shape[0] == 0:
                continue
            ax.plot(pts[:, 0], pts[:, 1], color='black', linewidth=2, zorder=3)

            # overlay conn segments for this trace
            if conn_segs_by_trace and t_idx in conn_segs_by_trace:
                for seg in conn_segs_by_trace[t_idx]:
                    i, j = int(seg.i), int(seg.j)
                    if i < 0 or j >= pts.shape[0] or i >= j:
                        continue
                    ax.plot(pts[i:j+1, 0], pts[i:j+1, 1], linestyle='--', color='lime', linewidth=2.5, zorder=5)

        # draw boundary red lines
        for bl in boundary_lines:
            try:
                s = bl['start']; e = bl['end']
                ax.plot([s[0], e[0]], [s[1], e[1]], 'r--', linewidth=2.2, zorder=6)
            except Exception:
                continue

        # Configure adaptive axis limits so the saved image matches screen coordinates
        all_x = []
        all_y = []
        for trace in traces:
            if isinstance(trace, np.ndarray):
                pts = trace
            else:
                pts = np.array([[p.x, p.y] for p in trace]) if len(trace) > 0 else np.zeros((0, 2))
            if pts.shape[0] > 0:
                all_x.extend(pts[:, 0].tolist())
                all_y.extend(pts[:, 1].tolist())
        # include boundary line endpoints
        for bl in boundary_lines:
            try:
                s = bl['start']; e = bl['end']
                all_x.extend([s[0], e[0]]); all_y.extend([s[1], e[1]])
            except Exception:
                pass

        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 50
            y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 50
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            # Invert Y so image matches screen coordinates (y increases downward)
            ax.set_ylim(y_max + y_margin, y_min - y_margin)

        ax.set_axis_off()
        plt.tight_layout(pad=0)
        img = self._fig_to_array(fig)
        plt.close(fig)

        if save_path:
            try:
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Failed to save simple projection: {e}")

        return img
