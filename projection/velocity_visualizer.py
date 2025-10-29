import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import seaborn as sns
from utils.online_trace_converter import TracePoint


class VelocityVisualizer:
    """轨迹速度可视化器 - 生成轨迹坐标速度变化对应的图像"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化速度可视化器
        
        Args:
            figsize: 图像大小
        """
        self.figsize = figsize
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
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

    def generate_trajectory_velocity_map(self, traces: List[List[TracePoint]], 
                                       canvas_size: Tuple[int, int] = (1280, 720),
                                       save_path: Optional[str] = None) -> np.ndarray:
        """
        生成轨迹上速度颜色映射图
        """
        width, height = canvas_size
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 收集所有坐标和速度
        all_x_coords = []
        all_y_coords = []
        all_velocities = []
        for trace in traces:
            for point in trace:
                all_x_coords.append(point.x)
                all_y_coords.append(point.y)
                all_velocities.append(point.velocity)
        
        if not all_velocities:
            ax.text(0.5, 0.5, '没有轨迹数据', ha='center', va='center', 
                   fontsize=20, transform=ax.transAxes)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            plt.close(fig)
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        max_velocity = max(all_velocities) if all_velocities else 1
        min_velocity = min(all_velocities) if all_velocities else 0
        
        # 为每条轨迹绘制带速度颜色的线段
        for trace_idx, trace in enumerate(traces):
            if len(trace) < 2:
                continue
            
            for i in range(len(trace) - 1):
                point1 = trace[i]
                point2 = trace[i + 1]
                
                # 归一化速度到0-1范围
                normalized_velocity = (point2.velocity - min_velocity) / (max_velocity - min_velocity) if max_velocity > min_velocity else 0
                
                # 使用热力图颜色映射
                color = plt.cm.hot(normalized_velocity)
                
                # 绘制线段，线条粗细根据速度变化
                line_width = 2 + normalized_velocity * 4  # 2-6像素
                ax.plot([point1.x, point2.x], [point1.y, point2.y], 
                       color=color, linewidth=line_width, alpha=0.8)
            
            # 标记起点和终点
            start_point = trace[0]
            end_point = trace[-1]
            ax.plot(start_point.x, start_point.y, 'go', markersize=10)
            ax.plot(end_point.x, end_point.y, 'ro', markersize=10)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, 
                                  norm=plt.Normalize(vmin=min_velocity, vmax=max_velocity))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('速度 (像素/秒)', rotation=270, labelpad=20)
        
        # 设置自适应坐标轴范围
        if all_x_coords and all_y_coords:
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            
            # 添加10%边距
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_max + y_margin, y_min - y_margin)  # Y轴反转
        else:
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
        
        ax.set_xlabel('X 坐标 (像素)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y 坐标 (像素)', fontsize=12, fontweight='bold')
        ax.set_title('轨迹速度映射图', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹速度映射图已保存到: {save_path}")
        
        # 转换为numpy数组
        img_array = self._fig_to_array(fig)
        
        plt.close(fig)
        return img_array

    # small utility to display with OpenCV
    @staticmethod
    def display_image(image: np.ndarray, window_name: str = "Velocity Visualization"):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
