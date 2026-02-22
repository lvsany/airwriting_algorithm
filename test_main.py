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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_track.dual_hand_detector import DualHandDetector
from hand_track.finger_tracking import HandWritingDetector, smart_smooth
from utils.online_trace_converter import OnlineTraceConverter

with open("./config.yaml", 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

# ─── 配色方案 ───
C = {
    'bg':         (20, 20, 25),
    'panel':      (30, 30, 38),
    'panel_edge': (55, 55, 68),
    'text':       (210, 210, 215),
    'text_dim':   (120, 120, 130),
    'accent':     (230, 165, 60),     # 暖橙 accent
    'accent2':    (90, 200, 250),     # 亮蓝 accent
    'palm':       (80, 220, 200),     # 青色手掌
    'palm_fill':  (60, 160, 150),
    'writing':    (80, 230, 120),     # 书写绿
    'idle':       (90, 90, 100),
    'hover':      (200, 180, 80),
    'touch':      (80, 180, 240),
    'lifted':     (180, 100, 220),
    'warn':       (60, 60, 240),      # 红色警告 (BGR)
    'axis_x':     (80, 80, 255),
    'axis_y':     (80, 220, 80),
    'axis_z':     (255, 160, 80),
    'traj':       (45, 45, 50),       # 轨迹线颜色
    'traj_dot':   (230, 165, 60),     # 轨迹当前点
    'canvas_bg':  (245, 245, 248),
    'canvas_grid':(225, 225, 230),
    'canvas_line':(40, 40, 50),
    'canvas_border': (180, 180, 190),
}

STATE_STYLE = {
    'idle':     {'color': C['idle'],    'label': 'IDLE'},
    'hovering': {'color': C['hover'],   'label': 'HOVER'},
    'touching': {'color': C['touch'],   'label': 'TOUCH'},
    'writing':  {'color': C['writing'], 'label': 'WRITE'},
    'lifted':   {'color': C['lifted'],  'label': 'LIFT'},
}


def rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """绘制圆角矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    # 四个角圆弧 + 填充
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1+r, y1), (x2-r, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1+r), (x2, y2-r), 255, -1)
    cv2.circle(mask, (x1+r, y1+r), r, 255, -1)
    cv2.circle(mask, (x2-r, y1+r), r, 255, -1)
    cv2.circle(mask, (x1+r, y2-r), r, 255, -1)
    cv2.circle(mask, (x2-r, y2-r), r, 255, -1)
    if thickness == -1:
        img[mask > 0] = color
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness, cv2.LINE_AA)


def draw_pill(img, center, w, h, color, alpha=1.0):
    """绘制药丸形状标签底色"""
    x, y = center
    r = h // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x - w//2 + r, y - h//2), (x + w//2 - r, y + h//2), color, -1)
    cv2.circle(overlay, (x - w//2 + r, y), r, color, -1)
    cv2.circle(overlay, (x + w//2 - r, y), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img, text, pos, scale=0.45, color=None, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """统一文字绘制"""
    if color is None:
        color = C['text']
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_hbar(img, x, y, w, h, ratio, fg_color, bg_color=(50, 50, 58)):
    """绘制水平进度条"""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = max(1, int(w * np.clip(ratio, 0, 1)))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), fg_color, -1)


# ─── 手掌平面可视化 ───
def draw_palm_plane_visualization(frame, detector):
    """绘制手掌平面多边形和局部坐标轴"""
    if not hasattr(detector, 'palm_tracker'):
        return
    palm_system = detector.palm_tracker.get_current_system()
    if palm_system is None or detector.palm_lm is None:
        return

    h, w = frame.shape[:2]
    lm = detector.palm_lm.landmark

    def p2d(p3):
        return (int(p3[0] * w), int(p3[1] * h))

    # 手掌凸包（全21关键点取ConvexHull）
    all_pts = np.array([p2d([lm[i].x, lm[i].y, lm[i].z]) for i in range(21)])
    hull = cv2.convexHull(all_pts)

    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, hull, C['palm_fill'])
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [hull], True, C['palm'], 1, cv2.LINE_AA)

    # 标注各指根
    for idx in [0, 5, 9, 13, 17]:
        pt = p2d([lm[idx].x, lm[idx].y, lm[idx].z])
        cv2.circle(frame, pt, 3, C['palm'], -1, cv2.LINE_AA)

    # 坐标轴
    origin = palm_system.origin
    o2d = p2d(origin)
    axis_len = 0.08

    for axis, color, label in [
        (palm_system.u_axis, C['axis_x'], 'u'),
        (palm_system.v_axis, C['axis_y'], 'v'),
        (palm_system.w_axis, C['axis_z'], 'n'),
    ]:
        end = origin + axis * axis_len
        e2d = p2d(end)
        cv2.arrowedLine(frame, o2d, e2d, color, 2, cv2.LINE_AA, tipLength=0.22)
        put_text(frame, label, (e2d[0] + 5, e2d[1] - 5), 0.38, color, 1)

    cv2.circle(frame, o2d, 4, C['accent'], -1, cv2.LINE_AA)


# ─── HUD 信息面板 ───
def draw_debug_info(frame, detector, is_writing, fps, palm_mode=True):
    h, w = frame.shape[:2]

    # 左侧面板
    pw = 260
    panel = frame.copy()
    rounded_rect(panel, (8, 8), (8 + pw, 250), C['panel'], 10)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    rounded_rect(frame, (8, 8), (8 + pw, 250), C['panel_edge'], 10, 1)

    x0 = 22
    y = 32

    # 标题
    put_text(frame, "PalmWrite", (x0, y), 0.55, C['accent'], 1)
    put_text(frame, "BLOCK A", (x0 + 135, y), 0.35, C['text_dim'], 1)
    y += 12
    cv2.line(frame, (x0, y), (x0 + pw - 28, y), C['panel_edge'], 1, cv2.LINE_AA)
    y += 20

    # FPS
    fps_color = C['writing'] if fps >= 25 else C['hover'] if fps >= 15 else C['warn']
    put_text(frame, f"FPS", (x0, y), 0.38, C['text_dim'])
    put_text(frame, f"{fps:.0f}", (x0 + 50, y), 0.42, fps_color, 1)

    if palm_mode:
        # 状态 pill
        state = detector.contact_sm.state.value
        ss = STATE_STYLE.get(state, STATE_STYLE['idle'])
        pill_x = x0 + 150
        draw_pill(frame, (pill_x + 40, y - 5), 80, 20, ss['color'], 0.7)
        put_text(frame, ss['label'], (pill_x + 12, y), 0.38, (255, 255, 255), 1)

        y += 28

        # 手部角色
        lr = detector.left_role.value if detector.left_role else "–"
        rr = detector.right_role.value if detector.right_role else "–"
        put_text(frame, "L hand", (x0, y), 0.35, C['text_dim'])
        put_text(frame, lr.upper(), (x0 + 65, y), 0.35, C['accent2'] if lr == 'palm' else C['accent'])
        put_text(frame, "R hand", (x0 + 130, y), 0.35, C['text_dim'])
        put_text(frame, rr.upper(), (x0 + 195, y), 0.35, C['accent2'] if rr == 'palm' else C['accent'])
        y += 26

        # 距离
        thr = detector.contact_sm.writing_threshold
        put_text(frame, "Distance", (x0, y), 0.35, C['text_dim'])
        if detector.dist_palm is not None:
            d = detector.dist_palm
            max_d = 120.0
            ratio = d / max_d
            bar_color = C['writing'] if d <= thr else C['touch'] if d < thr * 2 else C['hover'] if d < 100 else C['idle']
            draw_hbar(frame, x0 + 80, y - 10, 140, 12, ratio, bar_color)
            put_text(frame, f"{d:.0f}mm", (x0 + 82, y), 0.32, (255, 255, 255), 1)
        else:
            put_text(frame, "OUT", (x0 + 80, y), 0.35, C['warn'])
        y += 26

        # 自适应阈值
        sm = detector.contact_sm
        buf_size = len(sm.adaptive_buffer)
        is_adaptive = sm.adaptive_enabled and buf_size >= sm.adaptive_min_samples
        put_text(frame, "Threshold", (x0, y), 0.35, C['text_dim'])
        thr_color = C['writing'] if is_adaptive else C['warn']
        put_text(frame, f"{thr:.1f}mm", (x0 + 80, y), 0.38, thr_color, 1)
        mode_label = "ADAPTIVE" if is_adaptive else f"WARMUP {buf_size}/{sm.adaptive_min_samples}"
        put_text(frame, mode_label, (x0 + 130, y), 0.28, thr_color, 1)
        y += 26

        # 坐标
        put_text(frame, "Palm XY", (x0, y), 0.35, C['text_dim'])
        if detector.write_pos_palm:
            px, py = detector.write_pos_palm
            put_text(frame, f"({px:.3f}, {py:.3f})", (x0 + 80, y), 0.35, C['text'])
        else:
            put_text(frame, "–", (x0 + 80, y), 0.35, C['text_dim'])
        y += 26

        # 平面信息
        psys = detector.palm_tracker.get_current_system()
        if psys:
            put_text(frame, "Inliers", (x0, y), 0.35, C['text_dim'])
            put_text(frame, str(psys.n_inliers), (x0 + 80, y), 0.35, C['text'])
            put_text(frame, "Plane", (x0 + 130, y), 0.35, C['text_dim'])
            put_text(frame, "OK", (x0 + 180, y), 0.35, C['writing'])
        else:
            put_text(frame, "Plane", (x0, y), 0.35, C['text_dim'])
            put_text(frame, "NO FIT", (x0 + 80, y), 0.35, C['warn'])

    else:
        y += 28
        status = "WRITING" if is_writing else "IDLE"
        sc = C['writing'] if is_writing else C['idle']
        draw_pill(frame, (x0 + 80, y - 5), 90, 20, sc, 0.7)
        put_text(frame, status, (x0 + 48, y), 0.38, (255, 255, 255), 1)

    # 底部操作提示
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), C['panel'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), C['panel_edge'], 1)
    hints = "ESC  Exit     S  Save     C  Clear"
    put_text(frame, hints, (w // 2 - 150, h - 9), 0.37, C['text_dim'], 1)


# ─── 轨迹画布 ───
def create_canvas(h, w):
    """创建带网格线的画布"""
    canvas = np.full((h, w, 3), C['canvas_bg'], dtype=np.uint8)
    # 网格
    step = 40
    for gx in range(0, w, step):
        cv2.line(canvas, (gx, 0), (gx, h), C['canvas_grid'], 1)
    for gy in range(0, h, step):
        cv2.line(canvas, (0, gy), (w, gy), C['canvas_grid'], 1)
    return canvas


def draw_canvas_overlay(frame, canvas, scale=0.28):
    """在右上角绘制画布小窗"""
    fh, fw = frame.shape[:2]
    ch = int(fh * scale)
    cw = int(fw * scale)
    mini = cv2.resize(canvas, (cw, ch), interpolation=cv2.INTER_AREA)

    # 位置：右上角留边
    margin = 12
    x1 = fw - cw - margin
    y1 = margin
    x2 = x1 + cw
    y2 = y1 + ch

    # 投影（带阴影边框）
    shadow = frame.copy()
    cv2.rectangle(shadow, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), -1)
    cv2.addWeighted(shadow, 0.3, frame, 0.7, 0, frame)

    frame[y1:y2, x1:x2] = mini
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), C['canvas_border'], 1, cv2.LINE_AA)

    # 标签
    put_text(frame, "CANVAS", (x1 + 4, y1 + 14), 0.32, C['text_dim'], 1)


# ─── 书写点高亮 ───
def draw_writing_cursor(frame, pos, is_writing, dist=None):
    """绘制当前书写光标"""
    if pos is None or pos == (0, 0):
        return
    x, y = pos
    if is_writing:
        # 外圈 + 内圈
        cv2.circle(frame, (x, y), 14, C['writing'], 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, C['traj_dot'], -1, cv2.LINE_AA)
    else:
        # 悬停指示
        cv2.circle(frame, (x, y), 10, C['hover'], 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, C['hover'], -1, cv2.LINE_AA)


def main():
    print("=" * 50)
    print("  Block A - Palm Writing Real-time Test")
    print("=" * 50)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERR] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['video']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['video']['height'])
    cap.set(cv2.CAP_PROP_FPS, CONFIG['video']['fps'])

    palm_mode = CONFIG.get('palm_writing', {}).get('enabled', False)

    if palm_mode:
        detector = DualHandDetector()
        print("[OK] Dual-hand palm writing mode")
    else:
        detector = HandWritingDetector(gesture_mode=CONFIG['hand_detection']['gesture_mode'])
        print("[OK] Single-hand air writing mode")

    otc = OnlineTraceConverter(smoothing_window=CONFIG['online_trace']['smoothing_window'])
    trajectory = []
    prev_pos = None
    hist_pts = []
    prev_writing = False

    vh, vw = CONFIG['video']['height'], CONFIG['video']['width']
    canvas = create_canvas(vh, vw)

    fps_t0 = time.time()
    fps_cnt = 0
    fps = 0.0

    output_dir = "./results/test"
    os.makedirs(output_dir, exist_ok=True)

    print("\n  ESC/Q  Exit  |  S  Save  |  C  Clear\n")

    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts = time.time() - t0

            if palm_mode:
                is_writing = detector.process(frame, ts)
                cur_pos = detector.get_screen_position()
            else:
                is_writing = detector.process(frame)
                cur_pos = detector.index_tip_position

            # 笔画分割
            if prev_writing and not is_writing:
                valid = [p for p in trajectory if p is not None]
                if len(valid) > 5:
                    otc.end_trace()
                trajectory.clear()
                prev_pos = None
                hist_pts.clear()
            elif not prev_writing and is_writing:
                trajectory.clear()
                prev_pos = None
                hist_pts.clear()

            prev_writing = is_writing

            # 记录轨迹
            if is_writing and cur_pos != (0, 0):
                if prev_pos is not None:
                    sp = smart_smooth(cur_pos, prev_pos, hist_pts)
                    prev_pos = sp
                    trajectory.append(sp)
                    otc.add_point(sp[0], sp[1])
                    if len(trajectory) > 1 and trajectory[-2] is not None:
                        cv2.line(canvas, trajectory[-2], sp, C['canvas_line'], 2, cv2.LINE_AA)
                else:
                    trajectory.append(None)
                    sp = cur_pos
                    prev_pos = cur_pos
                    trajectory.append(sp)
                    otc.add_point(sp[0], sp[1])
            else:
                prev_pos = None
                hist_pts.clear()
                sp = cur_pos

            # FPS
            fps_cnt += 1
            if fps_cnt >= 30:
                fps = fps_cnt / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_cnt = 0

            # 可视化
            if palm_mode:
                draw_palm_plane_visualization(frame, detector)

            draw_debug_info(frame, detector, is_writing, fps, palm_mode)
            draw_canvas_overlay(frame, canvas)

            dist = detector.dist_palm if palm_mode and hasattr(detector, 'dist_palm') else None
            draw_writing_cursor(frame, cur_pos if cur_pos != (0, 0) else None, is_writing, dist)

            cv2.imshow('PalmWrite - Block A', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                ts_str = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(output_dir, f"traj_{ts_str}.png")
                cv2.imwrite(path, canvas)
                print(f"[SAVE] {path}")
            elif key == ord('c'):
                canvas = create_canvas(vh, vw)
                otc = OnlineTraceConverter(smoothing_window=CONFIG['online_trace']['smoothing_window'])
                trajectory.clear()
                print("[CLEAR] Canvas reset")

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()
        traces = otc.get_all_traces()
        total = sum(len(t) for t in traces)
        print(f"\n  Strokes: {len(traces)}  Points: {total}")
        print(f"  Output:  {output_dir}")


if __name__ == "__main__":
    main()
