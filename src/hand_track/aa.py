import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# 从你原有的文件中导入检测器类
# 假设你的原文件名为 contact_detector.py
from contact_detector import MultiFeatureContactDetector

# ── 掌面拟合与特征提取辅助函数 (从你原代码中提取) ──────────────────────────
_PALM_IDX = [0, 1, 5, 9, 13, 17]
_BOUND_IDX = list(range(21))

def fit_palm_plane(world_lms):
    pts = np.array([[world_lms[i].x, world_lms[i].y, world_lms[i].z] for i in _PALM_IDX])
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin)
    normal = vt[-1]
    return origin, normal

def tip_to_plane_mm(world_lms, origin, normal, tip_idx=8):
    tip = np.array([world_lms[tip_idx].x, world_lms[tip_idx].y, world_lms[tip_idx].z])
    dist_m = abs(float(np.dot(tip - origin, normal)))
    return dist_m * 1000.0

def tip_in_palm_region(norm_lms_write, norm_lms_palm, shape, margin=0.08):
    h, w = shape[:2]
    palm_pts = np.array([[norm_lms_palm[i].x * w, norm_lms_palm[i].y * h] for i in _BOUND_IDX], dtype=np.float32)
    hull = cv2.convexHull(palm_pts)
    tip = norm_lms_write[8]
    tx, ty = tip.x * w, tip.y * h
    return cv2.pointPolygonTest(hull, (tx, ty), False) >= -margin * w

def brightness(gray, cx, cy, r=18):
    roi = gray[max(cy-r,0):cy+r, max(cx-r,0):cx+r]
    return float(np.mean(roi)) if roi.size else 128.0

def dist2d_palm(norm_w, norm_p, shape):
    h, w = shape[:2]
    tx = norm_w[8].x * w; ty = norm_w[8].y * h
    return {i: float(np.hypot(norm_p[i].x*w - tx, norm_p[i].y*h - ty)) for i in (0, 5, 9, 13, 17)}


# ── 视频预测主函数 ────────────────────────────────────────────────────────
def process_video_with_model(video_path: str, model_path: str, output_path: str):
    print(f"正在加载视频: {video_path}")
    print(f"正在加载模型: {model_path}")
    
    # 1. 初始化检测器 (使用 model 模式)
    det = MultiFeatureContactDetector(mode="model", model_path=model_path)
    
    # 2. 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    # 3. 初始化视频读写
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] 无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    # 特征缓冲区
    DIST_WIN, SIGMA_WIN = 5, 8
    _dbuf = deque(maxlen=DIST_WIN + 2)
    _sbuf = deque(maxlen=SIGMA_WIN)

    def _push(d):  _dbuf.append(d); _sbuf.append(d)
    def _vn(): return float(np.mean(np.diff(list(_dbuf)[-DIST_WIN:]))) if len(_dbuf)>=2 else 0.0
    def _sd(): return float(np.std(list(_sbuf))) if len(_sbuf)>1 else 0.0
    def _clear(): _dbuf.clear(); _sbuf.clear()

    frame_idx = 0
    print(f"开始处理，共 {total_frames} 帧...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = hands.process(rgb)

        write_lm = write_wlm = palm_lm = palm_wlm = None
        dist = vn = sd = 0.0

        # 处理关键点
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, wlm, hd in zip(res.multi_hand_landmarks, res.multi_hand_world_landmarks, res.multi_handedness):
                label = hd.classification[0].label
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                if label == "Right": # 依据你原代码的左右手分配规则
                    palm_lm, palm_wlm = lm, wlm
                else:
                    write_lm, write_wlm = lm, wlm

            if palm_wlm and write_wlm:
                origin, normal = fit_palm_plane(palm_wlm.landmark)
                if tip_in_palm_region(write_lm.landmark, palm_lm.landmark, frame.shape):
                    dist = tip_to_plane_mm(write_wlm.landmark, origin, normal)
                    _push(dist)
                    vn, sd = _vn(), _sd()
                else:
                    dist = None
                    _clear()

                d2d = dist2d_palm(write_lm.landmark, palm_lm.landmark, frame.shape)
                tip_px = (int(write_lm.landmark[8].x * fw), int(write_lm.landmark[8].y * fh))
                br = brightness(gray, *tip_px)
            else:
                dist = None; _clear()
                d2d = {}; br = None; tip_px = None
        else:
            dist = None; _clear()
            d2d = {}; br = None; tip_px = None

        # 4. 更新检测器并获取预测状态
        det.update(dist_raw=dist, v_n=vn, sigma_d=sd, dist2d_palm=d2d, brightness=br)

        # 5. 渲染可视化画面
        if tip_px:
            col = (60, 60, 220) if det.is_contact() else (60, 160, 220)
            cv2.circle(frame, tip_px, 14, col, 2, cv2.LINE_AA)
            cv2.circle(frame, tip_px, 4,  col, -1, cv2.LINE_AA)

        state_str = "CONTACT" if det.is_contact() else "IDLE"
        score = det.get_score()
        
        # 绘制文本信息
        cv2.putText(frame, state_str, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60,60,220) if det.is_contact() else (100,100,110), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Model Prob: {score:.3f}", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,190), 1, cv2.LINE_AA)
        dist_str = f"dist: {dist:.1f} mm" if dist is not None else "dist: --"
        cv2.putText(frame, dist_str, (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,190), 1, cv2.LINE_AA)
        
        # 写入进度打印
        if frame_idx % 30 == 0:
            print(f"进度: {frame_idx}/{total_frames} 帧")

        out.write(frame)

    cap.release()
    out.release()
    hands.close()
    print(f"处理完成！输出视频已保存至: {output_path}")


if __name__ == "__main__":
    # 使用示例
    # 替换为你实际的 视频路径、pkl模型路径 和 期望输出的路径
    VIDEO_IN = "test1.mp4"
    MODEL_IN = "models/aa.pkl" 
    VIDEO_OUT = "result_video.mp4"
    
    process_video_with_model(VIDEO_IN, MODEL_IN, VIDEO_OUT)