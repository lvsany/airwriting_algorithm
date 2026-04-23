import cv2
import mediapipe as mp
import numpy as np
import yaml
import os

# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)

class HandWritingDetector:
    def __init__(self, static_image_mode=None, max_num_hands=None,
                 min_detection_confidence=None, min_tracking_confidence=None, gesture_mode=None):
        """
        初始化手势书写检测器
        """
        # 从配置文件读取参数，如果传入参数则使用传入的参数
        hand_config = CONFIG['hand_detection']
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode if static_image_mode is not None else hand_config['static_image_mode'],
            max_num_hands=max_num_hands if max_num_hands is not None else hand_config['max_num_hands'],
            min_detection_confidence=min_detection_confidence if min_detection_confidence is not None else hand_config['min_detection_confidence'],
            min_tracking_confidence=min_tracking_confidence if min_tracking_confidence is not None else hand_config['min_tracking_confidence']
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.is_writing = False
        self.index_tip_position = (0, 0)
        self.landmarks = None
        self.gesture_mode = gesture_mode if gesture_mode is not None else hand_config['gesture_mode']

    @staticmethod
    def calculate_distance(point1, point2):
        """计算两点之间的欧式距离"""
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    def is_finger_straight(self, finger_tip, finger_root, finger_joints, threshold=None):
        """
        判断手指是否伸直
        """
        if threshold is None:
            threshold = CONFIG['hand_detection']['finger_straightness_threshold']
            
        if self.landmarks is None:
            return False

        tip_to_root = self.calculate_distance(self.landmarks[finger_tip], self.landmarks[finger_root])

        finger_length = sum(self.calculate_distance(self.landmarks[finger_joints[i]], self.landmarks[finger_joints[i + 1]])
                            for i in range(len(finger_joints) - 1))

        return tip_to_root > finger_length * threshold if finger_length >= 0.05 else False

    # 拇指和食指捏合
    def is_thumbtip_indextip_closed(self, threshold=None):
        if threshold is None:
            threshold = CONFIG['hand_detection']['thumb_index_close_threshold']
            
        if self.landmarks is None:
            return False
        return self.calculate_distance(self.landmarks[4], self.landmarks[8]) <= threshold

    # 拇指按住食指中部关节，其余手指弯曲
    def is_thumb_index_middle_joint_pressed(self, threshold=None):
        if threshold is None:
            threshold = CONFIG['hand_detection']['thumb_index_middle_press_threshold']
            
        if self.landmarks is None:
            return False
        return self.calculate_distance(self.landmarks[4], self.landmarks[6]) <= threshold


    # 拇指与食指中指捏合
    def is_two_finger_pinch(self, threshold=None):
        """判断拇指与食指和中指捏合"""
        if threshold is None:
            threshold = CONFIG['hand_detection']['two_finger_pinch_threshold']
            
        if self.landmarks is None:
            return False
        thumb_to_index = self.calculate_distance(self.landmarks[4], self.landmarks[8]) <= threshold
        thumb_to_middle = self.calculate_distance(self.landmarks[4], self.landmarks[12]) <= threshold
        ring_bent = not self.is_finger_straight(16, 13, [13, 14, 15, 16])
        pinky_bent = not self.is_finger_straight(20, 17, [17, 18, 19, 20])
        return thumb_to_index and thumb_to_middle and ring_bent and pinky_bent

    # 食指和拇指伸直
    def is_thumb_index_straight(self):
        thumb_straight = self.is_finger_straight(4, 1, [1, 2, 3, 4], 0.8)
        index_straight = self.is_finger_straight(8, 5, [5, 6, 7, 8])
        middle_bent = not self.is_finger_straight(12, 9, [9, 10, 11, 12])
        ring_bent = not self.is_finger_straight(16, 13, [13, 14, 15, 16])
        pinky_bent = not self.is_finger_straight(20, 17, [17, 18, 19, 20])

        return index_straight 
        
    def update_writing_status(self):
        """更新书写状态"""
        if self.landmarks is None:
            self.is_writing = False
            return
            
        """当食指和拇指捏合时"""
        if self.gesture_mode == 1:
            self.is_writing = self.is_thumbtip_indextip_closed()
        if self.gesture_mode == 2:
            self.is_writing = self.is_thumb_index_middle_joint_pressed()
        if self.gesture_mode == 3:
            self.is_writing = self.is_two_finger_pinch()
        if self.gesture_mode == 4:
            self.is_writing = self.is_thumb_index_straight()

    def process(self, frame):
        """处理输入帧并检测手势"""
        self.frame_shape = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        self.landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.landmarks = hand_landmarks.landmark
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                self.update_writing_status()
                if self.gesture_mode == 1:
                    self.index_tip_position = (int((self.landmarks[4].x + self.landmarks[8].x)/2 * self.frame_shape[1]),
                                           int((self.landmarks[4].y + self.landmarks[8].y)/2 * self.frame_shape[0]))
                elif self.gesture_mode == 2:
                    self.index_tip_position = (int(self.landmarks[8].x  * self.frame_shape[1]),
                                           int(self.landmarks[8].y * self.frame_shape[0]))
                                    
                elif self.gesture_mode == 3:
                    self.index_tip_position = (int((self.landmarks[4].x + self.landmarks[8].x + self.landmarks[12].x)/3 * self.frame_shape[1]),
                                           int((self.landmarks[4].y + self.landmarks[8].y + self.landmarks[12].y)/3 * self.frame_shape[0]))
                elif self.gesture_mode == 4:
                    self.index_tip_position = (int(self.landmarks[8].x  * self.frame_shape[1]),
                                           int(self.landmarks[8].y * self.frame_shape[0]))

        return self.is_writing
