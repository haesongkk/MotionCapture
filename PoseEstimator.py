import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import deque


class PoseEstimator():
    def __init__(self, video):
        self.vc = cv.VideoCapture(video)
        self.videoFPS = self.vc.get(cv.CAP_PROP_FPS)
        
        self.pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
        smooth_landmarks=True
        )
        
        self.joints = {}
        self.prev_landmarks = None
        self.landmark_window = deque(maxlen=5)
        
        print("ok")
    
    def lms2joint(self, landmarks):
        center = (landmarks[23] + landmarks[24]) / 2
        neck = (landmarks[11] + landmarks[12]) / 2
        
        # hips는 24->23 를 x+ 축으로 돌려야함x
        self.joints["mixamorig:Hips"] = {
            "start": landmarks[24],
            "end": landmarks[23]
        }
        
        # 코어
        self.joints["mixamorig:Spine"] = {
            "start": center,
            "end": neck
        }
        
        # 오른발
        self.joints["mixamorig:RightUpLeg"] = {
            "start": landmarks[24],
            "end": landmarks[26]
        }
        self.joints["mixamorig:RightLeg"] = {
            "start": landmarks[26],
            "end": landmarks[28]
        }
        
        # 왼발
        self.joints["mixamorig:LeftUpLeg"] = {
            "start": landmarks[23],
            "end": landmarks[25]
        }
        self.joints["mixamorig:LeftLeg"] = {
            "start": landmarks[25],
            "end": landmarks[27]
        }
        
        # 오른팔
        self.joints["mixamorig:RightShoulder"] = {
            "start": neck,
            "end": landmarks[12]
        }   
        self.joints["mixamorig:RightArm"] = {
            "start": landmarks[12],
            "end": landmarks[14]
        } 
        self.joints["mixamorig:RightForeArm"] = {
            "start": landmarks[14],
            "end": landmarks[16]
        } 
        
        # 왼팔
        self.joints["mixamorig:LeftShoulder"] = {
            "start": neck,
            "end": landmarks[11]
        }   
        self.joints["mixamorig:LeftArm"] = {
            "start": landmarks[11],
            "end": landmarks[13]
        } 
        self.joints["mixamorig:LeftForeArm"] = {
            "start": landmarks[13],
            "end": landmarks[15]
        } 
        
        for idx, joint in self.joints.items():
            joint["start"][2] = joint["start"][2] * 0.5
            joint["end"][2] = joint["end"][2] * 0.5
        
        
    def update(self):
        if not self.vc.isOpened(): return True
        
        # 프레임 읽기
        valid, frame = self.vc.read()
        if not valid: return True
        
        # 관절 위치 추출
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks: return False
        lms = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        
        # 보정 1
        alpha = 0.5 
        if self.prev_landmarks is None: self.prev_landmarks = lms
        self.prev_landmarks = alpha * self.prev_landmarks + (1 - alpha) * lms
        
        # 보정 2
        self.landmark_window.append(self.prev_landmarks)
        smoothed = np.mean(self.landmark_window, axis=0)
        
        # 위치 -> 방향
        self.lms2joint(smoothed)
        
        h, w, _ = frame.shape
        for joint in self.joints.values():
            p1 = (int(joint["start"][0] * w), int(joint["start"][1] * h))
            p2 = (int(joint["end"][0] * w), int(joint["end"][1] * h))
            cv.line(frame, p1, p2, (0, 255, 0), 2)
        cv.imshow("video", frame)
         
        return False
        
        
