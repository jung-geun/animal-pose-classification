import pickle
import cv2
import math
from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def cal_rad(arr):
    rad = []

    a = math.atan2(arr["x"][0] - arr["x"][1], arr["y"][0] - arr["y"][1]) - \
        math.atan2(arr["x"][1] - arr["x"][2], arr["y"][1] - arr["y"][2])
    # print(a)
    rad.append(a)
    b = math.atan2(arr["x"][1] - arr["x"][2], arr["y"][1] - arr["y"][2]) - \
        math.atan2(arr["x"][2] - arr["x"][3], arr["y"][2] - arr["y"][3])
    rad.append(b)

    PI = math.pi

    deg = [(rad[0]*180)/PI, (rad[1]*180)/PI]
    # print(deg[0])

    return deg


def real_infereance(self=None, cam=0):

    try:
        with open(f'./model/model_{self.model}.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        return -2
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # try:

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        
        cap = cv2.VideoCapture(cam)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
                continue

            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                continue

            # 각도 도출을 위한 파트 추출
            # 왼 팔 ( 오른쪽 어깨 - 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼 손 )
            arm_left = {"x": [results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[15].x],
                        "y": [results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[15].y]}

            # 오른 팔 ( 왼쪽 어깨 - 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른 손 )
            arm_right = {"x": [results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[16].x],
                         "y": [results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[16].y]}

            # 왼 다리 ( 왼쪽 어깨 - 왼쪽 허리 - 왼쪽 무릎 - 왼 발 )
            leg_left = {"x": [results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[23].x, results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[27].x],
                        "y": [results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[23].y, results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[27].y]}

            # 오른 다리 ( 오른쪽 어깨 - 오른쪽 허리 - 오른쪽 무릎 - 오른 발 )
            leg_right = {"x": [results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[28].x],
                         "y": [results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[24].y, results.pose_landmarks.landmark[26].y, results.pose_landmarks.landmark[28].y]}

            x = np.array([cal_rad(arm_left)[0], cal_rad(arm_left)[1], cal_rad(arm_right)[0], cal_rad(arm_right)[
                1], cal_rad(leg_left)[0], cal_rad(leg_left)[1], cal_rad(leg_right)[0], cal_rad(leg_right)[1]])

            y_pred = model.predict(x.reshape(1, -1))

            # 포즈 주석을 이미지 위에 그립니다.
            image.flags.writeable = True

            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 보기 편하게 이미지를 좌우 반전합니다.
            # cv2.imshow(f'{y_pred}', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            if self.flag == 1:
                cap.release()
                
                break

            h, w, c = image.shape
            qlmg = QImage(image.data, w, h, w*c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qlmg)

            p = pixmap.scaled(480, 360, Qt.IgnoreAspectRatio)
            self.label.setPixmap(p)
            self.label_value.setText(f'결과 : {y_pred}')

    cap.release()
