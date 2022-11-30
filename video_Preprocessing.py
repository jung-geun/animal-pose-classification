import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json

def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
        

def img_media(img_path = None ,csv_path = "./data.csv", json_path = "./pose.json"):
    IMAGE_FILES = []
    # IMAGE_FILES.clear()
    
    if not img_path == None:
        for v in path:
            IMAGE_FILES.append(v)
    data = pd.read_csv(csv_path)
    # print(data)
    
    x = []
    
    for index, row in data.iterrows():
        if row["pose"] in  ["goddess", "tree", "warrior2", "plank", "downdog"]:
            # print(os.path.splitext(row["image"]))
            if os.path.splitext(row["image"])[1] in [".png", ".jpg"]:
                r = {"pose" : row["pose"], "image" : row["image"]}
                IMAGE_FILES.append(r)
                # print(IMAGE_FILES)
        
    BG_COLOR = (192, 192, 192)  # 회색
    # mediapipe 의 pose 모델 사용
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            print(idx)
            image = cv2.imread(file["image"])
            
            # 파일
            print(f"file : {file}")
            
            # image_height, image_width, _ = image.shape
            
            # 처리 전 BGR 이미지를 RGB로 변환합니다.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # pose 인식에 실패하면 넘어감
            if not results.pose_landmarks:
                continue
            
            # 각도 도출을 위한 파트 추출
            # 왼 팔 ( 오른쪽 어깨 - 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼 손 ) 
            arm_left = {"x":[results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[15].x],
                        "y":[results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[15].y]}
            
            # 오른 팔 ( 왼쪽 어깨 - 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른 손 )
            arm_right = {"x":[results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[16].x],
                         "y":[results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[16].y]}

            # 왼 다리 ( 왼쪽 어깨 - 왼쪽 허리 - 왼쪽 무릎 - 왼 발 )
            leg_left = {"x":[results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[23].x, results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[27].x],
                        "y":[results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[23].y, results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[27].y]}

            # 오른 다리 ( 오른쪽 어깨 - 오른쪽 허리 - 오른쪽 무릎 - 오른 발 )
            leg_right = {"x":[results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[28].x],
                         "y":[results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[24].y, results.pose_landmarks.landmark[26].y, results.pose_landmarks.landmark[28].y]}

            # 정상적으로 인식 되는지 확인
            print(
                f"arm_left > {arm_left}, \n"
                f"arm_right > {arm_right}, \n"
                f"leg_left > {leg_left}, \n"
                f"leg_right > {leg_right}, \n"
            )
            
            
            # 이미지 출력을 위한 부분
            # python 파일 실행시 필요 없음
            # annotated_image = image.copy()
            # 이미지를 분할합니다.
            # 경계 주변의 분할을 개선하려면 "image"가 있는
            # "results.segmentation_mask"에 공동 양방향 필터를 적용하는 것이 좋습니다.
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # bg_image = np.zeros(image.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, annotated_image, bg_image)
            
            # 이미지 위에 포즈 랜드마크를 그립니다.
            # mp_drawing.draw_landmarks(
                # annotated_image,
                # results.pose_landmarks,
                # mp_pose.POSE_CONNECTIONS,
                # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # cv2.imwrite('/tmp/annotated_image' +
                        # str(idx) + '.png', annotated_image)


            # 포즈 월드 랜드마크를 그립니다. (3D)
            # mp_drawing.plot_landmarks(
                # results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

            # 이미지 테스트 출력
            # plt_imshow(["Original", "Find Faces"], [image, annotated_image], figsize=(16,10))

            a = {"arm_left" : arm_left, "arm_right" : arm_right, "leg_left" : leg_left, "leg_right" : leg_right, "pose" : file["pose"]}

            x.append(a)
                    
    with open(json_path, 'w') as outfile:
        json.dump(x, outfile)
        
img_media()