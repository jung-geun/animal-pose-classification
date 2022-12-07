import os

import json
import pandas as pd
import numpy as np
import cv2
import time
import mediapipe as mp


def get_csv():
    x_train = []
    y_train = []

    data = []

    # 파일 경로 설정
    path = "./frames"
    
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except:
        print(f"Error: {path} 폴더 생성 실패")
        

    # y 값이 될 리스트 추출
    # 디렉토리 명을 Y 로 지정하였음
    y_list = os.listdir(path)

    # print(y_list)
    # y_list = ["goddess", "tree", "warrior2", "plank", "downdog"]

    for y in y_list:
        print(y)
        y_path = path + "/" + y
        x_list = os.listdir(y_path)
        # print(y , ">" , x_list)
        for v in x_list:
            # print(v)
            v = y_path + "/" + v
            if os.path.isdir(v):
                _dir = os.listdir(v)
                for data in _dir:
                # print(f"{v}/{data}")
                # print(y)
                    if os.path.splitext(data)[1] in [".png", ".jpg"]:
                        x_train.append(f"{v}/{data}")
                        y_train.append(y)
                        data = pd.DataFrame({"image": x_train, "pose": y_train})
            else:
                if os.path.splitext(v)[1] in [".png", ".jpg"]:
                    x_train.append(v)
                    y_train.append(y)
                    data = pd.DataFrame({"image": x_train, "pose": y_train})

    x_train = pd.DataFrame(x_train)
    # print(x_train.shape)
    y_train = pd.DataFrame(y_train)
    # print(y_train.shape)

    # print(x_train.info())
    # print(y_train.info())

    # print(x_train.value_counts())
    # print(y_train.value_counts())
    print(data.info())
    # print(data.value_counts())
    data.to_csv("data.csv", index=False)

get_csv()

def img_media(self=None, img_path=None, csv_path="./data.csv", json_path="./pose.json"):
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    IMAGE_FILES = []
    # IMAGE_FILES.clear()

    try:
        if not img_path == None:
            for v in img_path:
                IMAGE_FILES.append(v)
        else:
            data = pd.read_csv(csv_path)
            # print(data)

            for index, row in data.iterrows():
                # if row["pose"] in ["goddess", "tree", "warrior2", "plank", "downdog"]:
                    # print(os.path.splitext(row["image"]))
                if os.path.splitext(row["image"])[1] in [".png", ".jpg"]:
                    r = {"pose": row["pose"], "image": row["image"]}
                    IMAGE_FILES.append(r)
                        # print(IMAGE_FILES)
    except:
        return -2

    x = []

    try:
        if not self == None:
            # print(len(IMAGE_FILES))
            __len = len(IMAGE_FILES)
            self.progressbar_pre.setMaximum(__len - 1)
            # return
    except:
        return -3

    try:

        BG_COLOR = (192, 192, 192)  # 회색
        # mediapipe 의 pose 모델 사용
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5) as pose:
            for idx, file in enumerate(IMAGE_FILES):
                # print(idx)
                image = cv2.imread(file["image"])

                # 파일
                # print(f"file : {file}")

                # image_height, image_width, _ = image.shape

                # 처리 전 BGR 이미지를 RGB로 변환합니다.
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # pose 인식에 실패하면 넘어감
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

                # 정상적으로 인식 되는지 확인
                # print(
                #     f"arm_left > {arm_left}, \n"
                #     f"arm_right > {arm_right}, \n"
                #     f"leg_left > {leg_left}, \n"
                #     f"leg_right > {leg_right}, \n"
                # )

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

                a = {"arm_left": arm_left, "arm_right": arm_right,
                     "leg_left": leg_left, "leg_right": leg_right, "pose": file["pose"]}

                x.append(a)

                print(f'{idx} / {__len - 1}')
                if not self == None:
                    # self.preprocess_list.appendPlainText(f"img : {file['image']} ")
                    self.progressbar_pre.setValue(idx)
    except:
        return -4
    now = time

    try:
        with open(json_path, 'w') as outfile:
            json.dump(x, outfile)
            now = now.strftime("%Y %m %d %H %M %S")

            self.preprocess_list.appendPlainText(
                f"json 파일 저장됨 : {json_path} {now}")
    except:
        return -1

    return 1

# img_media()
