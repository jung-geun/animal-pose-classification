# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 디버그 메시지 끄기

import cv2
from data_sensor import get_img_coord
import angle_out as angle

import yaml
import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from ultralytics import YOLO

with open(
    "model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/config.yaml"
) as f:
    conf_yaml = yaml.load(f, Loader=yaml.FullLoader)

    body_parts = []
    link_parts = []
    for part in conf_yaml["bodyparts"]:
        body_parts.append(part)
    for link in conf_yaml["skeleton"]:
        link_parts.append(link)


frontRightView = ["RightWrist", "RightArm", "Nose"]
frontLeftView = ["LeftWrist", "LeftArm", "Nose"]
backRightView = ["TailTip", "RightFemur", "RightAnkle"]
backLeftView = ["TailTip", "LeftFemur", "LeftAnkle"]
frontRightTilt = ["Neck", "RightWrist", "TailStart"]
frontLeftTilt = ["Neck", "LeftWrist", "TailStart"]
backRightTilt = ["Neck", "TailStart", "RightAnkle"]
backLeftTilt = ["Neck", "TailStart", "LeftAnkle"]
frontRight = ["Neck", "RightArm", "RightWrist"]
frontLeft = ["Neck", "LeftArm", "LeftWrist"]
backRight = ["TailStart", "RightFemur", "RightAnkle"]
backLeft = ["TailStart", "LeftFemur", "LeftAnkle"]
frontBody = ["Nose", "Neck", "TailStart"]
backBody = ["Neck", "TailStart", "TailTip"]
mouth = ["Nose", "MouthCorner", "LowerLip"]
head = ["Nose", "Forehead", "Neck"]
tail = ["TailTip", "TailStart", "LeftAnkle"] or ["TailTip", "TailStart", "RightAnkle"]
direction = ["Nose", "RightArm", "RightFemur"] or ["Nose", "LeftArm", "LeftFemur"]
label_parts = [
    frontRightView,
    frontLeftView,
    backRightView,
    backLeftView,
    frontRightTilt,
    frontLeftTilt,
    backRightTilt,
    backLeftTilt,
    frontRight,
    frontLeft,
    backRight,
    backLeft,
    frontBody,
    backBody,
    mouth,
    head,
    tail,
    direction,
]

# with open("model/xgboost.pkl", "rb") as f:
#     model = pickle.load(f)

with open("model/onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# with open("model/label.json", "rb") as f:
# json_ = json.load(f)
# with open("model/LSTM/model.h5", "rb") as f:

weights_path = "runs/detect/train4/weights/best.pt"

detect_model = YOLO(weights_path)
classfication_model = load_model(f"model/LSTM_15/model.h5")


# %%
def predict(video_path=None, video_save=False):
    video = video_path
    cap = cv2.VideoCapture(video)

    # 영상 기본 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(f"width: {width}, height: {height}, fps: {fps}")

    # 영상의 가공 정보
    window_size = 15
    flag = int(fps / 5)
    count = 1

    if width > height:
        width_tmp = 640
        height_tmp = 480
    else:
        width_tmp = 480
        height_tmp = 640

    if cap.isOpened():
        label = []
        result = ""
        out_video = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            count += 1

            frame = cv2.resize(
                frame, dsize=(width_tmp, height_tmp), interpolation=cv2.INTER_AREA
            )

            results = detect_model.predict(frame)
            detect = results[0]
            box_xywh = detect.boxes.xywh.cpu().numpy()
            x, y, w, h = box_xywh[0]
            img = frame[int(y) : int(y + h), int(x) : int(x + w)]

            point = get_img_coord(img)

            if point is not None:
                print("detect")
            else:
                print("not detect")
                continue
            print(point)

            for p in point:
                if p[2] > 0.01:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255))
            for link in link_parts:
                # print(link)
                p1 = body_parts.index(link[0])
                p2 = body_parts.index(link[1])
                cv2.line(
                    img,
                    (int(point[p1][0]), int(point[p1][1])),
                    (int(point[p2][0]), int(point[p2][1])),
                    (0, 255, 0),
                    2,
                )
            if count % flag == 0:
                print(count)
                label.append(angle.out(point, body_parts, label_parts))

                if len(label) > window_size:
                    label.pop(0)

                if len(label) == window_size:
                    label_ = np.array([label]).reshape(-1, window_size, 18)
                    pred = classfication_model.predict([label_])
                    print(pred)
                    try:
                        la_ = encoder.inverse_transform(pred)
                        result = la_
                        print(f"pred: {la_}")
                    except:
                        print("분류하지 못함")

            cv2.putText(
                img,
                f"result: {result}",
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
                2,
            )
            if video_save:
                out_video.append(img)
    else:
        print("Error")

    cap.release()

    if video_save:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_name = video.split("/")[-1].split(".")[0]
        out = cv2.VideoWriter(
            f"out/{video_name}_predic.mp4", fourcc, fps, (width_tmp, height_tmp)
        )
        print(len(out_video))
        for i in range(len(out_video)):
            out.write(out_video[i])
        out.release()

    cv2.destroyAllWindows()


predict(
    video_path="model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/videos/dog-walkrun-085352.avi",
    video_save=True,
)

# %%
