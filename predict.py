import cv2

import numpy as np
import yaml
import pickle
from time import sleep

from ultralytics import YOLO
from angle_out import out
from tensorflow import keras
from tensorflow.keras.models import load_model
from data_sensor import get_img_coord, ini_DLC

pose_cfg, cfg = (
    "model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/dlc-models/iteration-0/preLabeled_type(Labrador_Retriever)Sep19-trainset95shuffle1/train/pose_cfg.yaml",
    "model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/config.yaml",
)
ini_DLC(pose_cfg, cfg)

# body_parts = [
#     "nose",
#     "upper_jaw",
#     "lower_jaw",
#     "mouth_end_right",
#     "mouth_end_left",
#     "right_eye",
#     "right_earbase",
#     "right_earend" "right_antler_base",
#     "right_antler_end",
#     "left_eye",
#     "left_earbase",
#     "left_earend",
#     "left_antler_base",
#     "left_antler_end",
#     "neck_base",
#     "neck_end",
#     "throat_base",
#     "throat_end",
#     "back_base",
#     "back_end",
#     "back_middle",
#     "tail_base",
#     "tail_end",
#     "front_left_thai",
#     "front_left_knee",
#     "front_left_paw",
#     "front_right_thai",
#     "front_right_knee",
#     "front_right_paw",
#     "back_left_paw",
#     "back_left_thai",
#     "back_right_thai",
#     "back_left_knee",
#     "back_right_knee",
#     "back_right_paw",
#     "belly_bottom",
#     "body_middle_right",
#     "body_middle_left",
# ]

# label_parts = [
#     # 오른쪽 손목, 오른쪽 팔, 코
#     ["front_right_paw", "front_left_thai", "nose"],
#     # 왼쪽 손목, 왼쪽 팔, 코
#     ["front_left_paw", "front_left_thai", "nose"],
#     # 꼬리 끝, 오른쪽 허벅지, 오른쪽 발목
#     ["tail_base", "back_right_thai", "back_right_paw"],
#     # 꼬리 끝, 왼쪽 허벅지, 왼쪽 발목
#     ["tail_base", "back_left_thai", "back_left_paw"],
#     # 목, 오른쪽 손목, 꼬리 시작
#     ["neck_base", "front_right_paw", "tail_base"],
#     # 목, 왼쪽 손목, 꼬리 시작
#     ["neck_base", "front_left_paw", "tail_base"],
#     # 목, 꼬리 시작, 오른쪽 발목
#     ["neck_base", "tail_base", "back_right_paw"],
#     # 목, 꼬리 시작, 왼쪽 발목
#     ["neck_base", "tail_base", "back_left_paw"],
#     # 목, 오른쪽 팔, 오른쪽 손목
#     ["neck_base", "front_right_thai", "front_right_paw"],
#     # 목, 왼쪽 팔, 왼쪽 손목
#     ["neck_base", "front_left_thai", "front_left_paw"],
#     # 꼬리 시작, 오른쪽 허벅지, 오른쪽 발목
#     ["tail_base", "back_right_thai", "back_right_paw"],
#     # 꼬리 시작, 왼쪽 허벅지, 왼쪽 발목
#     ["tail_base", "back_left_thai", "back_left_paw"],
#     # 코, 목, 꼬리 시작
#     ["nose", "neck_base", "tail_base"],
#     # 목, 꼬리 시작, 꼬리 끝
#     ["neck_base", "tail_base", "tail_end"],
#     # 코, 입 모서리, 아랫입술
#     ["nose", "mouth_end_right", "lower_jaw"],
#     # 코, 이마, 목
#     ["nose", "upper_jaw", "neck_base"],
#     # 꼬리 끝, 꼬리 시작, 왼쪽 발목 or 꼬리 끝, 꼬리 시작, 오른쪽 발목
#     ["tail_end", "tail_base", "back_left_paw"]
#     or ["tail_end", "tail_base", "back_right_paw"],
#     # 코, 오른쪽 팔, 오른쪽 허벅지 or 코, 왼쪽 팔, 왼쪽 허벅지
#     ["nose", "front_right_thai", "front_right_paw"]
#     or ["nose", "front_left_thai", "front_left_paw"],
# ]

# link_parts = [
#     ["front_right_paw", "front_left_thai"],
#     ["front_left_paw", "front_left_thai"],
#     ["tail_base", "back_right_thai"],
#     ["tail_base", "back_left_thai"],
#     ["neck_base", "front_right_paw"],
#     ["neck_base", "front_left_paw"],
#     ["neck_base", "tail_base"],
#     ["neck_base", "tail_base"],
#     ["neck_base", "front_right_thai"],
#     ["neck_base", "front_left_thai"],
#     ["tail_base", "back_right_thai"],
#     ["tail_base", "back_left_thai"],
#     ["nose", "neck_base"],
#     ["neck_base", "tail_base"],
#     ["nose", "mouth_end_right"],
#     ["nose", "upper_jaw"],
#     ["tail_end", "tail_base"],
#     ["nose", "front_right_thai"],
#     ["nose", "front_left_thai"],
# ]


body_parts = []
link_parts = []

with open(
    "model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/config.yaml"
) as f:
    conf_yaml = yaml.load(f, Loader=yaml.FullLoader)
    for part in conf_yaml["bodyparts"]:
        body_parts.append(part)
    for link in conf_yaml["skeleton"]:
        link_parts.append(link)


frontRightView = ["RightWrist", "RightArm", "Nose"]
# 오른쪽 손목, 오른쪽 팔, 코
frontLeftView = ["LeftWrist", "LeftArm", "Nose"]
# 왼쪽 손목, 왼쪽 팔, 코
backRightView = ["TailTip", "RightFemur", "RightAnkle"]
# 꼬리 끝, 오른쪽 허벅지, 오른쪽 발목
backLeftView = ["TailTip", "LeftFemur", "LeftAnkle"]
# 꼬리 끝, 왼쪽 허벅지, 왼쪽 발목
frontRightTilt = ["Neck", "RightWrist", "TailStart"]
# 목, 오른쪽 손목, 꼬리 시작
frontLeftTilt = ["Neck", "LeftWrist", "TailStart"]
# 목, 왼쪽 손목, 꼬리 시작
backRightTilt = ["Neck", "TailStart", "RightAnkle"]
# 목, 꼬리 시작, 오른쪽 발목
backLeftTilt = ["Neck", "TailStart", "LeftAnkle"]
# 목, 꼬리 시작, 왼쪽 발목
frontRight = ["Neck", "RightArm", "RightWrist"]
# 목, 오른쪽 팔, 오른쪽 손목
frontLeft = ["Neck", "LeftArm", "LeftWrist"]
# 목, 왼쪽 팔, 왼쪽 손목
backRight = ["TailStart", "RightFemur", "RightAnkle"]
# 꼬리 시작, 오른쪽 허벅지, 오른쪽 발목
backLeft = ["TailStart", "LeftFemur", "LeftAnkle"]
# 꼬리 시작, 왼쪽 허벅지, 왼쪽 발목
frontBody = ["Nose", "Neck", "TailStart"]
# 코, 목, 꼬리 시작
backBody = ["Neck", "TailStart", "TailTip"]
# 목, 꼬리 시작, 꼬리 끝
mouth = ["Nose", "MouthCorner", "LowerLip"]
# 코, 입 모서리, 아랫입술
head = ["Nose", "Forehead", "Neck"]
# 코, 이마, 목
tail = ["TailTip", "TailStart", "LeftAnkle"] or ["TailTip", "TailStart", "RightAnkle"]
# 꼬리 끝, 꼬리 시작, 왼쪽 발목 or 꼬리 끝, 꼬리 시작, 오른쪽 발목
direction = ["Nose", "RightArm", "RightFemur"] or ["Nose", "LeftArm", "LeftFemur"]
# 코, 오른쪽 팔, 오른쪽 허벅지 or 코, 왼쪽 팔, 왼쪽 허벅지

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

# print("body_parts > ", body_parts)
# print("link_parts > ", link_parts)


def main():
    # weights_path = "_mina/project/YOLO_test_train/weights/best.pt"
    weights_path = "runs/detect/train4/weights/best.pt"

    yolo_model = YOLO(weights_path)
    window_size = 15
    classfier_model = load_model(f"model/LSTM_{window_size}/model.h5")

    with open("model/onehot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # video_name = "model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/videos/dog-walkrun-085352.avi"
    video_name = "_mina/data/AI-Hub/poseEstimation/Validation/DOG/sourceBODYLOWER/videos/dog-bodylower-079495.avi"

    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    time_seq_avg = 0

    if width > height:
        width_ = 640
        height_ = 480
    else:
        width_ = 480
        height_ = 640

    print(f"width: {width}, height: {height}, fps: {fps}")
    print(f"width_: {width_}, height_: {height_}")

    # threshold = 0.5

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_name = video_name.split("/")[-1].split(".")[0]

    out_writer = cv2.VideoWriter(
        f"out/{video_name}_predic.mp4", fourcc, fps, (width_, height_)
    )
    count = 0
    flag = int(fps // 5)  # 초당 5개의 프레임을 추출
    seq = []
    while True:
        start_time = cv2.getTickCount()
        ret, frame = cap.read()
        count += 1

        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        img = frame.copy()
        img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

        results = yolo_model.predict(img, imgsz=640)
        # results = model.track(img, imgsz=640, show=True, iou=threshold)

        detect = results[0]
        box_xywh = detect.boxes.xywh.cpu().numpy()

        # img_ = img.copy()
        # print(f"박스의 좌표 > {box_xywh}")
        if len(box_xywh) == 0:
            continue
        else:
            x, y, w, h = box_xywh[0]
            img = img[int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)]
        img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

        # print("img shape ", img_.shape)

        point = get_img_coord(img)

        for p in point:
            if p[2] > 0.01:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255))
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

        # print("point > ", point)

        if count % flag == 0:
            print(f"count: {count}")
            inputs = []
            for i, p in enumerate(point):
                # print(f"p > {p}")
                inputs.append({"x": p[0], "y": p[1], "likelihood": p[2]})
            seq.append(out(inputs, body_parts=body_parts, label_parts=label_parts))

            if len(seq) > window_size:
                seq.pop(0)

            if len(seq) == window_size:
                seq_ = np.array(seq)
                seq_ = seq_.reshape(1, window_size, -1)
                pred = classfier_model.predict(seq_)
                label = encoder.inverse_transform(pred)
                # print("label ? ", label)
                cv2.putText(
                    img,
                    label[0][0],
                    (int(width_ / 2), int(height_ / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        # out_video.append(img_)
        end_time = cv2.getTickCount()
        out_writer.write(img)
        time_seq = (end_time - start_time) / cv2.getTickFrequency()
        if count > 4:
            time_seq_avg += time_seq

        # img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("img", img)
        print(f"프레임 수행 시간: {time_seq}")

        sleep(1 / (fps + 1))

    # 사용한 자원을 해제합니다.

    # cv2.imwrite(f"out/{video_name}_predic.jpg", img_)
    print(f"평균 프레임 수행 시간: {time_seq_avg / (count-4)}")
    # for frame in out_video:

    cap.release()
    out_writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
