import cv2
import os
import numpy as np
import yaml
import pickle
from time import sleep
from tqdm import tqdm

from ultralytics import YOLO
from angle_out import out
from tensorflow import keras
from tensorflow.keras.models import load_model
from data_sensor import get_img_coord, ini_DLC

project_path = "model/dog_-_77503 (720p)-1-2023-10-24"

pose_config_path_iter = f"{project_path}/dlc-models/iteration-0"
pose_config_path_train = (
    f"{pose_config_path_iter}/{os.listdir(pose_config_path_iter)[0]}/train"
)

os.listdir(pose_config_path_train)
config_path = f"{project_path}/config.yaml"
pose_config_path = f"{pose_config_path_train}/pose_cfg.yaml"

weights_path = f"{pose_config_path_train}/snapshot-100000"


ini_DLC(pose_config_path, config_path, weights_path)


# label_parts = [
#     ["Nose", "MouthCorner", "LowerLip"],
#     ["LowerLip", "Nose", "Forehead"],
#     ["Nose", "Forehead", "Neck"],
#     ["Forehead", "Neck", "TailStart"],
#     ["Neck", "TailStart", "TailTip"],
#     ["Forehead", "Neck", "RightArm"],
#     ["Neck", "RightArm", "RightWrist"],
#     ["Forehead", "Neck", "LeftArm"],
#     ["Neck", "LeftArm", "LeftWrist"],
#     ["Neck", "TailStart", "RightFemur"],
#     ["Neck", "RightFemur", "RightAnkle"],
#     ["Neck", "TailStart", "LeftFemur"],
#     ["Neck", "LeftFemur", "LeftAnkle"],
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

with open(config_path) as f:
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
    # weights_path = "runs/detect/train4/weights/best.pt"

    # yolo_model = YOLO(weights_path)
    window_size = 15
    classification_model = load_model(f"model/LSTM_{window_size}_18/model.h5")

    with open("model/onehot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # video_name = "model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/videos/dog-walkrun-085352.avi"
    video_name = "model/dog_-_77503 (720p)-1-2023-10-24/videos/dog_-_77503 (720p).mp4"

    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_seq_avg = 0

    if width > height:
        width_ = 640
        height_ = 480
    else:
        width_ = 480
        height_ = 640

    print(f"width: {width}, height: {height}, fps: {fps}")
    print(f"width_: {width_}, height_: {height_}")

    threshold = 0.05

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_name = video_name.split("/")[-1].split(".")[0]

    out_writer = cv2.VideoWriter(
        f"out/{video_name}_predic.mp4", fourcc, fps, (width_, height_)
    )
    count = 0
    flag = int(fps // 5)  # 초당 5개의 프레임을 추출
    seq = []
    label = [["None"]]

    pbar = tqdm(iterable=range(total_frame), desc="frame", unit="frame")
    try:
        while True:
            start_time = cv2.getTickCount()
            ret, frame = cap.read()
            count += 1

            if not ret:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            img = frame.copy()
            img = cv2.resize(
                img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR
            )

            # results = yolo_model.predict(img, imgsz=640)
            # results = model.track(img, imgsz=640, show=True, iou=threshold)

            # detect = results[0]
            # box_xywh = detect.boxes.xywh.cpu().numpy()

            # img_ = img.copy()
            # print(f"박스의 좌표 > {box_xywh}")
            # if len(box_xywh) == 0:
            # continue
            # else:
            # x, y, w, h = box_xywh[0]
            # img = img[int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)]
            # img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

            # print("img shape ", img_.shape)

            point = get_img_coord(img)

            for p in point:
                if p[2] > threshold:
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255))
            for link in link_parts:
                if (
                    point[body_parts.index(link[0])][2] < threshold
                    or point[body_parts.index(link[1])][2] < threshold
                ):
                    continue
                # print(link)
                p1 = body_parts.index(link[0])
                p2 = body_parts.index(link[1])
                cv2.line(
                    img,
                    (int(point[p1][0]), int(point[p1][1])),
                    (int(point[p2][0]), int(point[p2][1])),
                    (0, 255, 0),
                    1,
                )

            # print("point > ", point)

            if count % flag == 0:
                # print(f"count: {count}")
                inputs = []
                for i, p in enumerate(point):
                    # print(f"p > {p}")
                    inputs.append({"x": p[0], "y": p[1], "likelihood": p[2]})
                seq.append(
                    out(
                        inputs,
                        body_parts=body_parts,
                        label_parts=label_parts,
                        threshold=threshold,
                    )
                )

                if len(seq) > window_size:
                    seq.pop(0)

                if len(seq) == window_size:
                    seq_ = np.array(seq)
                    seq_ = seq_.reshape(1, window_size, -1)
                    # print("seq_ shape > ", seq_.shape)
                    pred = classification_model.predict(seq_)
                    # print("pred > ", pred)
                    label = encoder.inverse_transform(pred)
                    # print("label ? ", label)
            if not label[0][0] == "None":
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
            img_ = img.copy()
            end_time = cv2.getTickCount()
            out_writer.write(img)
            time_seq = (end_time - start_time) / cv2.getTickFrequency()
            if count > 4:
                time_seq_avg += time_seq

            # img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("img", img_)
            # print(f"프레임 수행 시간: {time_seq}")
            # print(1 / (fps))

            # if time_seq < 1 / (fps):
            # sleep(1 / (fps) - time_seq)
            # else:
            # sleep(1 / (fps))

            pbar.update(1)

        # 사용한 자원을 해제합니다.
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)

    # cv2.imwrite(f"out/{video_name}_predic.jpg", img_)
    print(f"평균 프레임 수행 시간: {time_seq_avg / (count)}")
    print(f"초당 처리 프레임: {1 / (time_seq_avg / (count))}")
    # for frame in out_video:

    cap.release()
    out_writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
