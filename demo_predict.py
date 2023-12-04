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


def analyze(video, name, model_name, show=False):
    """
    video: 비디오 경로
    name: 저장할 비디오 이름
    model_name: 모델 경로
    show: 비디오 출력 여부
    """
    project_path = model_name
    pose_config_path_iter = f"{project_path}/dlc-models/iteration-0"
    pose_config_path_train = (
        f"{pose_config_path_iter}/{os.listdir(pose_config_path_iter)[0]}/train"
    )
    os.listdir(pose_config_path_train)
    config_path = f"{project_path}/config.yaml"
    pose_config_path = f"{pose_config_path_train}/pose_cfg.yaml"
    weights_path = f"{pose_config_path_train}/snapshot-100000"

    ini_DLC(pose_config_path, config_path, weights_path)

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
    tail = ["TailTip", "TailStart", "LeftAnkle"] or [
        "TailTip",
        "TailStart",
        "RightAnkle",
    ]
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

    # weights_path = "/home/dlc/DLC/runs/detect/train4/weights/best.pt"
    # weights_path = "yolov8n.pt"

    # yolo_model = YOLO(weights_path)
    window_size = 15
    classification_model = load_model(
        f"/home/dlc/DLC/model/LSTM_{window_size}_18/model.h5"
    )

    with open("/home/dlc/DLC/model/onehot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    video_name = name
    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total_frame_count: {total_frame_count}")

    time_seq_avg = 0

    if width > height:
        width_ = 640
        height_ = 480
    else:
        width_ = 480
        height_ = 640

    print(f"width: {width}, height: {height}, fps: {fps}")

    threshold = 0.1

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    out_writer = cv2.VideoWriter(
        f"/home/dlc/DLC/out/{video_name}_predic.mp4", fourcc, fps, (width_, height_)
    )
    count = 0
    flag = int(fps // 5)  # 초당 5개의 프레임을 추출
    seq = []
    label = [["None"]]
    label_ = "None"
    pbar = tqdm(iterable=range(total_frame_count), desc="Processing", unit="frame")
    before_label = []
    # dog_index = 16.0
    while True:
        start_time = cv2.getTickCount()
        ret, frame = cap.read()
        count += 1

        if not ret:
            break

        img = frame.copy()
       
        img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)
        # results = yolo_model.predict(img, imgsz=640)

        # detect = results[0]
        # box_xywh = detect.boxes.xywh.cpu().numpy()
        # box_class = detect.boxes.cls.cpu().tolist()
        # print(box_class)

        # # img_ = img.copy()
        # # print(f"박스의 좌표 > {box_xywh}")
        # if len(box_xywh) == 0 or dog_index not in box_class:
        #     continue
        # else:
        #     box_dog = box_class.index(dog_index)
        #     x, y, w, h = box_xywh[box_dog]
        #     img = img[int(y - h / 2)-20 : int(y + h / 2)+20, int(x - w / 2)-20 : int(x + w / 2)+20]
        #     img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

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
            # print(f"count: {count}")
            inputs = []
            for i, p in enumerate(point):
                inputs.append({"x": p[0], "y": p[1], "likelihood": p[2]})

            # 영상 시작 시 같은 프레임으로 분류 후 시작 - None 으로 분류하는걸 방지
            while len(seq) < window_size:
                seq.append(
                    out(
                        inputs,
                        body_parts=body_parts,
                        label_parts=label_parts,
                        threshold=threshold,
                    )
                )
            # 프레임이 15개 이상일 경우 15개의 프레임만 유지 - 연산 완료 후 가장 오래된 프레임 제거
            if len(seq) >= window_size:
                seq_ = np.array(seq)
                seq_ = seq_.reshape(1, window_size, -1)
                pred = classification_model.predict(seq_)
                label = encoder.inverse_transform(pred)

                # 현재 라벨이 이전 3개의 라벨과 다를 경우 이전 3개의 라벨을 비교하여 가장 많은 라벨을 현재 라벨로 설정
                before_label.append(label[0][0])

                if not label[0][0] in before_label and len(before_label) > 0:
                    label_ = max(set(before_label), key=before_label.count)
                else:
                    label_ = label[0][0]

                if len(before_label) > 2:
                    before_label.pop(0)

                seq.pop(0)

        if not label_ == "None":
            # 우측 하단에 라벨 표시
            cv2.putText(
                img,
                label_,
                (int(10), int(height_ - 10)),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 0),
                2,
            )

        end_time = cv2.getTickCount()
        img_ = img.copy()
        out_writer.write(img)

        time_seq = (end_time - start_time) / cv2.getTickFrequency()

        if count > 0:
            time_seq_avg += time_seq
        if show:
            cv2.imshow("img", img_)
            cv2.waitKey(1)

        pbar.update(1)

    print(f"평균 프레임 수행 시간: {time_seq_avg / count}")
    print(f"초당 처리 프레임: {1 / (time_seq_avg / count)}")

    # 사용한 자원을 해제합니다.
    cap.release()
    out_writer.release()

    cv2.destroyAllWindows()

    return f"/home/dlc/DLC/out/{video_name}_predic.mp4"
