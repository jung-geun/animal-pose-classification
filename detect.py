# from collections import defaultdict

import cv2

import numpy as np

# import torch
from ultralytics import YOLO

# from sort import *

# mot_tracker = Sort()


def main():
    # weights_path = "_mina/project/YOLO_test_train/weights/best.pt"
    weights_path = "runs/detect/train4/weights/best.pt"
    # weights_path = "yolov8n.pt"

    model = YOLO(weights_path)

    video_name = "model/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/videos/dog-walkrun-085352.avi"
    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    print(f"width: {width}, height: {height}, fps: {fps}")

    if width > height:
        width_ = 640
        height_ = 480
    else:
        width_ = 480
        height_ = 640
    # scale_width = width / width_
    # scale_height = height / height_

    print(f"width_: {width_}, height_: {height_}")
    # print(f"scale_width: {scale_width}, scale_height: {scale_height}")

    threshold = 0.5

    # out_video = []
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_name = video_name.split("/")[-1].split(".")[0]

    out = cv2.VideoWriter(
        f"out/{video_name}_predic.mp4", fourcc, fps, (width_, height_)
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        img = frame.copy()
        img = cv2.resize(img, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

        results = model.predict(img, imgsz=640)
        # results = model.track(img, imgsz=640, show=True, iou=threshold)

        detect = results[0]
        box_xywh = detect.boxes.xywh.cpu().numpy()

        # img_ = img.copy()
        # print(f"박스의 좌표 > {box_xywh}")

        x, y, w, h = box_xywh[0]
        img_ = img[int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)]
        img_ = cv2.resize(img_, dsize=(width_, height_), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("img", img_)
        # out_video.append(img_)
        out.write(img_)

    # 사용한 자원을 해제합니다.

    # out_width, out_height = out_video[0].shape[:2]
    # cv2.imwrite(f"out/{video_name}_predic.jpg", img_)

    # for frame in out_video:

    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
