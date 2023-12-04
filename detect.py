
import cv2

import numpy as np

from ultralytics import YOLO



def main():
    weights_path = "runs/detect/train4/weights/best.pt"

    model = YOLO(weights_path)

    video_name = "model/animal_-_111251 (720p)-1-2023-10-24/videos/animal_-_111251 (720p).mp4"
    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    print(f"width: {width}, height: {height}, fps: {fps}")

    # 가로 세로 비율에 따라서 width_, height_를 정합니다.

    if width > height:
        width_ = 640
        height_ = 480
    else:
        width_ = 480
        height_ = 640

    print(f"width_: {width_}, height_: {height_}")

    threshold = 0.5

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

    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
