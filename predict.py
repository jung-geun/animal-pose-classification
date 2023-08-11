# %%
import cv2
from data_sensor import get_img_coord
import angle_out as angle

import yaml
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pickle
import json

with open("Zoo/config.yaml") as f:
    conf_yaml = yaml.load(f, Loader=yaml.FullLoader)

    body_parts = []
    link_parts = []
    for part in conf_yaml["bodyparts"]:
        body_parts.append(part)
    for link in conf_yaml["skeleton"]:
        link_parts.append(link)


label_parts = [
    ["nose", "neck_base", "neck_end"],
    ["neck_base", "neck_end", "back_base"],
    ["neck_end", "back_base", "back_middle"],
    ["back_base", "back_middle", "back_end"],
    ["back_middle", "back_end", "tail_base"],
    ["back_end", "tail_base", "tail_end"],
    ["back_base", "front_left_thai", "front_left_knee"],
    ["front_left_thai", "front_left_knee", "front_left_paw"],
    ["back_base", "front_right_thai", "front_right_knee"],
    ["front_right_thai", "front_right_knee", "front_right_paw"],
    ["back_end", "back_left_thai", "back_left_knee"],
    ["back_left_thai", "back_left_knee", "back_left_paw"],
    ["back_end", "back_right_thai", "back_right_knee"],
    ["back_right_thai", "back_right_knee", "back_right_paw"],
]

with open("model/xgboost.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("model/label.json", "rb") as f:
    json_ = json.load(f)


# %%
def predict(video_save=False):
    video = "videos2/shortTest2 - 복사본.mp4"
    cap = cv2.VideoCapture(video)

    # 영상 기본 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 영상의 가공 정보
    window_size = 10
    flag = int(int(fps) / (30 / 5))
    count = 1

    if cap.isOpened():
        ret, frame = cap.read()
        print(f"width: {width}, height: {height}, fps: {fps}")
        label = []
        result = ""
        out_video = []
        width_tmp = 720
        height_tmp = 480
        while ret:
            # e1 = cv2.getTickCount()
            count += 1
            # frame = cv2.putText(
            #     frame,
            #     f"FPS: {fps}",
            #     (10, 30),
            #     cv2.FONT_HERSHEY_PLAIN,
            #     1,
            #     (0, 255, 0),
            #     2,
            # )
            # frame = cv2.flip(frame, -1)  # 상하 좌우 대
            if width > height:
                width_tmp = 720
                height_tmp = 480
            else:
                width_tmp = 480
                height_tmp = 720
            frame = cv2.resize(
                frame, dsize=(width_tmp, height_tmp), interpolation=cv2.INTER_AREA
            )
            # frame = cv2.resize(
            # frame, dsize=(width/2, height/2), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            # )
            point = get_img_coord(frame)
            for p in point:
                if p[2] > 0.01:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255))
            for link in link_parts:
                # print(link)
                p1 = body_parts.index(link[0])
                p2 = body_parts.index(link[1])
                cv2.line(frame, (int(point[p1][0]), int(point[p1][1])), (int(point[p2][0]), int(point[p2][1])), (0, 255, 0), 2)
            if count % flag == 0:
                print(count)
                label.append(angle.out(point, body_parts, label_parts))

                if len(label) > window_size:
                    label.pop(0)

                if len(label) == window_size:
                    # label_ = (label)
                    # label_ = label_.reshape(-1, label.shape[0] * label.shape[1])
                    # print(f"label: {np.shape([label])}")
                    label_ = np.array([label]).reshape(-1)
                    pred = model.predict([label_])
                    print(pred)
                    try:
                        la_ = encoder.inverse_transform(pred)
                        result = la_
                        print(f"pred: {la_}")
                    except:
                        print("분류하지 못함")
                        # result = "분류하지 못함"
                # print(f"count: {count}, label: {np.shape(label)}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # e2 = cv2.getTickCount()
            # time = (e2 - e1) / cv2.getTickFrequency()
            # print(f"time: {time}")
            # cv2.imshow(f"video", frame)
            cv2.putText(
                frame,
                f"result: {result}",
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
                2,
            )
            if video_save:
                out_video.append(frame)
            ret, frame = cap.read()
    else:
        print("Error")

    cap.release()
    cv2.destroyAllWindows()

    if video_save:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_name = video.split("/")[-1].split(".")[0]
        out = cv2.VideoWriter(
            f"out/{video_name}_predic.mp4", fourcc, fps, (width_tmp, height_tmp)
        )

        for i in range(len(out_video)):
            out.write(out_video[i])
        out.release()


predict(video_save=True)

# %%
