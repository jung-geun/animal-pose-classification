import json
import cv2

def get_img_coord(ref):
    with open(ref, encoding='utf-8') as f:
        json_object = json.load(f)
    annotations = json_object["annotations"]
    frame = annotations[60]     # 60번째 frame
    # print(type(annotations))     # list
    # print(type(frame))     # dict
    # for keypoints, coord in frame["keypoints"].items():
    #     print(f"{keypoints}: x-{coord['x']}, y-{coord['y']}")
    coords = []
    for i in range(15):
        if type(frame["keypoints"].get(str(i+1), "null")) is dict:
            coords.append(list(frame["keypoints"].get(str(i+1), "null").values()))
        else :
            coords.append([None, None])
    print(coords)
    # key_list = list(json_object.keys())
    return coords

def annotate_image(img_path, json_path):
    img = cv2.imread(img_path)
    points = get_img_coord(json_path)
    for i, (x, y) in enumerate(points):
        if x is not None or y is not None :
            # draw a small circle at the point
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            # add a label with the point number
            cv2.putText(img, str(i+1), (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return img

# "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/sourceSIT/images/dog-sit-012055/frame_25_timestamp_1000.jpg"
image = "/home/dlc/DLC/_mina/data/AI-Hub/poseEstimation/Validation/DOG/sourceSIT/images/20201029_dog-sit-000219.mp4/frame_360_timestamp_12000.jpg"
info = "/home/dlc/DLC/_mina/data/AI-Hub/poseEstimation/Validation/DOG/labelSIT/json/20201029_dog-sit-000219.mp4.json"
annotated_img = annotate_image(image, info)

# Save the image to a file
cv2.imwrite('output.jpg', annotated_img)
