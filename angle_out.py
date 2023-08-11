import sys

import numpy as np

# 전체 키포인트(관절좌표)
# bodyparts = [
#     "Left_Front_Paw",
#     "Left_Front_Wrist",
#     "Left_Front_Elbow",
#     "Left_Back_Paw",
#     "Left_Back_Wrist",
#     "Left_Back_Elbow",
#     "Right_Front_Paw",
#     "Right_Front_Wrist",
#     "Right_Front_Elbow",
#     "Right_Back_Paw",
#     "Right_Back_Wrist",
#     "Right_Back_Elbow",
#     "Tail_Set",
#     "Tail_Tip",
#     "Left_Base_Ear",
#     "Right_Base_Ear",
#     "Nose",
#     "Chin",
#     "Left_Tip_Ear",
#     "Right_Tip_Ear",
#     "Withers",
# ]

# # 각 연결된 키포인트에 대한 데이터
# link_parts = [
#     ["Left_Front_Paw", "Left_Front_Wrist"],
#     ["Left_Front_Wrist", "Left_Front_Elbow"],
#     ["Left_Back_Paw", "Left_Back_Wrist"],
#     ["Left_Back_Wrist", "Left_Back_Elbow"],
#     ["Right_Front_Paw", "Right_Front_Wrist"],
#     ["Right_Front_Wrist", "Right_Front_Elbow"],
#     ["Right_Back_Paw", "Right_Back_Wrist"],
#     ["Right_Back_Wrist", "Right_Back_Elbow"],
#     ["Tail_Set", "Tail_Tip"],
#     ["Withers", "Tail_Set"],
#     ["Nose", "Withers"],
#     ["Chin", "Nose"],
#     ["Nose", "Left_Base_Ear"],
#     ["Left_Base_Ear", "Left_Tip_Ear"],
#     ["Nose", "Right_Base_Ear"],
#     ["Right_Base_Ear", "Right_Tip_Ear"],
#     ["Withers", "Left_Front_Elbow"],
#     ["Withers", "Right_Front_Elbow"],
#     ["Tail_Set", "Left_Back_Elbow"],
#     ["Tail_Set", "Right_Back_Elbow"],
# ]

# # 실제 학습할 때 사용할 관절좌표의 관계 예
# label_parts = [
#     ["Nose", "Withers", "Tail_Set", "Tail_Tip"],  # 코 - 목 - 엉덩이 - 꼬리
#     ["Withers", "Left_Front_Elbow", "Left_Front_Wrist", "Left_Front_Paw"],  # 목 - 왼쪽 앞다리
#     [
#         "Withers",
#         "Right_Front_Elbow",
#         "Right_Front_Wrist",
#         "Right_Front_Paw",
#     ],  # 목 - 오른쪽 앞다리
#     ["Tail_Set", "Left_Back_Elbow", "Left_Back_Wrist", "Left_Back_Paw"],  # 엉덩이 - 왼쪽 뒷다리
#     [
#         "Tail_Set",
#         "Right_Back_Elbow",
#         "Right_Back_Wrist",
#         "Right_Back_Paw",
#     ],  # 엉덩이 - 오른쪽 뒷다리
# ]

import math


def Angle(arr):
    # a = math.atan2(arr[0]["x"] - arr[1]["x"], arr[0]["y"] - arr[1]["y"]) - math.atan2(
    #     arr[1]["x"] - arr[2]["x"], arr[1]["y"] - arr[2]["y"]
    # )

    # PI = math.pi

    # deg = (a * 180) / PI


    a = np.arctan2(arr[0]["x"] - arr[1]["x"], arr[0]["y"] - arr[1]["y"]) - np.arctan2(
        arr[1]["x"] - arr[2]["x"], arr[1]["y"] - arr[2]["y"]
    )

    PI = np.pi

    deg = (a * 180) / PI
    return deg


def out(
    inputs: np.ndarray,
    body_parts=None,
    label_parts=None,
):
    """
    # 앵글 데이터를 추출하는 함수
    ex> inputs = np.array([{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3}])
        body_parts = ["a", "b", "c", "d"]
        label_parts = [["a","b","c"],["b","c","d"]]

        return = np.array([[180],[180]])


    ## Args:
        inputs (np.ndarray): _description_
        body_parts (list): _description_. Defaults to None.
        label_parts (list): _description_. Defaults to None.

    ## Returns:
        _type_: _description_
    """
    try:
        if body_parts is None:
            assert body_parts is not None, "body_parts is None"
        if label_parts is None:
            assert label_parts is not None, "label_parts is None"
    except AssertionError as e:
        print(e)
        sys.exit(1)

    loc_data = {}
    data = inputs
    for index in range(len(inputs)):
        loc_x = data[index][0]
        loc_y = data[index][1]
        loc_likelihood = data[index][2]

        loc_data[body_parts[index]] = {
            "x": loc_x,
            "y": loc_y,
            "likelihood": loc_likelihood,
        }

        # count += 1

    pose_angle = []

    for label in label_parts:
        # if (
        #     loc_data[label[0]]["likelihood"] < 0.01
        #     or loc_data[label[1]]["likelihood"] < 0.01
        #     or loc_data[label[2]]["likelihood"] < 0.01
        # ):
        #     loc_3 = [0, 0, 0]
        #     pass
        # else:
        loc_3 = [
            loc_data[label[0]],
            loc_data[label[1]],
            loc_data[label[2]],
        ]
        # print(loc_4)

        deg = Angle(loc_3)
        # print(f"{label[1]} - {deg}")
        pose_angle.append(deg)

    data = np.array(pose_angle)

    # data = data.reshape(-1, 1)
    # data = data.reshape(-1)

    return data
