import math
import sys

import numpy as np


def Angle(arr) -> float:
    """
    # 앵글을 계산하는 함수
    ## 3개의 점을 받아서 앵글을 계산한다.
    """
    a = np.arctan2(arr[0]["x"] - arr[1]["x"], arr[0]["y"] - arr[1]["y"]) - np.arctan2(
        arr[1]["x"] - arr[2]["x"], arr[1]["y"] - arr[2]["y"]
    )

    PI = np.pi

    deg = float((a * 180) / PI)
    return deg


def out(
    inputs: np.ndarray,
    body_parts=None,
    label_parts=None,
) -> np.ndarray:
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
        (np.nbarray): _description_
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
        # print(data[index])
        loc_x = data[index]["x"]
        loc_y = data[index]["y"]
        # loc_likelihood = data[index]['likelihood']

        loc_data[body_parts[index]] = {
            "x": loc_x,
            "y": loc_y,
            # "likelihood": loc_likelihood,
        }

        # count += 1

    pose_angle = []
    for label in label_parts:
        loc_3 = [
            loc_data[label[0]],
            loc_data[label[1]],
            loc_data[label[2]],
        ]
        # print(loc_3)

        deg = Angle(loc_3)
        # print(f"{label[1]} - {deg}")
        pose_angle.append(deg)

    data = np.array(pose_angle)

    # data = data.reshape(-1, 1)
    # data = data.reshape(-1)

    return data
