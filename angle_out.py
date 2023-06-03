# import deeplabcut
import tensorflow as tf

import numpy as np
import pandas as pd

# print("dlc",deeplabcut.__version__)
# print("tf",tf.__version__)

# 전체 키포인트(관절좌표)
keypoint = [
    "Left_Front_Paw",
    "Left_Front_Wrist",
    "Left_Front_Elbow",
    
    "Left_Back_Paw",
    "Left_Back_Wrist",
    "Left_Back_Elbow",
         
    "Right_Front_Paw",
    "Right_Front_Wrist",
    "Right_Front_Elbow",
    
    "Right_Back_Paw",
    "Right_Back_Wrist",
    "Right_Back_Elbow",
         
    "Tail_Set",
    "Tail_Tip",
    
    "Left_Base_Ear",
    "Right_Base_Ear",
    
    "Nose",
    "Chin",
    
    "Left_Tip_Ear",
    "Right_Tip_Ear",
    
    "Withers"]

# 각 연결된 키포인트에 대한 데이터
link_parts = [
    ["Left_Front_Paw","Left_Front_Wrist"],
    ["Left_Front_Wrist","Left_Front_Elbow"],
    
    ["Left_Back_Paw","Left_Back_Wrist"],
    ["Left_Back_Wrist","Left_Back_Elbow"],
    
    ["Right_Front_Paw","Right_Front_Wrist"],
    ["Right_Front_Wrist","Right_Front_Elbow"],
    
    ["Right_Back_Paw","Right_Back_Wrist"],
    ["Right_Back_Wrist","Right_Back_Elbow"],
    
    ["Tail_Set","Tail_Tip"],
    ["Withers","Tail_Set"],
    ["Nose","Withers"],
    ["Chin","Nose"],
    
    ["Nose","Left_Base_Ear"],
    ["Left_Base_Ear","Left_Tip_Ear"],
    
    ["Nose","Right_Base_Ear"],
    ["Right_Base_Ear","Right_Tip_Ear"],
    
    ["Withers","Left_Front_Elbow"],
    ["Withers","Right_Front_Elbow"],
    
    ["Tail_Set","Left_Back_Elbow"],
    ["Tail_Set","Right_Back_Elbow"]]

# 실제 학습할 때 사용할 관절좌표의 관계 예
label_parts = [
    ["Nose", "Withers", "Tail_Set", "Tail_Tip"],                              # 코 - 목 - 엉덩이 - 꼬리
    ["Withers", "Left_Front_Elbow", "Left_Front_Wrist", "Left_Front_Paw"],    # 목 - 왼쪽 앞다리
    ["Withers", "Right_Front_Elbow", "Right_Front_Wrist", "Right_Front_Paw"], # 목 - 오른쪽 앞다리
    ["Tail_Set", "Left_Back_Elbow", "Left_Back_Wrist", "Left_Back_Paw"],      # 엉덩이 - 왼쪽 뒷다리
    ["Tail_Set", "Right_Back_Elbow", "Right_Back_Wrist", "Right_Back_Paw"]    # 엉덩이 - 오른쪽 뒷다리
]

import math

def Angle_3rd_point(arr):
    a = math.atan2(arr[0]["x"] - arr[1]["x"], arr[0]["y"] - arr[1]["y"]) - \
        math.atan2(arr[1]["x"] - arr[2]["x"], arr[1]["y"] - arr[2]["y"])

    PI = math.pi

    deg = (a*180)/PI

    return deg

def Angle(arr):
    
    a = Angle_3rd_point([arr[0], arr[1], arr[2]])

    b = Angle_3rd_point([arr[1], arr[2], arr[3]])

    PI = math.pi

    deg = [a, b]
    
    return deg

import json 

def out(inputs : np.ndarray, keypoint = keypoint, link_parts = link_parts, label_parts = label_parts):
    loc_data = {}
    data = inputs
    # print(len(inputs))
    # print(keypoint)
    # count = 0
    for index in range(len(inputs)):
        # print(data)
        # print(data[index])
        loc_x = data[index]['x']
        loc_y = data[index]['y']
        # print(loc_x, loc_y)
        
        loc_data[keypoint[index]] = {"x" : loc_x, "y" : loc_y}

        # count += 1
    
    pose_angle = []

    for label in label_parts:
        # print(link_data)

        loc_4 = [loc_data[label[0]], loc_data[label[1]], loc_data[label[2]], loc_data[label[3]]]
        # print(loc_4)

        deg = Angle(loc_4)
        # print(deg)

        pose_angle.append(deg)
    
    data = np.array(pose_angle)
    
    data = data.reshape(-1,1)
    
    return data