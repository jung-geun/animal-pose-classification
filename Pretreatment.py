#!/usr/bin/env python
# coding: utf-8

# In[4]:


# 동영상을 입력으로하며, 학습된 모델을 이용하여 프레임 단위로 분석하고 예측한 관절 좌표를 생성 및 저장한다.
import os

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

import deeplabcut
import sys
import natsort 
import cv2 
import yaml
import numpy as np
import tqdm.notebook as tqdm
from skimage.util import img_as_ubyte
import time
import json

from data_sensor import ini_DLC, get_img_coord, analyze_frames, analyze_frame

import angle_out as angle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
 ----- param -----
 src_file_path : 분석할 동영상 파일 
 config_path : 학습된 모델의 yaml 파일 = dlc_cfg
 cfg_path : DLC 프로젝트의 yaml 파일 = cfg
 ----- pose_cfg.yaml에 추가 -----
 
init_weights: "/drive/samba/private_files/jupyter/DLC/dog1/dlc-models/iteration-0/dog1Mar24-trainset95shuffle1/train/snapshot-30000"
mean_pixel: [123.68, 116.779, 103.939]
weight_decay: 0.0001
pairwise_predict: False
partaffinityfield_predict: False
stride: 8.0
intermediate_supervision: False
dataset_type: imgaug
"""


""" edit. 폴더명 파라미터로 받아서 실행하기
"""


# In[ ]:


def analyze_video(src=None, out=None):
    data = []
    
    if os.path.isdir(src):
        # print(f"소스 디렉토리 > {src}")
        for src_list in natsort.natsorted(os.listdir(src)):
            if src_list == ".ipynb_checkpoints":
                continue
            src_path = f"{src}/{src_list}"
            # print(f"소스 경로 > {src_path}")

            src_arr = src_path.split('/')

            src_tmp = f"./data"
            des_tmp = f"{out}/"

            label = f"{src_arr[-2]}"

            for i in src_arr[2:-1]:
                # print(f"소스 경로 리스트 > {i}")
                des_tmp += f"{i}/"
                # print(f"생성할 파일 리스트 > {des_tmp}")

                if not os.path.isdir(des_tmp):
                    os.mkdir(des_tmp)

            vedio_name, video_ext = os.path.splitext(os.path.basename(src_path))
            label = des_tmp + label

            data.append(analyze_frame(src_path))
            # print(f"file path > {label}")
            # print(f"des_tmp > {des_tmp}")
            # print("LB : ",type(data),data.shape) # <class 'numpy.ndarray'> (271, 21, 3)
    
    else:
        print(f"소스 경로 > {src}")

        src_arr = src.split('/')

        src_tmp = f"./data"
        des_tmp = f"{out}/"

        label = f"{src_arr[-2]}"

        for i in src_arr[2:-1]:
            # print(f"소스 경로 리스트 > {i}")
            des_tmp += f"{i}/"
            # print(f"생성할 파일 리스트 > {des_tmp}")

            if not os.path.isdir(des_tmp):
                os.mkdir(des_tmp)

        vedio_name, video_ext = os.path.splitext(os.path.basename(src))
        label = des_tmp + label

        data = analyze_frame(src)
        # print(f"file path > {label}")
        # print(f"des_tmp > {des_tmp}")
        # print("LB : ",type(data),data.shape) # <class 'numpy.ndarray'> (271, 21, 3)
    
    np.save(label, data)

    return data



inputs_dir = sys.argv[1]
outputs_dir = sys.argv[2]
    
# inputs_dir = "./data"

if not os.path.isdir(inputs_dir):
    analyze_video(src=inputs_dir, out=outputs_dir)
    
else:
    for label_list in natsort.natsorted(os.listdir(inputs_dir)):
        if label_list == ".ipynb_checkpoints":
            continue
        # print(inputs_dir+"/"+label_list)
        for label_data in natsort.natsorted(os.listdir(inputs_dir+"/"+label_list)):
            if label_data == ".ipynb_checkpoints":
                continue
            src_img=f"{inputs_dir}/{label_list}/{label_data}"
            print(src_img)
            analyze_video(src=src_img, out=outputs_dir)
