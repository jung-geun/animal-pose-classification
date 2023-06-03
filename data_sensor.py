# 동영상을 입력으로하며, 학습된 모델을 이용하여 프레임 단위로 분석하고 예측한 관절 좌표를 생성 및 저장한다.

import deeplabcut
import os
import cv2 
import yaml
import numpy as np
import tqdm.notebook as tqdm
from skimage.util import img_as_ubyte
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.pose_estimation_tensorflow.core import predict
import time
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

path = os.getcwd()
project_path = path+"/dog1/"
weight_count = 30000
weight_path = project_path+"dlc-models/iteration-0/dog1Mar24-trainset95shuffle1/train/snapshot-"+str(weight_count)

pose_yaml_path = project_path+"dlc-models/iteration-0/dog1Mar24-trainset95shuffle1/test/pose_cfg.yaml"
base_conf_path = project_path+"config.yaml"

film = {}

with open(pose_yaml_path) as f:
    film = yaml.load(f, Loader=yaml.FullLoader)
#     # display(film)
#     film["init_weights_old"] = film["init_weights"]
    film["init_weights"] = weight_path
    film["mean_pixel"] = [123.68, 116.779, 103.939]
    film["weight_decay"] = 0.0001
    film["pairwise_predict"] = False
    film["partaffinityfield_predict"] = False
    film["stride"] = 8.0
    film["intermediate_supervision"] = False
    film["dataset_type"] = "imgaug"
    
# display(film)
with open(pose_yaml_path, 'w') as f:
    yaml.dump(film, f)
print("호출 완료")

def ini_DLC(config_path=pose_yaml_path,
            cfg_path=base_conf_path):
    with open(config_path) as f:
        dlc_cfg = yaml.full_load(f)      # 학습된 모델의 yaml 데이터
    with open(cfg_path) as f:
        cfg = yaml.full_load(f)      # 프로젝트의 base yaml 데이터
    try :
        batchsize = dlc_cfg["batch_size"]
    except :
        # update batchsize (based on parameters in config.yaml)
        dlc_cfg["batch_size"] = cfg["batch_size"]
        batchsize = dlc_cfg["batch_size"]

    return  dlc_cfg,cfg,batchsize


def get_img_coord(src_img, dlc_cfg, cfg, batchsize):
    #print("img_ : ",src_img.shape) # (360, 640, 3)
    im = src_img
    
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
    
    ny, nx, nc = np.shape(im)
    # batch_size에 따라 반복...?
    batch_ind = 0    # keeps track of which image within a batch should be written to
    batch_num = 0    # keeps track of which batch you are at

    # cropping 설정 or batch_size=1일 때를 제외한 나머지 경우
    frames = np.empty((batchsize, ny, nx, 3), dtype="ubyte"      # this keeps all the frames of a batch
    )  
    frames[batch_ind] = img_as_ubyte(im)


    if batch_ind == batchsize - 1:
        pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
        # PredictedData[batch_num * batchsize : (batch_num + 1) * batchsize, :]
        # = pose
        batch_ind = 0
        batch_num += 1
    else:
        batch_ind += 1

    if (batch_ind > 0):
        # take care of the last frames (the batch that might have been
        # processed)
        pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)     # process the whole batch (some frames might be from previous batch!)
        
        # print("getPose : ",type(pose),pose) # <class 'numpy.ndarray'> [[
        # 1.45855195e+02 6.34100614e+01 7.33174324e-01 2.44310002e+02 ...]..]

        count = len(pose[0])  # 63 (point_21 * values_3)
        index = 0
        ans = []
        # coords 값 추출
        while index < count:
            point = [pose[0][index + 0],pose[0][index + 1],pose[0][index + 2]]
            ans.append(point)
            index += 3
        
        ans = np.array(ans)

        return ans 

def analyze_frames(src_avi="./videos/coco_train02.mov"):

    x_train = []

    dlc_cfg,cfg,batchsize = ini_DLC()
    data = []
    # Open the AVI file
    cap = cv2.VideoCapture(src_avi)

    # Check if the file was opened successfully
    if not cap.isOpened():
        print("Error opening the file")

    # 진행률 process_bar
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)      # 분석할 프레임 개수
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"전체 길이 : {length}")
    print(f"초당 프레임 : {fps}")

    flag = int(int(fps) / (100/50))

    print(f"flag : {flag}")
    # pbar = tqdm(total=int(length))

    count = 0 

    # Read frames from the file
    while True:
        start = time.time()

        # Read a frame
        ret, frame = cap.read()
        mid = time.time()

        if count > length:
            end = time.time()
            break
        elif count % flag != 0:
            count += 1
            end = time.time()
            continue

        # print(f"{count}번째 프레임")

        # Break if we reached the end of the file
        elif not ret:
            end = time.time()
            break
        # Display the frame
        # cv2.imshow("Frame", frame)
        pose_start = time.time()
        pos = get_img_coord(frame,dlc_cfg,cfg,batchsize)
        pose_end = time.time()
        #print("mk_from_avi.pos : ",type(pos),pos.shape) # <class
        #'numpy.ndarray'> (21, 3)
        data.append(pos)

        # if count != 0:
            # pbar.update()

        # draw points to frame
        for i in range(len(pos)):
            cv2.circle(frame, (int(pos[i][0]), int(pos[i][1])), 3, (0, 0, 255), -1)
            # cv2.putText(frame, str(i), (int(pos[i][0]), int(pos[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # print(type(pos))
        # Display the frame
        # cv2.imshow("Frame", frame)

        # cv2.imwrite(f'{out_path}/{count}.png',frame)

        #print("count : ",count) # 271 (sec_9 * fps_30)

        print(f"추가 한 후 {len(x_train)}")
        if len(x_train) >= 4:
            print(np.shape(x_train))
            x_train.pop(0)

        x_train.append(angle.out(inputs = pos))
        print(f"조건 식 후 : {len(x_train)}")

        count += 1

        end = time.time()
        print(f"한번 실행하는데 걸리는 시간 : {end - start:.5f} sec")
        print(f"이미지 호출하는데 걸리는 시간 : {mid - start:.5f} sec")
        print(f"프리딕션에 걸리는 시간 : {pose_end - pose_start:.5f} sec")


    # pbar.close()
    # Release the video capture object
    cap.release()

    # Close all the windows
    # cv2.destroyAllWindows()

    data = np.array(data)

    return data

def analyze_frame(src_img="./videos/coco_train02.mov"):

    x_train = []

    dlc_cfg,cfg,batchsize = ini_DLC()

    data = []
    # Open the AVI file
    print(f"소스 출력 > {src_img}")
    cap = cv2.VideoCapture(src_img)

    # Check if the file was opened successfully
    if not cap.isOpened():
        print("Error opening the file")

    # 진행률 process_bar

    count = 0 

    # Read frames from the file
    while True:

        # Read a frame
        ret, frame = cap.read()

        # Break if we reached the end of the file
        if not ret:
            break
        pos = get_img_coord(frame,dlc_cfg,cfg,batchsize)
        #print("mk_from_avi.pos : ",type(pos),pos.shape) # <class
        #'numpy.ndarray'> (21, 3)
        data = pos

        # draw points to frame
        for i in range(len(pos)):
            cv2.circle(frame, (int(pos[i][0]), int(pos[i][1])), 3, (0, 0, 255), -1)

    # Release the video capture object
    cap.release()

    # Close all the windows

    data = np.array(data)

    return data

def analyze_video(src=None, out=None):
    data = []
    
    if os.path.isdir(src):
        # print(f"소스 디렉토리 > {src}")
        for src_list in natsort.natsorted(os.listdir(src)):
            src_path = f"{src}/{src_list}"
            print(f"소스 경로 > {src_path}")

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

            data = analyze_frame(src_path)
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

# get_img_coord(src_img_,dlc_cfg_,cfg_,batchsize_)