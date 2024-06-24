# 동영상을 입력으로하며, 학습된 모델을 이용하여 프레임 단위로 분석하고 예측한 관절 좌표를 생성 및 저장한다.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

import cv2
import numpy as np
import yaml
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import imread
from skimage.util import img_as_ubyte

import angle_out as angle


"""
 ----- param -----
 src_file_path : 분석할 동영상 파일 
 config_path : 학습된 모델의 yaml 파일 = dlc_cfg
 cfg_path : DLC 프로젝트의 yaml 파일 = cfg
 ----- pose_cfg.yaml에 추가 -----
 
init_weights: "/drive/samba/private_files/jupyter/DLC/_mina/project/DLC/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/dlc-models/iteration-0/preLabeled_type(Labrador_Retriever)Sep19-trainset95shuffle1/train/snapshot-50000"
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


# 18개. 정면 오른쪽 앞다리, 정면 왼쪽 앞다리, 후면 오른쪽 뒷다리, 후면 왼쪽 뒷다리, 오른쪽 앞다리 기울기, 왼쪽 앞다리 기울기, 오른쪽 뒷다리 기울기, 왼쪽 뒷다리 기울기, 오른쪽 앞다리, 왼쪽 앞다리, 오른쪽 뒷다리, 왼쪽 뒷다리, 몸통 앞쪽, 몸통 뒷쪽, 입 벌림, 머리, 꼬리, 방향
frontRightView = ["RightWrist", "RightArm", "Nose"]
frontLeftView = ["LeftWrist", "LeftArm", "Nose"]
backRightView = ["TailTip", "RightFemur", "RightAnkle"]
backLeftView = ["TailTip", "LeftFemur", "LeftAnkle"]
frontRightTilt = ["Neck", "RightWrist", "TailStart"]
frontLeftTilt = ["Neck", "LeftWrist", "TailStart"]
backRightTilt = ["Neck", "TailStart", "RightAnkle"]
backLeftTilt = ["Neck", "TailStart", "LeftAnkle"]
frontRight = ["Neck", "RightArm", "RightWrist"]
frontLeft = ["Neck", "LeftArm", "LeftWrist"]
backRight = ["TailStart", "RightFemur", "RightAnkle"]
backLeft = ["TailStart", "LeftFemur", "LeftAnkle"]
frontBody = ["Nose", "Neck", "TailStart"]
backBody = ["Neck", "TailStart", "TailTip"]
mouth = ["Nose", "MouthCorner", "LowerLip"]
head = ["Nose", "Forehead", "Neck"]
tail = ["TailTip", "TailStart", "LeftAnkle"] or ["TailTip", "TailStart", "RightAnkle"]
direction = ["Nose", "RightArm", "RightFemur"] or ["Nose", "LeftArm", "LeftFemur"]
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

# path = os.getcwd()
# project_path = (
# path + "/model/animal_-_111251 (720p)-1-2023-10-24"
# )
# weight_count = 50000
# weight_path = (
# project_path
# + "/dlc-models/iteration-0/animal_-_111251 (720p)Oct24-trainset95shuffle1/train/snapshot"
# + str(weight_count)
# )

# pose_yaml_path = (
# project_path
# + "/dlc-models/iteration-0/animal_-_111251 (720p)Oct24-trainset95shuffle1/train/pose_cfg.yaml"
# )
# base_conf_path = project_path + "/config.yaml"

# with open(base_conf_path) as f:
#     conf_yaml = yaml.load(f, Loader=yaml.FullLoader)

#     # print(conf_yaml["bodyparts"])
#     bodyparts = []
#     for part in conf_yaml["bodyparts"]:
#         bodyparts.append(part)

weight_path = "/home/pieroot/jupyter/DLC/model/animal_-_111251 (720p)-1-2023-10-24/dlc-models/iteration-0/animal_-_111251 (720p)Oct24-trainset95shuffle1/train/snapshot-100000"


def ini_DLC(config_path, cfg_path, weight_path):
    """
    Args:
        config_path = DeepLabCut 의 학습 pose_cfg.yaml 파일 경로
        cfg_path = DeepLabCut 의 기본 설정 파일 config.yaml 경로
    """
    print("초기화 중...")

    print("config_path : ", config_path)
    print("cfg_path : ", cfg_path)

    film = {}

    with open(config_path) as f:
        film = yaml.load(f, Loader=yaml.FullLoader)
        # display(film)
        # film['init_weights'] = film['init_weights_old']
        # film["init_weights_old"] = film["init_weights"]
        film["init_weights"] = weight_path
        film["mean_pixel"] = [123.68, 116.779, 103.939]
        film["weight_decay"] = 0.0001
        film["pairwise_predict"] = False
        film["partaffinityfield_predict"] = False
        film["stride"] = 8.0
        film["intermediate_supervision"] = False
        film["dataset_type"] = "imgaug"

    with open(config_path, "w") as f:
        yaml.dump(film, f)

    global dlc_cfg, cfg, batchsize

    with open(config_path) as f:
        dlc_cfg = yaml.full_load(f)  # 학습된 모델의 yaml 데이터

    with open(cfg_path) as f:
        cfg = yaml.full_load(f)  # 프로젝트의 base yaml 데이터

    try:
        batchsize = dlc_cfg["batch_size"]
    except:
        # update batchsize (based on parameters in config.yaml)
        dlc_cfg["batch_size"] = cfg["batch_size"]
        batchsize = dlc_cfg["batch_size"]

    # batchsize = 8
    # print("batchsize : ", batchsize)
    # 기본 dlc 설정 호출

    global sess, inputs, outputs
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

    return dlc_cfg, cfg, batchsize


# dlc_cfg, cfg, batchsize = ini_DLC()


batch_ind = 0  # keeps track of which image within a batch should be written to
batch_num = 0  # keeps track of which batch you are at


def get_img_coord(src_img):
    global sess, inputs, outputs, dlc_cfg
    global batch_ind, batch_num, batchsize
    # print("img_ : ",src_img.shape) # (360, 640, 3)

    im = src_img
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ny, nx, nc = np.shape(im)
    # batch_size에 따라 반복...?

    # cropping 설정 or batch_size=1일 때를 제외한 나머지 경우

    frames = np.empty(
        (batchsize, ny, nx, 3), dtype="ubyte"  # this keeps all the frames of a batch
    )
    # frames[batch_ind] = img_as_ubyte(im)
    if cfg["cropping"]:
        frames[batch_ind] = img_as_ubyte(
            im[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
        )
    else:
        frames[batch_ind] = img_as_ubyte(im)

    # if batch_ind == batchsize - 1:
    #     pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
    #     # PredictedData[batch_num * batchsize : (batch_num + 1) * batchsize, :]
    #     # = pose
    #     batch_ind = 0
    #     batch_num += 1
    # else:
    #     batch_ind += 1

    # if batch_ind > 0:
    # take care of the last frames (the batch that might have been
    # processed)
    pose = predict.getposeNP(
        frames, dlc_cfg, sess, inputs, outputs
    )  # process the whole batch (some frames might be from previous batch!)
    # print("getPose : ",type(pose),pose) # <class 'numpy.ndarray'> [[
    # 1.45855195e+02 6.34100614e+01 7.33174324e-01 2.44310002e+02 ...]..]

    # print("pose,shape : ", pose.shape)  # (1, 63)
    # print("pose : ", pose)  # (1, 63)
    count = len(pose[0])  # 63 (point_21 * values_3)
    index = 0
    ans = []
    # coords 값 추출
    while index < count:
        point = [pose[0][index + 0], pose[0][index + 1], pose[0][index + 2]]
        ans.append(point)
        index += 3

    ans = np.array(ans)
    return ans


def analyze_frames(src_avi="./videos/coco_train02.mov", window_size=10):
    x_train = []
    data = []
    # Open the AVI file
    cap = cv2.VideoCapture(src_avi)

    # Check if the file was opened successfully
    if not cap.isOpened():
        print("Error opening the file")

    # length = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 분석할 프레임 개수
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    # print(f"전체 길이 : {length}")
    # print(f"fps : {fps}")

    flag = round(fps * 0.2)

    # print(f"flag : {flag}")

    count = 0

    # Read frames from the file
    while True:
        # start = time.time()
        # start = cv2.getTickCount()
        # Read a frame
        ret, frame = cap.read()

        if not ret:
            break
        elif count % flag == 0:
            # Break if we reached the end of the file
            frame = cv2.resize(frame, dsize=(720, 480), interpolation=cv2.INTER_AREA)

            pos = get_img_coord(frame)

            # print(f"추가 한 후 {len(x_train)}")

            x_train.append(
                angle.out(inputs=pos, body_parts=bodyparts, label_parts=label_parts)
            )
            if len(x_train) > window_size:
                x_train.pop(0)
            data.append(x_train)
        count += 1

        # end = cv2.getTickCount()
        # print(f"프레임당 걸린 시간 : {(end - start) / cv2.getTickFrequency()}")

    # Release the video capture object
    cap.release()

    data = np.array(data)

    return data


def analyze_frame(src_avi="./videos/coco_train02.mov", likehood=0.05):
    data = []
    # Open the AVI file
    cap = cv2.VideoCapture(src_avi)

    # Check if the file was opened successfully
    if not cap.isOpened():
        print("Error opening the file")

    # 진행률 process_bar
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 분석할 프레임 개수
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"fps : {fps}")

    # Read frames from the file
    while True:
        # Read a frame
        ret, frame = cap.read()

        if not ret:
            break

        # Break if we reached the end of the file
        frame = cv2.resize(frame, dsize=(720, 480), interpolation=cv2.INTER_AREA)
        pos = get_img_coord(frame)

        data.append(pos)

    # Release the video capture object
    cap.release()

    data = np.array(data)

    return data


def analyze_img(src_img="./videos/test.jpg", likehood=0.1):
    img = cv2.imread(src_img, cv2.IMREAD_ANYCOLOR)

    pos = get_img_coord(img)
    #'numpy.ndarray'> (10, 14)

    for i, p in enumerate(pos):
        if p[2] <= likehood:
            p[0] = 0
            p[1] = 0
            p[2] = 0
            pos[i] = p

    data = pos

    # draw points to frame
    # for i in range(len(pos)):
    # cv2.circle(img, (int(pos[i][0]), int(pos[i][1])), 3, (0, 0, 255), -1)

    # Close all the windows

    data = np.array(data)

    return data