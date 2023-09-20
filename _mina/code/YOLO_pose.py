import os
import cv2
import json
import random
import tensorflow as tf
from ultralytics import YOLO
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        

"""
  prepare : YOLO 학습을 위해 데이터를 준비하는 함수
"""
def prepare():
    data_path = "/home/dlc/DLC/_mina/data/AI-Hub/poseEstimation/Validation/DOG"
    action = ["BODYSHAKE", "BODYLOWER", "BODYSCRATCH", 
              "WALKRUN", "SIT", "HEADING", "LYING", 
              "MOUNTING", "TAILING", "TAILLOW", "TURN", 
              "FOOTUP", "FEETUP"]
    src_path = []
    lab_path = []
    dataset = []
    total = 0, 0
    max_data_per_action = 160         # 한 동작 당 160개의 영상

    for i in range(len(action)):
        src_path.append(
            data_path + "/source" + action[i]
        )  # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/sourceSIT"
        lab_path.append(
            data_path + "/label" + action[i]
        )  # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/labelSIT"
        
        image_folders = os.listdir(src_path[i] + "/images/")
        for folder in image_folders:
            frames_folder = src_path[i] + "/images/" + folder
        dataset.append(frames_folder)
        
        print("Action_", action[i], " 총 데이터 수 : ", len(dataset))
        print("Action_", action[i], " 학습할 데이터 수 : ", len(dataset[i]))
        total += len(dataset)
    print("실제 영상 total : ", total)
    print("학습 영상 total : ", len(dataset)*len(dataset[0]))
    return src_path, lab_path, dataset, action
  
  
"""
makeData : Yolo 학습 시키기 위해 이미지와 라벨 데이터 텍스트 파일 생성
"""
def makeData():
  cnt = 0
  src_path, lab_path, dataset, action = prepare()
  save_img_path_train = '/home/dlc/DLC/_mina/data/YOLO/YOLO_pose/images/train/'  # 이미지를 저장할 경로
  save_txt_path_train = '/home/dlc/DLC/_mina/data/YOLO/YOLO_pose/labels/train/'  # 라벨 데이터를 저장할 경로
  save_img_path_val = '/home/dlc/DLC/_mina/data/YOLO/YOLO_pose/images/val/'  
  save_txt_path_val = '/home/dlc/DLC/_mina/data/YOLO/YOLO_pose/labels/val/'  
  os.makedirs(save_img_path_train, exist_ok=True)
  os.makedirs(save_txt_path_train, exist_ok=True)
  os.makedirs(save_img_path_val, exist_ok=True)
  os.makedirs(save_txt_path_val, exist_ok=True)
  
  # action 수만큼 반복
  for i in range(len(src_path)):
    # tarin, val data를 일정 비율로 randomly split
    # dataset[i] 개수가 500개 이상이면 8:2로 split, 아니면 9:1로 split
    if len(dataset[i]) >= 500:
      # k = dataset[i] 길이의 80%에 해당하는 정수 => dataset[i] 개수 중에 k개를 무작위 선택하여 저장
      train_indices = random.sample(range(len(dataset[i])), k=int(len(dataset[i])*0.8))
    else:
      train_indices = random.sample(range(len(dataset[i])), k=int(len(dataset[i])*0.9))
    
    # max_data_per_action 만큼 반복
    for j in range(len(dataset[i])):
      # print("src : ", src_path, ", dataset : ", dataset[i][j])
      frames_folder = (
        src_path[i] + "/images/" + dataset[i][j]
      )  # dataset의 frame들의 폴더 위치
      images = [
        img for img in os.listdir(frames_folder) if img.endswith(".jpg")
      ]
      
      # 파일 이름을 기준으로 정렬하여 목록 생성
      try:
        images.sort(key=lambda x: int(x.split("_")[1]))
      except ValueError as e:
        print(f"\n Please CHECK !! \nError sorting images in {frames_folder}: {e}")
          
      json_path = (
        lab_path[i] + "/json/" + dataset[i][j] + ".json"
      )  # coords가 있는 json 경로

      with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

      # 선택한 키의 값만 추출
      selected_data = []
      # 모든 annotation에 대해 반복
      for idx, annotation in enumerate(data["annotations"]):
        row = []
        for key, value in annotation["bounding_box"].items():
          if value is not None:
            row.append(value)
          else:
            row.append(None)
        selected_data.append(row)
          
      if j in train_indices:
        save_img_path = save_img_path_train
        save_txt_path = save_txt_path_train
      else:
        save_img_path = save_img_path_val
        save_txt_path = save_txt_path_val
          
      # 32fps일 때, 초당 1.5장의 프레임 추출
      current_frame_number = annotation["frame_number"]   # 현재 프레임 번호
      # 현재 프레임 번호가 추출 간격에 맞는지 확인
      
      # JSON 데이터에서 프레임 번호가 매번 6씩 증가하므로, 이는 실제로는 초당 약 32 / 6 ≈ 5.33장의 프레임. 
      if idx % 4 == 0:  # 매 4번째 프레임을 선택 (2fps)
        frame = cv2.imread(os.path.join(frames_folder, images[current_frame_number]))
        # bounding box의 좌표와 너비, 높이
        x, y, w, h = annotation["bounding_box"].values()   
        # 프레임 크기
        height, width, layers = frame.shape         
        
        # 이미지를 저장 (이름이 동일해야 한다.)
        cv2.imwrite(os.path.join(save_img_path, f'{action[i]}_{j+1}-{current_frame_number+1}.jpg'), frame)
        # 텍스트를 저장 
        # Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
        with open(os.path.join(save_txt_path, f'{action[i]}_{j+1}-{current_frame_number+1}.txt'), 'w') as f:
          # class_id : 0, 바운딩 박스 중심에 대한 좌표, 너비, 높이
          f.write(f"{0} {(x + w/2)/width} {(y + h/2)/height} {w/width} {h/height}\n")     # 정규화시키기 위해 나눠준다
                    
        cnt += 1

                
  print("라벨링 데이터 수 : ",cnt)
  print("train 라벨 총 개수 : ", len(os.listdir(save_img_path_train)))
  print("val 라벨 총 개수 : ", len(os.listdir(save_img_path_val)))