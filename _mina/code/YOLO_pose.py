import os
import cv2
import json
import random
import tensorflow as tf
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        

def prepare():
  """
    description
    -----------
    Pre-process the data for YOLO training.
    
    variables
    ----------
    data_path : str
      dataset root directory
    action : list
      list of action names
    src_path : list
      list of paths to the source data for each action
    lab_path : list
      list of paths to the label data for each action
    dataset : list
      list of lists of video names for each action
    
    returns
    -------
    src_path : list
      list of paths to the source data for each action
    lab_path : list
      list of paths to the label data for each action
    dataset : list
      list of lists of video names for each action
    action : list
      list of action names    
  """
  print("------------- prepare() Start -------------")
  data_path = "/home/dlc/DLC/_mina/data/AI-Hub/poseEstimation/Validation/DOG"
  action = ["BODYSHAKE", "BODYLOWER", "BODYSCRATCH", 
            "WALKRUN", "SIT", "HEADING", "LYING", 
            "MOUNTING", "TAILING", "TAILLOW", "TURN", 
            "FOOTUP", "FEETUP"]
  src_path = []
  lab_path = []
  dataset = []

  for action_name in action:
    src_path_action = data_path + "/source" + action_name  # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/sourceSIT"
    lab_path_action = data_path + "/label" + action_name  # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/labelSIT"
    src_path.append(src_path_action)
    lab_path.append(lab_path_action)

    # 해당 동작의 영상 이름 목록 (Image set folder for each action)
    action_img_folders = os.listdir(src_path_action + "/images/")
    # list comprehension
    # newlist = [expression for item in iterable if condition == True]
    # 반복이 가능한 객체에 대해 item을 반복하며 해당 item이 조건에 부합한다면 표현식에 맞춰 요소를 추가하면서 새로운 list를 생성한다.
    dataset_vdo_name = [folder for folder in action_img_folders]
    dataset.append(dataset_vdo_name)

    print("Action_", action_name, " 학습 데이터 수 : ", len(dataset_vdo_name))

  print("영상 total : ", sum(len(data) for data in dataset))
  print("------------- prepare() End -------------\n")
  # src_path : 여러 동작의 원천 데이터가 있는 경로
  # lab_path : 여러 동작의 라벨링 데이터가 있는 경로
  # dataset : 여러 동작의 영상 이름 (2차원 list)
  # action : 동작 이름 목록
  return src_path, lab_path, dataset, action
  
  
def makeData():
  """
    description
    -----------
    Make dataset for YOLO training.
    
    variables
    ----------
    cnt : int
      number of label data
    td : int
      number of train data
    vd : int
      number of val data
    src_path : list
      list of paths to the source data for each action
    lab_path : list
      list of paths to the label data for each action
    dataset : list
      list of lists of video names for each action
    action : list
      list of action names
    save_img_path_train : str
      path to save the images for training
    save_txt_path_train : str
      path to save the label data for training
    save_img_path_val : str
      path to save the images for validation
    save_txt_path_val : str
      path to save the label data for validation
    
    returns
    -------
    None
  """
  print("------------- makeData() Start -------------")
  # cnt : 라벨링 데이터 수
  # td : train data 수
  # vd : val data 수  
  # src_path : 동작 n개에 대한 원천 데이터가 있는 경로 목록
  # lab_path : 동작 n개에 대한 라벨링 데이터가 있는 경로 목록
  # dataset : 동작 n개에 대한 영상 이름 목록 (2차원 list)
  # action : 동작 이름 목록
  cnt, td, vd = 0, 0, 0
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
  for i in range(len(action)):
    # dataset[i] (하나의 동작에 대한 영상 이름 목록)
    action_data = dataset[i]
    
    # train, val data를 일정 비율로 randomly split
    # dataset[i] 개수가 500개 이상이면 8:2로 split, 아니면 9:1로 split
    # dataset[i] 개수가 300개 이상이면 8:2로 split, 아니면 9:1로 split
    if len(action_data) >= 300:
      train_data, val_data = train_test_split(action_data, test_size=0.2, random_state=42)
    else:
      train_data, val_data = train_test_split(action_data, test_size=0.1, random_state=42)

    for name in action_data:
      # name : 영상 이름
      json_path = lab_path[i] + "/json/" + name + ".json"   # 해당 영상의 정보가 있는 json 경로   
      frames_folder = src_path[i] + "/images/" + name   # 해당 영상의 프레임들이 있는 폴더 경로
      # frames_folder에 있는 이미지 파일들을 파일 이름을 기준으로 정렬하여 목록 생성
      images = sorted([img for img in os.listdir(frames_folder) if img.endswith(".jpg")], key=lambda x: int(x.split("_")[1]))
      
      # train, val 경로 설정
      save_img_path = save_img_path_train if name in train_data else save_img_path_val
      save_txt_path = save_txt_path_train if name in train_data else save_txt_path_val

      if name in train_data:
        td += 1
      else:
        vd += 1
      
      with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

      # 선택한 키의 값만 추출
      for annotation in data["annotations"]:
        # 현재 프레임 번호가 추출 간격에 맞는지 확인
        current_frame_number = annotation["frame_number"]
        # JSON 데이터에서 프레임 번호가 매번 6씩 증가하므로, 이는 실제로는 초당 약 32 / 6 ≈ 5.33장의 프레임. 
        if current_frame_number % 4 == 0:  # 매 4번째 프레임을 선택 (2fps)
          # images 리스트에서 current_frame_number와 일치하는 파일을 찾습니다.
          for img in images:
            frame_number = int(img.split('_')[1])  # 파일 이름에서 프레임 번호를 추출합니다.
            if frame_number == current_frame_number:
              frame = cv2.imread(os.path.join(frames_folder, img))  # 해당 이미지를 읽어옵니다.
              break  # 일치하는 이미지를 찾았으므로 루프를 종료합니다.
          else:
            print(f"Cannot find image file for frame {current_frame_number} in {frames_folder}.")
            continue  # 일치하는 이미지를 찾지 못했으므로 다음 annotation으로 넘어갑니다.
          # 프레임 정보 
          height, width, layers = frame.shape          
          
          # 이미지를 저장 (image와 label 데이터 이름이 동일해야 한다.)
          cv2.imwrite(os.path.join(save_img_path, f'{action[i]}_{name}-{current_frame_number}.jpg'), frame)
          # 텍스트를 저장 
          # Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
          with open(os.path.join(save_txt_path, f'{action[i]}_{name}-{current_frame_number}.txt'), 'w') as f:
            # bounding box 데이터 추출
            x, y, w, h = annotation["bounding_box"].values()   
            # coords 간 공백 유의
            # class_id : 0, 바운딩 박스 중심에 대한 좌표, 너비, 높이
            f.write(f"{0} {(x + w/2)/width} {(y + h/2)/height} {w/width} {h/height} ")   # 정규화시키기 위해 나눠준다.
            # keypoints 데이터 추출
            for keypoint in annotation["keypoints"].values():
              # visibility 정보 제외한 dimmension 2 model
              f.write(f"{(keypoint['x']/width) if keypoint else 0} {(keypoint['y']/height) if keypoint else 0} ")
          
          # json 파일 하나 처리 완료    
          cnt += 1
      print("Action_", action[i], " train 데이터 수 : ", td)
      print("Action_", action[i], " val 데이터 수 : ", vd)
      td, vd = 0, 0
              
  print("라벨링 데이터 수 : ",cnt)
  print("train 라벨 총 개수 : ", len(os.listdir(save_img_path_train)))
  print("val 라벨 총 개수 : ", len(os.listdir(save_img_path_val)))   
  print("------------- makeData() End -------------\n")
  
 
def trainModel():
  """
    description
    -----------
    Train YOLOv8 model.
    
    additional information
    ----------------------
    pip install ultralytics
    pip install "deeplabcut[tf,modelzoo]"
    pip install tensorflow==2.11
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

    pip install --upgrade --fore-reinstall ultralytics
    "which python"와 "sys.executable" 같은지 확인  
  """
  # Load a model
  model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
  # Train the model
  model.train(data='/home/dlc/DLC/_mina/data/YOLO/YOLO_pose/config.yaml',  resume=True, epochs=100, 
              project="/home/dlc/DLC/_mina/project/YOLO/YOLO_pose", name="train", imgsz=640)
  

def predict():
  """
    description
    -----------
    Predict using YOLOv8 model.
  """
  project = "/home/dlc/DLC/_mina/project/YOLO/YOLO_pose"
  name = "predict"
  # Load a model
  model = YOLO('/home/dlc/DLC/_mina/project/YOLO/YOLO_pose/train/weights/best.pt')  # load a custom model

  # Define path to video file
  source = '/home/dlc/DLC/_mina/data/experimental_videos/coco01.mp4'

  # Run inference on the source
  results = model.predict(source = source, project=project, name=name,
      stream = True, save=True, save_crop=True)
  for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

  
  
  
  # # Open the video file
  # video_path = "path/to/your/video/file.mp4"
  # cap = cv2.VideoCapture(video_path)

  # # Loop through the video frames
  # while cap.isOpened():
  #   # Read a frame from the video
  #   success, frame = cap.read()

  #   if success:
  #     # Run YOLOv8 inference on the frame
  #     results = model(frame)

  #     # Visualize the results on the frame
  #     annotated_frame = results[0].plot()

  #     # Display the annotated frame
  #     cv2.imshow("YOLOv8 Inference", annotated_frame)

  #     # Break the loop if 'q' is pressed
  #     if cv2.waitKey(1) & 0xFF == ord("q"):
  #       break
  #   else:
  #     # Break the loop if the end of the video is reached
  #     break

  # # Release the video capture object and close the display window
  # cap.release()
  # cv2.destroyAllWindows() 
  
  
  
predict()