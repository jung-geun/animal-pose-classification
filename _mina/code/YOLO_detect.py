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


def prepare():
    """
        Data pre-processing for YOLO training
        
        Args:
            None
        
        Returns:
            src_path: list of source path for each action
            lab_path: list of label path for each action
            dataset: list of video names for each action
            action: list of action names
            
        Example:
            src_path, lab_path, dataset, action = prepare()
        
    """
    data_path = "/home/dlc/DLC/_mina/data/AI-Hub/poseEstimation/Validation/DOG"
    action = ["BODYSHAKE", "BODYLOWER", "BODYSCRATCH", 
              "WALKRUN", "SIT", "HEADING", "LYING", 
              "MOUNTING", "TAILING", "TAILLOW", "TURN", 
              "FOOTUP", "FEETUP"]
    src_path = []
    lab_path = []
    dataset = []
    data_cnt, total = 0, 0
    train_data_tot = 0
    max_data_per_action = 160         # 한 동작 당 160개의 영상

    for action_name in action:
        src_path_action = data_path + "/source" + action_name
        lab_path_action = data_path + "/label" + action_name
        src_path.append(src_path_action)      # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/sourceSIT"
        lab_path.append(lab_path_action)      # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/labelSIT"

        image_folders = os.listdir(src_path_action + "/images/")
        # list comprehension
        # newlist = [expression for item in iterable if condition == True]
        # 반복이 가능한 객체에 대해 item을 반복하며 해당 item이 조건에 부합한다면 표현식에 맞춰 요소를 추가하면서 새로운 list를 생성한다.
        selected_folders = [
            folder for folder in image_folders
            if len(os.listdir(src_path_action + "/images/" + folder)) > 34
        ][:max_data_per_action]     # max_data_per_action 개수만큼 선택하여 재할당

        dataset.append(selected_folders)
        data_cnt = len(os.listdir(src_path_action + "/images/"))
        print("Action_", action_name, " 총 데이터 수 : ", data_cnt)
        print("Action_", action_name, " 학습할 데이터 수 : ", len(selected_folders))
        total += data_cnt
        train_data_tot += len(selected_folders)

    print("실제 영상 total : ", total)
    print("학습 영상 total : ", train_data_tot)
    return src_path, lab_path, dataset, action


def drawOnImages():    
    """
        각 동작 당, max_data_per_action개의 영상. 하나의 영상마다 8장의 프레임 이미지에 바운딩 박스를 그린다.
    """   
    src_path, lab_path, dataset, action = prepare()
    save_path = '/home/dlc/DLC/_mina/yolo_output2/'  # 이미지를 저장할 경로
    os.makedirs(save_path, exist_ok=True)
    image_count = 0
    for i in range(len(src_path)):
        for j in range(len(dataset[i])):
            frames_folder = (
                src_path[i] + "/images/" + dataset[i][j]
            )  # dataset의 frame들의 폴더 위치
            images = [
                img for img in os.listdir(frames_folder) if img.endswith(".jpg")
            ]
            # 파일 이름을 기준으로 정렬하여 목록 생성
            images.sort(key=lambda x: int(x.split("_")[1]))
                
            frame = cv2.imread(os.path.join(frames_folder, images[0]))      
            height, width, layers = frame.shape         # 프레임 크기

            json_path = (
                lab_path[i] + "/json/" + dataset[i][j] + ".json"
            )  # coords가 있는 json 경로

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 선택한 키의 값만 추출
            selected_data = []
            for annotation in data["annotations"]:
                row = []
                for key, value in annotation["bounding_box"].items():
                    if value is not None:
                        row.append(value)
                    else:
                        row.append(None)
                selected_data.append(row)
                
            if len(images) > 34:
                for index in [0, 2, 7, 9, 16, 20, 33, 34]:
                    frame = cv2.imread(os.path.join(frames_folder, images[index]))
                    # bounding box의 좌표와 너비, 높이
                    x, y, w, h = selected_data[index]
                    # 5px 더 큰 bounding box를 그림
                    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
                    # 박스 중점 x, y 좌표에 빨간색 점을 찍음
                    cv2.circle(frame, (int(x + w/2), int(y + h/2)), radius=5, color=(0, 0, 255), thickness=-1)
                    # 이미지를 저장
                    cv2.imwrite(os.path.join(save_path, f'{action[i]}_{i+1}-{index+1}.jpg'), frame)
            else:
                print(f"Warning: Not enough images in {frames_folder}")


def makeData():
    """
        Make images and labels for YOLOv8 detect model training
        
        Args:
            None
        
        Returns:
            None
        
        Example:
            makeData()
    """
    cnt = 0
    src_path, lab_path, dataset, action = prepare()
    save_img_path_train = '/home/dlc/DLC/_mina/data/YOLO/YOLO_detect/images/train/'  # 이미지를 저장할 경로
    save_txt_path_train = '/home/dlc/DLC/_mina/data/YOLO/YOLO_detect/labels/train/'  # 라벨 데이터를 저장할 경로
    save_img_path_val = '/home/dlc/DLC/_mina/data/YOLO/YOLO_detect/images/val/'  
    save_txt_path_val = '/home/dlc/DLC/_mina/data/YOLO/YOLO_detect/labels/val/'  
    os.makedirs(save_img_path_train, exist_ok=True)
    os.makedirs(save_txt_path_train, exist_ok=True)
    os.makedirs(save_img_path_val, exist_ok=True)
    os.makedirs(save_txt_path_val, exist_ok=True)
    
    # action 수만큼 반복
    for i in range(len(src_path)):
        # tarin, val data를 8:2 비율로 randomly split
        # k = dataset[i] 길이의 80%에 해당하는 정수 => dataset[i] 개수 중에 k개를 무작위 선택하여 저장
        train_indices = random.sample(range(len(dataset[i])), k=int(len(dataset[i])*0.8))
        
        # max_data_per_action 만큼 반복
        for j in range(len(dataset[i])):
            # dataset의 frame들의 폴더 위치
            frames_folder = src_path[i] + "/images/" + dataset[i][j]
            # 파일 이름을 기준으로 정렬하여 목록 생성
            try:
                images = sorted([img for img in os.listdir(frames_folder) if img.endswith(".jpg")], key=lambda x: int(x.split("_")[1]))
            except ValueError as e:
                print(f"\n Please CHECK !! \nError sorting images in {frames_folder}: {e}")
            
            # coords가 있는 json 경로    
            json_path = lab_path[i] + "/json/" + dataset[i][j] + ".json"

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            selected_data = [[value for key, value in annotation["bounding_box"].items()] for annotation in data["annotations"]]
                
            save_img_path = save_img_path_train if j in train_indices else save_img_path_val
            save_txt_path = save_txt_path_train if j in train_indices else save_txt_path_val
             
            # 1, 3, 8, 10, 17, 21, 34, 35 번째 프레임 사용 (frame number X, 총 8장)   
            for index in [0, 2, 7, 9, 16, 20, 33, 34]:
                frame = cv2.imread(os.path.join(frames_folder, images[index]))
                # bounding box의 좌표와 너비, 높이
                x, y, w, h = selected_data[index]   
                # 프레임 크기
                height, width, layers = frame.shape         
                
                # 이미지를 저장 (이름이 동일해야 한다.)
                cv2.imwrite(os.path.join(save_img_path, f'{action[i]}_{j+1}-{index+1}.jpg'), frame)
                # 텍스트를 저장 
                # Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
                with open(os.path.join(save_txt_path, f'{action[i]}_{j+1}-{index+1}.txt'), 'w') as f:
                    # class_id : 0, 바운딩 박스 중심에 대한 좌표, 너비, 높이
                    f.write(f"{0} {(x + w/2)/width} {(y + h/2)/height} {w/width} {h/height}\n")     # 정규화시키기 위해 나눠준다
                    
                cnt += 1
                
    print("라벨링 데이터 수 : ",cnt)
    print("train 라벨 총 개수 : ", len(os.listdir(save_img_path_train)))
    print("val 라벨 총 개수 : ", len(os.listdir(save_img_path_val)))
   
 
 
def trainModel():
    """
        pip install ultralytics
        pip install "deeplabcut[tf,modelzoo]"
        pip install tensorflow==2.11
        pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

        pip install --upgrade --fore-reinstall ultralytics
        "which python"와 "sys.executable" 같은지 확인

        train_model : YOLOv8 model 학습
    """ 
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="/home/dlc/DLC/_mina/data/YOLO/YOLO_detect/config.yaml", epochs=120,  resume=False,
                project="/home/dlc/DLC/_mina/project/YOLO/YOLO_detect", name="train")  # train the model
    # # Validate the model
    # model.val()  # evaluate model performance on the validation set
            

def predict():
    """
        Predict using YOLOv8 model
    """
    import moviepy.editor as mp
    # YOLO 모델 불러오기
    model = YOLO('/home/dlc/DLC/_mina/project/YOLO/YOLO_detect/train/weights/last.pt')

    project="/home/dlc/DLC/_mina/project/YOLO"
    name="predict"
    keyword = "snow"
    video_crop_path = "/home/dlc/DLC/_mina/yolo_output/" + keyword + ".avi"
    video_path = "/home/dlc/DLC/_mina/data/experimental_videos/" + keyword + ".mp4"
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
 
    # # 프레임 크기 조정
    # height, width, _ = frame.shape
    # clip = mp.VideoFileClip(video_path)
    # if width > height:
    #     clip_resized = clip.resize(width=640, height=360)
    # else:
    #     clip_resized = clip.resize(width=480, height=640)
    #  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    # clip_resized.write_videofile(video_crop_path)
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    
    # Run inference on the source
    results = model.predict(source = video_path, project=project, name=name,
        stream = True, save=True, save_crop=True)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
    
    # frames to video
    image_folder = os.path.join(project, name, "crops", "dog")
    video_name = os.path.join(project, name, "crops", "cropped_vdo", (keyword +".avi"))
    os.makedirs(os.path.join(project, name, "crops", "cropped_vdo"), exist_ok=True)

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    # cropped image resize
    # vdo_name, ?, fps, (width, height)
    video = cv2.VideoWriter(video_name, 0, 6.5, (640, 480))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        resized_img = cv2.resize(img, (640, 480))
        video.write(resized_img)

    cv2.destroyAllWindows()
    video.release()


def modelZoo():
    """
        modelZoo : DeepLapCut - modelZoo - superanimal_quadruped 사용
    """   
    import deeplabcut
    model_options = deeplabcut.create_project.modelzoo.Modeloptions
    project_name = 'myDLC_modelZoo'
    your_name = 'teamDLC'
    video_path = ['/home/dlc/DLC/_mina/project/YOLO/YOLO_detect/predict/crops/cropped_vdo/run.avi']
    model2use = model_options[8]            # 2 : full dog, 8 : superanimal_quadruped
    videotype = os.path.splitext(video_path[0])[-1].lstrip('.') #or MOV, or avi, whatever you uploaded!
    # video_path = deeplabcut.DownSampleVideo(video_path[1], width=300)
    working_directory = '/home/dlc/DLC/_mina/project/DLC/ModelZoo/'
    print("\nselected model : \n", model2use)
    
    config_path, train_config_path = deeplabcut.create_pretrained_project(
        project_name,
        your_name,
        video_path,
        working_directory=working_directory,
        videotype=videotype,
        model=model2use,
        analyzevideo=True,
        createlabeledvideo=True,
        copy_videos=True, #must leave copy_videos=True
    )
    print('\nconfig : \n', config_path)
    # Updating the plotting within the config.yaml file (without opening it ;):
    edits = {
        'dotsize': 3,  # size of the dots!
        'colormap': 'spring',  # any matplotlib colormap!
        'pcutoff': 0.5,  # the higher the more conservative the plotting!
    }
    deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)
    
    # re-create the labeled video (first you will need to delete in the folder to the LEFT!):
    project_path = os.path.dirname(config_path)
    full_video_path = os.path.join(
        project_path,
        'videos',
        os.path.basename(video_path),
    )

    #filter predictions (should already be done above ;):
    deeplabcut.filterpredictions(config_path, [full_video_path], videotype=videotype)

    #re-create the video with your edits!
    deeplabcut.create_labeled_video(config_path, [full_video_path], videotype=videotype, filtered=True)
    
    
    # video_path = '/home/dlc/DLC/_mina/project/YOLO/YOLO_detect/predict/crops/cropped_vdo/run.avi'
    # superanimal_name = 'superanimal_quadruped'
    # scale_list = range(200, 600, 50)  # image height pixel size range and increment
    # deeplabcut.video_inference_superanimal([video_path], superanimal_name, scale_list=scale_list)



predict()

def check_file():
    file_path = '/drive/samba/private_files/jupyter/DLC/_mina/project/DLC/ModelZoo/myDLC_modelZoo-teamDLC-2023-09-05/dlc-models/iteration-0/myDLC_modelZooSep5-trainset95shuffle1/train/models--mwmathis--DeepLabCutModelZoo-SuperAnimal-Quadruped/snapshots/673140e6dd9f7be492d77cab957f31c73a192f67'
    # file_path = '/home/dlc/DLC/_mina/project/DLC/ModelZoo/myDLC_modelZoo-teamDLC-2023-09-07/dlc-models/iteration-0/myDLC_modelZooSep7-trainset95shuffle1/train/models--mwmathis--DeepLabCutModelZoo-SuperAnimal-Quadruped/snapshots/673140e6dd9f7be492d77cab957f31c73a192f67/DLC_ma_superquadruped_resnet_50_iteration-0_shuffle-1.tar.gz'
    if os.path.exists(file_path):
        print('ok')
    else:
        print('cant found')
    print("os_cwd : ",os.getcwd())
# check_file()