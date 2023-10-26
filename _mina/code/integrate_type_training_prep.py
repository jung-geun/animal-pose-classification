import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

"""
    Dog Pose Estimation - Sensor Train :  AI-Hub 데이터셋 활용하여 데이터 전처리 & DLC 학습시키기
    
    main() 함수를 실행하면, AI-Hub 데이터셋을 활용하여 DeepLabCut을 학습시키는 과정을 진행한다.
"""
import cv2
import os
import shutil
import platform
import yaml
import json
import deeplabcut
import pandas as pd
from tqdm import tqdm


def frames_to_video(src_path, dataset, pbar):
    """
    Convert frames to AVI video.

    Parameters
    ----------
    src_path : list
        Full path of the source data folder.

    dataset : list
        List of the dataset.

    pbar : tqdm
        Progress bar object.
    """
    pbar.write("--- frames to video start ---")
    for i in range(len(src_path)):
        for j in range(len(dataset[i])):
            frames_folder = (
                src_path[i] + "/images/" + dataset[i][j]
            )  # dataset의 frame들의 폴더 위치
            video_name = src_path[i] + "/videos/" + dataset[i][j] + ".avi"  # 비디오 저장위치

            if not os.path.exists(os.path.join(src_path[i], "videos")):
                os.makedirs(os.path.join(src_path[i], "videos"))

            if not os.path.exists(video_name):
                images = [
                    img for img in os.listdir(frames_folder) if img.endswith(".jpg")
                ]
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                # 파일 이름을 기준으로 정렬하여 목록 생성
                images.sort(key=lambda x: int(x.split("_")[1]))
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                frame = cv2.imread(os.path.join(frames_folder, images[0]))
                height, width, layers = frame.shape
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                # 세 번째 인수가 FPS. 높을 수록 부드럽게 재생
                video = cv2.VideoWriter(video_name, 0, 6, (width, height))
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                for image in images:
                    video.write(cv2.imread(os.path.join(frames_folder, image)))
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                cv2.destroyAllWindows()
                video.release()
            else:
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 1)
    pbar.write("--- frames to video end ---")


## https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html#a-create-a-new-project
def create_new(project, scorer, working_directory, src_path, dataset, pbar):
    """
    Create a new project for pose estimation.

    Parameters
    ----------
    project : string
        Name of the project.

    scorer : string
        Name of the experimenter.

    working_directory : string, optional
        Full path of the working directory.

    src_path : list
        Full path of the source data folder.

    dataset : list
        List of the action of dataset.

    pbar : tqdm
        Progress bar object.

    Returns
    -------
    config_path : string
        Full path of the config.yaml file as a string.

    vdos : list
        List of the video files.
    """
    pbar.write("--- create new project start ---")
    vdos = []  # video set
    for i in range(len(src_path)):
        pbar.update((1 / len(src_path)) * 0.2)

        if not os.path.exists(src_path[i] + "/videos"):
            os.makedirs(src_path[i] + "/videos")
        pbar.update((1 / len(src_path)) * 0.2)

        for j in range(len(dataset[i])):
            full_path_of_video = src_path[i] + "/videos/" + dataset[i][j] + ".avi"
            if platform.system() == "Windows":
                inputPath = full_path_of_video.replace("/", "\\")
                vdos.append(inputPath)
            else:
                vdos.append(full_path_of_video)
        pbar.update((1 / len(src_path)) * 0.3)

    config_path = deeplabcut.create_new_project(
        project, scorer, vdos, working_directory=working_directory, copy_videos=True
    )
    pbar.update(1 - (0.2 + 0.2 + 0.3))
    pbar.write("--- create new project end ---")
    return config_path, vdos


## https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/create_project/new.py#L21
def config_edit(config_path, pbar):
    """
    Edit the config.yaml file to overwrite the bodyparts and skeleton, etc.

    Parameters
    ----------
    config_path : string
        Full path of the config.yaml file as a string.

    pbar : tqdm
        Progress bar object.
    """
    pbar.write("--- config edit start ---")
    with open(config_path, "r") as file:
        cfg_file = yaml.safe_load(file)
    pbar.update(0.5)

    cfg_file["bodyparts"] = [
        "Nose",
        "Forehead",
        "MouthCorner",
        "LowerLip",
        "Neck",
        "RightArm",
        "LeftArm",
        "RightWrist",
        "LeftWrist",
        "RightFemur",
        "LeftFemur",
        "RightAnkle",
        "LeftAnkle",
        "TailStart",
        "TailTip",
    ]
    cfg_file["skeleton"] = [
        ["Nose", "Forehead"],
        ["Forehead", "MouthCorner"],
        ["Forehead", "Neck"],
        ["Neck", "TailStart"],
        ["TailStart", "TailTip"],
        ["Lowerlip", "MouthCorner"],
        ["Neck", "RightArm"],
        ["Neck", "LeftArm"],
        ["TailStart", "RightFemur"],
        ["TailStart", "LeftFemur"],
        ["RightArm", "RightWrist"],
        ["LeftArm", "LeftWrist"],
        ["RightFemur", "RightAnkle"],
        ["LeftFemur", "LeftAnkle"],
    ]
    cfg_file["skeleton_color"] = "white"
    cfg_file["dotsize"] = 6

    with open(config_path, "w") as file:
        yaml.dump(cfg_file, file)
    pbar.update(0.5)
    pbar.write("--- config edit end ---")


## https://github.com/DeepLabCut/DeepLabCut/wiki/Using-labeled-data-in-DeepLabCut-that-was-annotated-elsewhere/35eb6bc2079d3e1125dafa87b05e07e47df388d3
def json_to_csv(scorer, src_path, lab_path, conclusion_path, dataset, pbar):
    """
    Extract joint coordinates in a JSON file and convert to a CSV file.

    Parameters
    ----------
    scorer : string
        Name of the experimenter.

    src_path : list
        Full path of the source data folder.

    lab_path : list
        Full path of the labeled data folder.

    conclusion_path : string
        Full path of the created project folder.

    dataset : list
        List of the dataset.

    pbar : tqdm
        Progress bar object.
    """
    pbar.write("--- json to csv start ---")
    for i in range(len(src_path)):
        for j in range(len(dataset[i])):
            json_path = (
                lab_path[i] + "/json/" + dataset[i][j] + ".json"
            )  # coords가 있는 json 경로
            csv_path = lab_path[i] + "/csv/" + dataset[i][j] + ".csv"  # 저장될 csv 경로
            frames_folder = src_path[i] + "/images/" + dataset[i][j]  # frames 폴더 경로
            labeled_path = "labeled-data/" + dataset[i][j] + "/"  # ImgPrefixPath
            copy_csv_path = (
                conclusion_path
                + "/labeled-data/"
                + dataset[i][j]
                + "/CollectedData_"
                + scorer
                + ".csv"
            )  # copied csv save 경로
            copy_img_src = frames_folder  # copy 할 framses directory 경로
            copy_img_dst = (
                conclusion_path + "/labeled-data/" + dataset[i][j]
            )  # frames copy and save 경로
            pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

            if not os.path.exists(lab_path[i] + "/csv/"):
                os.makedirs(lab_path[i] + "/csv/")

            if not os.path.exists(csv_path) or not os.path.exists(copy_csv_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 선택한 키의 값만 추출
                selected_data = []
                for annotation in data["annotations"]:
                    row = []
                    for keypoint in annotation["keypoints"].values():
                        if keypoint is not None:
                            row.extend([float(keypoint["x"]), float(keypoint["y"])])
                        else:
                            row.extend([None, None])
                    selected_data.append(row)
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                # Create a DataFrame with the selected data and column names
                df = pd.DataFrame(
                    selected_data, columns=[scorer] * len(selected_data[0])
                )

                df.loc[-1] = ["x", "y"] * ((len(df.columns)) // 2)
                df.index = df.index + 1
                df = df.sort_index()

                # 리스트 내포(list comprehension)를 사용하여 각 요소가 두 번씩 반복되는 리스트 생성
                bodyparts = [
                    "Nose",
                    "Forehead",
                    "MouthCorner",
                    "LowerLip",
                    "Neck",
                    "RightArm",
                    "LeftArm",
                    "RightWrist",
                    "LeftWrist",
                    "RightFemur",
                    "LeftFemur",
                    "RightAnkle",
                    "LeftAnkle",
                    "TailStart",
                    "TailTip",
                ]
                df.loc[-1] = [part for part in bodyparts for _ in range(2)]
                df.index = df.index + 1
                df = df.sort_index()
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                images = [
                    img for img in os.listdir(frames_folder) if img.endswith(".jpg")
                ]
                images.sort(key=lambda x: int(x.split("_")[1]))
                project_dataPath = labeled_path  # 앞에 labeled-data path 추가
                images = [project_dataPath + img for img in images]
                df.insert(0, "scorer", ["bodyparts", "coords"] + images)  # 0번 째 열에 삽입

                # Save the resulting DataFrame to a CSV file
                df.to_csv(csv_path, index=False, header=True)
                shutil.copy(csv_path, copy_csv_path)
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)

                # Copy the sourceFrames
                for file_name in os.listdir(copy_img_src):
                    src_file_path = os.path.join(copy_img_src, file_name)
                    dst_file_path = os.path.join(copy_img_dst, file_name)
                    shutil.copy2(src_file_path, dst_file_path)
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 0.2)
            else:
                pbar.update((1 / (len(src_path) * len(dataset[i]))) * 1)
    pbar.write("--- json to csv end ---")


## https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/utils/conversioncode.py#L30
def convert_csv(config_path, scorer, pbar):
    """
    Convert (image) annotation files in folder labeled-data from csv to h5.
    This function allows the user to manually edit the csv (e.g. to correct the scorer name and then convert it into hdf format).
    WARNING: conversion might corrupt the data.

    Parameters
    ----------
    config_path : string
        Full path of the config.yaml file as a string.

    userfeedback: bool, optional
        If true the user will be asked specifically for each folder in labeled-data if the containing csv shall be converted to hdf format.

    scorer: string, optional
        If a string is given, then the scorer/annotator in all csv and hdf files that are changed, will be overwritten with this name.

    pbar : tqdm
        Progress bar object.

    Examples
    --------
    Convert csv annotation files for reaching-task project into hdf.
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml')

    --------
    Convert csv annotation files for reaching-task project into hdf while changing the scorer/annotator in all annotation files to Albert!
    >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml',scorer='Albert')
    --------
    """
    pbar.write("--- convert csv to h5 start ---")
    deeplabcut.convertcsv2h5(config_path, userfeedback=False, scorer=scorer)
    pbar.update(1)
    pbar.write("--- convert csv to h5 end ---")


## https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/gui/tabs/create_training_dataset.py#L84
def train(config_path, vdos, pbar, destfolder=None):
    """
    Train a network for a project.

    Parameters
    ----------
    config_path : string
        Full path of the config.yaml file as a string.

    vdos : list
        List of video files to use for training.

    pbar : tqdm
        Progress bar object.

    destfolder : string or None, optional, default=None
        Specifies the destination folder that was used for storing analysis data.
        If 'None', the path of the video file is used.
    """
    pbar.write("--- train start ---")
    ## create training dataset
    deeplabcut.create_training_dataset(config_path, augmenter_type="imgaug")
    pbar.update(0.5)

    ## start training
    # deeplabcut.train_network(config_path,shuffle=1,trainingsetindex=0,gputouse=None,max_snapshots_to_keep=5,autotune=False,displayiters=300,saveiters=1500,maxiters=30000,allow_growth=True)
    deeplabcut.train_network(
        config_path,
        shuffle=1,
        trainingsetindex=0,
        gputouse=0,
        max_snapshots_to_keep=5,
        autotune=True,
        displayiters=200,
        saveiters=2500,
        maxiters=50000,
        allow_growth=True,
    )
    pbar.update(0.5)
    pbar.write("--- train end ---")

    ## start evaluating
    deeplabcut.evaluate_network(config_path,Shuffles=[1],plotting=True)

    # ## model video analyzing
    # deeplabcut.analyze_videos(config_path, vdos)
    # deeplabcut.filterpredictions(config_path,vdos)

    # ## create labeled video
    # deeplabcut.create_labeled_video(config_path,vdos, draw_skeleton=True, destfolder=destfolder)
    # deeplabcut.plot_trajectories(config_path,vdos,showfigures=True)


## AI-Hub의 데이터셋을 이용하여 DLC 학습하기 : frames_to_video -> create_new_project  -> config_edit -> json_to_csv -> convert_csv -> create_training_dataset -> training in CoLab
def start(project, scorer, working_directory, src_path, lab_path, dataset, destfolder):
    """
    Data pre-processing and sensor training.

    Parameters
    ----------
    project : string
        Project name.

    scorer : string
        Name of the experimenter.

    working_directory : string, optional
        Full path of the working directory.

    destfolder : string or None, optional, default=None
        Where the labeled video will be stored.
        If 'None', the path of your project's videos folder is used.

    src_path : list
        Full path of the source data folder. Frame-by-frame image files, and the folder where the video files will be stored.

    lab_path : list
        Full path of the label data folder. The folder where the JSON file is located, and where the CSV file will be saved.

    dataset : list
        List of the dataset folder names for training and analysis.

    More detailed explanation
    -------------------------
    total_steps : int
        Total number of steps.

    pbar : tqdm
        Progress bar object.

    config_path : string
        Created project config path.

    vdos : list
        List of the video files used for training.

    conclusion_path : string
        Created project folder full path.

    frames_to_video( ) : Convert frames to AVI video.
    create_new( ) : Create a new project.
    config_edit( ) : Edit the config.yaml file to overwrite the bodyparts and skeleton, etc.
    json_to_csv( ) : Convert JSON to CSV.
    convert_csv( ) : Convert CSV to HDF5.
    train( ) : Training dataset.
    """
    total_steps = 6
    with tqdm(total=total_steps, ncols=85) as pbar:
        pbar.write("\n----- START -----\n")
        frames_to_video(src_path, dataset, pbar)
        config_path, vdos = create_new(
            project, scorer, working_directory, src_path, dataset, pbar
        )
        conclusion_path = os.path.dirname(config_path)
        config_edit(config_path, pbar)
        json_to_csv(scorer, src_path, lab_path, conclusion_path, dataset, pbar)
        convert_csv(config_path, scorer, pbar)  # --> matrix 주의
        train(
            config_path, vdos, pbar, destfolder
        )  # colab에서 학습 진행할 경우, train() 함수는 colab에서 실행
        pbar.write("\n----- THE END -----\n")


def get_specific_dataset(breed, src_path, lab_path, dataset, action):
    """
    강아지 종이 특정견인 경우만 dataset에 추가
    Add only the case where the breed of the dog is a specific dog to the dataset.
    
    Parameters
    ----------
    breed : list
        List of the dog breeds.
    
    src_path : list
        Full path of the source data folder.
        
    lab_path : list
        Full path of the label data folder.
    
    dataset : list
        List of the dataset folder names for training and analysis.
    
    action : list
        List of actions.
    
    Returns
    -------
    real_dataset = specific_dataset : list
        List of the dataset folder names for training and analysis.
    """
    # src_path 길이만큼 2차 행렬 생성
    real_dataset = [[] for _ in range(len(src_path))]

    for i in range(len(src_path)):
        # {'poodle': 0, 'bichon': 0, 'shih tzu': 0}
        breed_count = {key: 0 for key in breed}  # Initialize breed count dictionary for this action
        for j in range(len(dataset[i])):
            json_path = lab_path[i] + "/json/" + dataset[i][j] + ".json"
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # breed가 특정견인 경우만 추출
            if data["metadata"]["animal"]["breed"] in breed:
                # real_dataset의 i번째 action에 추가
                real_dataset[i].append(dataset[i][j])
                breed_count[data["metadata"]["animal"]["breed"]] += 1  # Increase count for this breed

        # Print count for each breed for this action
        for b in breed:
            print(action[i], "하는 ", b, " total : ", breed_count[b])
    
    # 하나만 잘라서 새로운 2차원 배열로 테스트 return
    real_dataset = [real_dataset[0][:1]]
    print("\nreal_dataset all : ", real_dataset)
    return real_dataset



def main():
    """
    Make settings before starting the programme.

    Variable description
    --------------------
    Variables that need to be initialized
    -------------------------------------
    action : list
        List of the action of dataset.
        
    breed : list
        Name of the dog breed.
    
    breedEN : string
        Name of the dog breed in English.
        
    project : string
        Name of the project to create.

    scorer : string
        Name of the experimenter.

    data_path : string
        Full path of the source data folder.

    working_directory : string, optional
        Full path of the working directory.

    destfolder : string or None, optional, default=None
        Where the labeled video will be stored.
        If 'None', the path of your project's videos folder is used.
    

    Variables that don't need to be declared
    ----------------------------------------
    src_path : list
        Full path of the source data folder. Frame-by-frame image files, and the folder where the video files will be stored.

    lab_path : list
        Full path of the label data folder. The folder where the JSON file is located, and where the CSV file will be saved.

    dataset : list
        List of the dataset folder names for training and analysis.
        
    specific_dataset : list
        List of the dataset folders for specific species for training and analysis.

    data_cnt : int
        Number of the data.

    total : int
        Total number of the data.
    """
    # 말티즈, 비숑 프리제, 푸들,  치와와, 미니어처 핀셔, 닥스훈트,  포메라니안, 웰시 코기 (펨브록), 시추, 믹스견, 셔틀랜드 쉽독,  진돗개, 래브라도 리트리버, 보더 콜리 ...
    
    # action = ["BODYLOWER", "BODYSCRATCH", "BODYSHAKE",
    #           "FEETUP", "FOOTUP", "HEADING", "LYING",
    #           "MOUNTING","SIT", "TAILING", "TAILLOW", 
    #           "TURN", "WALKRUN"]
    # breed = ["말티즈", "비숑 프리제", "푸들"]
    action = ["WALKRUN"]
    breed = ["래브라도 리트리버"]
    breedEN = "Labrador_Retriever"
    project = "preLabeled_type("+breedEN+")"
    scorer = "test01"  # project 유/무 확인하기 
    data_path = "/home/dlc/DLC/_mina/data/AI-Hub/poseEstimation/Validation/DOG"
    working_directory = "/home/dlc/DLC/_mina/project/DLC"
    destfolder = "/home/dlc/DLC/_mina"

    print("\n\n")
    src_path = []
    lab_path = []
    dataset = []
    data_cnt, total = 0, 0

    for i in range(len(action)):
        src_path.append(
            data_path + "/source" + action[i]
        )  # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/sourceSIT"
        lab_path.append(
            data_path + "/label" + action[i]
        )  # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/labelSIT"
        dataset.append(
            os.listdir(src_path[i] + "/images/")
        )  # [['20201029_dog-sit-000219.mp4', 'dog-sit-000243.mp4']]
        data_cnt = len(os.listdir(src_path[i] + "/images/"))
        print("Action_", action[i] , " : ", data_cnt)
        total += data_cnt    
    specific_dataset = get_specific_dataset(breed, src_path, lab_path, dataset, action)
    # print("\nreal_dataset sample : ")
    # for i in range(len(specific_dataset)):
    #     print(specific_dataset[i][:3])
    print("\nwhole_dog_dataset total : ", total)
    # specific_dataset 안에 있는 전체 데이터 개수 출력
    print("specific_dog_dataset total : ", sum(len(x) for x in specific_dataset))
    
    
    # 폴더 이름이 project-scorer로 시작하는 폴더가 존재하면, 이미 생성된 project가 존재한다는 의미
    # 이미 생성된 project 있으면 한번 더 확인해서 실행 여부 물어보고 실행 or 종료
    folder_name = project + "-" + scorer
    folder_exists = any(folder.startswith(folder_name) for folder in os.listdir(working_directory))  
    if folder_exists:
        print("이미 생성된 project가 존재합니다.")
        print("기존 project를 덮어쓰고 다시 학습하시겠습니까? (y/n)")
        answer = input()
        if answer == "y":
            start(
                project, scorer, working_directory, src_path, lab_path, specific_dataset, destfolder=None
            )
            print("Overwrite")
        elif answer == "n":
            print("Done")
            exit()
        else:
            print("잘못된 입력")
            exit()
    else:
        start(
            project, scorer, working_directory, src_path, lab_path, specific_dataset, destfolder=None
        )


# main()



def additional():
    # ## start evaluating
    # deeplabcut.evaluate_network(config_path,Shuffles=[1],plotting=True)

    # ## model video analyzing
    # deeplabcut.analyze_videos(config_path, vdos)
    # deeplabcut.filterpredictions(config_path,vdos)

    # ## create labeled video
    # deeplabcut.create_labeled_video(config_path,vdos, draw_skeleton=True, destfolder=destfolder)
    # deeplabcut.plot_trajectories(config_path,vdos,showfigures=True)

    project_path = "/home/dlc/DLC/_mina/project/DLC/preLabeled_type(Labrador_Retriever)-test01-2023-09-19"
    config_path = os.path.join(project_path, "config.yaml")
    vdos = [
        "/home/dlc/DLC/_mina/project/DLC/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/videos/dog-walkrun-085352.avi"
    ]
    # video = os.path.join(project_path, "videos", "dog-walkrun-099821.avi")
    # vdos.append(video)


    deeplabcut.analyze_videos(config_path, videos=vdos, videotype='.avi', save_as_csv=True, destfolder=None, shuffle=1)
    deeplabcut.create_labeled_video(config_path, videos=vdos, videotype='.avi', draw_skeleton=True, keypoints_only=False)
    deeplabcut.export_model('/home/dlc/DLC/_mina/project/DLC/preLabeled_type(Labrador_Retriever)-test01-2023-09-19/config.yaml')
# additional()
