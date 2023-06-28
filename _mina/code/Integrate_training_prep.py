import cv2
import os
import shutil
import yaml
import json
import deeplabcut
import pandas as pd
from tqdm import tqdm



def frames_to_video(src_path, dataset):
    """
        Convert frames to AVI video.

        Parameters
        ----------
        src_path : list
            Full path of the source data folder.
        
        dataset : list
            List of the dataset.
    """
    for i in range(len(src_path)):
        for j in range(len(dataset[i])):
            frames_folder = src_path[i] + '/images/' + dataset[i][j]          # dataset의 frame들의 폴더 위치
            video_name = src_path[i] + '/videos/' + dataset[i][j] + '.avi'           # 비디오 저장위치

            # 리스트 내포(list comprehension) => images = []    for img in os.listdir(frames_folder):   if img.endswith(".jpg"):   images.append(img)
            images = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
            # 파일 이름을 기준으로 정렬하여 목록 생성
            images.sort(key=lambda x: int(x.split('_')[1]))
            frame = cv2.imread(os.path.join(frames_folder, images[0]))
            height, width, layers = frame.shape

            # 세 번째 인수가 FPS. 높을 수록 부드럽게 재생
            video = cv2.VideoWriter(video_name, 0, 6, (width,height))

            for image in images:
                video.write(cv2.imread(os.path.join(frames_folder, image)))

            cv2.destroyAllWindows()
            video.release()



## https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html#a-create-a-new-project
def create_new(project, scorer, working_directory, src_path, dataset):
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

        Returns
        -------
        config_path : string
            Full path of the config.yaml file as a string.

        vdos : list
            List of the video files.
    """
    vdos = []       # video set 
    for i in range(len(src_path)):
        if not os.path.exists(src_path[i] + '/videos'):
            os.makedirs(src_path[i] + '/videos')
        for j in range(len(dataset[i])):
            full_path_of_video = src_path[i] + '/videos/' + dataset[i][j] + '.avi'
            inputPath = full_path_of_video.replace('/', '\\')
            vdos.append(inputPath)
    config_path = deeplabcut.create_new_project(project, scorer, vdos, working_directory=working_directory, copy_videos=True)         
    return config_path, vdos



## https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/create_project/new.py#L21
def config_edit(config_path):
    """
        Edit the config.yaml file to overwrite the bodyparts and skeleton, etc.

        Parameters
        ----------
        config_path : string
            Full path of the config.yaml file as a string.
    """
    with open(config_path, 'r') as file:
            cfg_file = yaml.safe_load(file)

    cfg_file['bodyparts'] = ['Nose', 'Forehead', 'MouthCorner', 'LowerLip', 'Neck', 'RightArm', 'LeftArm', 'RightWrist', 'LeftWrist', 'RightFemur', 'LeftFemur', 'RightAnkle', 'LeftAnkle', 'TailStart', 'TailTip']
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

    with open(config_path, 'w') as file:
        yaml.dump(cfg_file, file)



## https://github.com/DeepLabCut/DeepLabCut/wiki/Using-labeled-data-in-DeepLabCut-that-was-annotated-elsewhere/35eb6bc2079d3e1125dafa87b05e07e47df388d3
def json_to_csv(scorer, src_path, lab_path, conclusion_path, dataset):
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
    """
    for i in range(len(src_path)):
        for j in range(len(dataset[i])):
            json_path = lab_path[i] + '/json/' + dataset[i][j] + '.json'             # coords가 있는 json 경로
            if not os.path.exists(lab_path[i] + '/csv/'):
                os.makedirs(lab_path[i] + '/csv/')
            csv_path = lab_path[i] + '/csv/' + dataset[i][j] + '.csv'            # 저장될 csv 경로
            frames_folder = src_path[i] + '/images/' + dataset[i][j]             # frames 폴더 경로
            labeled_path = 'labeled-data/' + dataset[i][j] + '/'              # ImgPrefixPath
            copy_csv_path = conclusion_path + '/labeled-data/' + dataset[i][j] + '/CollectedData_' + scorer + '.csv'            # copied csv save 경로
            copy_img_src = frames_folder                # copy 할 framses directory 경로
            copy_img_dst = conclusion_path + '/labeled-data/' + dataset[i][j]           # frames copy and save 경로

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 선택한 키의 값만 추출
            selected_data = []
            for annotation in data['annotations']:
                row = []
                for keypoint in annotation['keypoints'].values():
                    if keypoint is not None:
                        row.extend([float(keypoint['x']), float(keypoint['y'])])
                    else:
                        row.extend([None, None])
                selected_data.append(row)

            # Create a DataFrame with the selected data and column names
            df = pd.DataFrame(selected_data, columns=[scorer] * len(selected_data[0]))

            df.loc[-1] = ['x', 'y'] * ((len(df.columns)) // 2)
            df.index = df.index + 1
            df = df.sort_index()
        
            # 리스트 내포(list comprehension)를 사용하여 각 요소가 두 번씩 반복되는 리스트 생성
            bodyparts = ['Nose', 'Forehead', 'MouthCorner', 'LowerLip', 'Neck', 'RightArm', 'LeftArm', 'RightWrist', 'LeftWrist', 
                         'RightFemur', 'LeftFemur', 'RightAnkle', 'LeftAnkle', 'TailStart', 'TailTip']
            df.loc[-1] = [part for part in bodyparts for _ in range(2)]
            df.index = df.index + 1
            df = df.sort_index()

            images = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
            images.sort(key=lambda x: int(x.split('_')[1]))
            project_dataPath = labeled_path             # 앞에 labeled-data path 추가                  
            images = [project_dataPath + img for img in images]
            df.insert(0, 'scorer', ['bodyparts', 'coords'] + images)            # 0번 째 열에 삽입

            # Save the resulting DataFrame to a CSV file
            df.to_csv(csv_path, index=False, header=True)
            shutil.copy(csv_path, copy_csv_path)
            # Copy the sourceFrames 
            for file_name in os.listdir(copy_img_src):
                src_file_path = os.path.join(copy_img_src, file_name)
                dst_file_path = os.path.join(copy_img_dst, file_name)
                shutil.copy2(src_file_path, dst_file_path)



## https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/utils/conversioncode.py#L30
def convert_csv(config_path, scorer):
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

        Examples
        --------
        Convert csv annotation files for reaching-task project into hdf.
        >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml')

        --------
        Convert csv annotation files for reaching-task project into hdf while changing the scorer/annotator in all annotation files to Albert!
        >>> deeplabcut.convertcsv2h5('/analysis/project/reaching-task/config.yaml',scorer='Albert')
        --------
    """
    deeplabcut.convertcsv2h5(config_path, userfeedback=False, scorer=scorer)



## https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/gui/tabs/create_training_dataset.py#L84 
def train(config_path, vdos):
    """
        Train a network for a project.

        Parameters
        ----------
        config_path : string
            Full path of the config.yaml file as a string.

        vdos : list
            List of video files to use for training.
    """
    ## create training dataset
    deeplabcut.create_training_dataset(config_path,augmenter_type='imgaug')
    
    ## start training
    deeplabcut.train_network(config_path,shuffle=1,trainingsetindex=0,gputouse=None,max_snapshots_to_keep=5,autotune=False,displayiters=300,saveiters=1500,maxiters=30000,allow_growth=True)
    
    # ## start evaluating
    # deeplabcut.evaluate_network(config_path,Shuffles=[1],plotting=True)
    
    # ## model video analyzing
    # deeplabcut.analyze_videos(config_path, vdos)
    # deeplabcut.filterpredictions(config_path,vdos)
    
    # ## create labeled video
    # deeplabcut.create_labeled_video(config_path,vdos,draw_skeleton=True)
    # deeplabcut.plot_trajectories(config_path,vdos,showfigures=True)






## AI-Hub의 데이터셋을 이용하여 DLC 학습하기 : frames_to_video -> create_new_project  -> config_edit -> json_to_csv -> convert_csv -> create_training_dataset -> training in CoLab
def start(project, scorer, working_directory, src_path, lab_path, dataset):
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

        src_path : list
            Full path of the source data folder. Frame-by-frame image files, and the folder where the video files will be stored.

        lab_path : list
            Full path of the label data folder. The folder where the JSON file is located, and where the CSV file will be saved.

        dataset : list
            List of the dataset folder names for training and analysis.

        More detailed explanation
        -------------------------
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
    total_steps = 7
    with tqdm(total=total_steps) as pbar:
        frames_to_video(src_path, dataset)
        pbar.update(1)

        config_path, vdos = create_new(project, scorer, working_directory, src_path, dataset)
        pbar.update(1)

        conclusion_path = os.path.dirname(config_path)
        pbar.update(1)

        config_edit(config_path)
        pbar.update(1)

        json_to_csv(scorer, src_path, lab_path, conclusion_path, dataset)
        pbar.update(1)

        convert_csv(config_path, scorer)        # --> matrix 주의
        pbar.update(1)

        # colab에서 학습 진행할 경우, train() 함수는 colab에서 실행
        train(config_path, vdos)               
        pbar.update(1)



def main():
    """
        Make settings before starting the programme.

        Variable description
        --------------------        
        project : string
            Name of the project to create.

        scorer : string
            Name of the experimenter.

        data_path : string
            Full path of the source data folder.

        working_directory : string, optional
            Full path of the working directory.

        action : list
            List of the action of dataset. 

        src_path : list
            Full path of the source data folder. Frame-by-frame image files, and the folder where the video files will be stored.

        lab_path : list
            Full path of the label data folder. The folder where the JSON file is located, and where the CSV file will be saved.

        dataset : list
            List of the dataset folder names for training and analysis.

        data_cnt : int
            Number of the data.

        total : int
            Total number of the data.
    """
    project = 'preLabeled'
    scorer = 'test01'
    data_path = "/home/dlc/DLC/_mina/data/AI-Hub/poseEstimation/Validation/DOG"
    working_directory=r'\home\dlc\DLC\_mina\project'
    action = ["SIT", "LYING", "BODYLOWER"]                 
    src_path = []                  
    lab_path = []                  
    dataset = []                 
    data_cnt, total = 0 , 0

    for i in range(len(action)):
        src_path.append(data_path + '/source' + action[i])      # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/sourceSIT"
        lab_path.append(data_path + '/label' + action[i])       # "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/labelSIT"
        dataset.append(os.listdir(src_path[i]+'/images/'))      # [['20201029_dog-sit-000219.mp4', 'dog-sit-000243.mp4']]
        data_cnt = len(os.listdir(src_path[i]+'/images/'))
        print('Action_',action[i],' : ', data_cnt)
        total += data_cnt
    print('total : ', total)

    start(project, scorer, working_directory, src_path, lab_path, dataset)
main()











#import h5py
#f = h5py.File('D:/DeepLabCut/AI-Hub/poseEstimation/preLabeled-test02-2023-06-21/labeled-data/20201112_dog-sit-000635.mp4/CollectedData_test02.h5','r')
#print(list(f))
#print(list(f['df_with_missing']))
#for data in list(f['df_with_missing']):
#    print(list(f['df_with_missing'][data]))

#deeplabcut.launch_dlc()
