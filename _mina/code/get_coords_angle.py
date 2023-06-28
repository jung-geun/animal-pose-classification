import os
import json
import numpy as np

def getAngles(lab_path, dataset, angs):  
    angles = angs
    for elementName in dataset:
        json_path = os.path.join(lab_path, 'json', elementName + '.json')             # img_coords가 있는 json 경로
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        
        selected_data = []
        for annotation in data['annotations']:
            #print(annotation.keys())
            #row = []
            for keypoint, coords in annotation['keypoints'].items():
                my_tuple = keypoint,coords
                my_dict = dict([my_tuple])
                new_keys = ['Nose', 'Forehead', 'MouthCorner', 'LowerLip', 'Neck', 'RightArm', 'LeftArm', 'RightWrist', 'LeftWrist', 'RightFemur', 'LeftFemur', 'RightAnkle', 'LeftAnkle', 'TailStart', 'TailTip']
                for old_key, new_key in zip(list(my_dict.keys()), new_keys):
                    print(my_dict.keys())
                    my_dict[new_key] = my_dict.pop(old_key)

                #seq = int(keypoint)
                #print(coords)
                #if(seq==1):
                #    Nose = coords
                #elif(seq==2):
                #    Forehead = coords
                #elif(seq==3):
                #    MouthCorner = coords

                print(my_dict)
            #    if keypoint is not None:
            #        row.extend([keypoint['x'], keypoint['y']])
            #    else:
            #        row.extend([None, None])
            #selected_data.append(row)

            ## - Nose - Forehead - MouthCorner - LowerLip - Neck - RightArm - LeftArm - RightWrist - LeftWrist - RightFemur - LeftFemur - RightAnkle - LeftAnkle - TailStart - TailTip
            #mouth = np.arctan((LowerLip['y']-Forehead['y'])/(LowerLip['x']-Forehead['x'])) - np.arctan((Nose['y']-Forehead['y'])/(Nose['x']-Forehead['x']))
            #angles = []
            #try:

            #except:


def angleTrain(ang):
    pass



orgLabel = "D:/DeepLabCut/AI-Hub/poseEstimation/Validation/DOG/labelSIT"
dname = ['20201111_dog-sit-001039.mp4', '20201112_dog-sit-000635.mp4'] 
# [입 벌림 : [Nose, MouthCorner, LowerLip], 오른쪽 앞다리 : [Neck, RightArm, RightWrist], 왼쪽 앞다리 : [Neck, LeftArm, LeftWrist], 오른쪽 뒷다리 : [TailStart, RightFemur, RightAnkle], 왼쪽 뒷다리 : [TailStart, LeftFemur, LeftAnkle] , 머리 : [Nose, Forehead, Neck], 몸통 기울기 : [Neck, TailStart, _Ankle], 꼬리 : [Neck, TailStart, TailTip]]   
mouth = ["Nose", "MouthCorner", "LowerLip"]
rightFront = ["Neck", "RightArm", "RightWrist"]     
leftFront = ["Neck", "LeftArm", "LeftWrist"] 
rightBack = ["TailStart", "RightFemur", "RightAnkle"] 
rightBack2 = ["Neck", "TailStart", "RightFemur"]
leftBack = ["TailStart", "LeftFemur", "LeftAnkle"]
leftBack2 = ["Neck", "TailStart", "LeftFemur"]
head = ["Nose", "Forehead", "Neck"]     
body1 = ["Nose", "Neck", "_Wrist"]  
body2 = ["Neck", "TailStart", "_Ankle"]
tail = ["Neck", "TailStart", "TailTip"] 
angles = [mouth, rightFront, leftFront, rightBack, leftBack, head, body, tail]
getAngles(orgLabel, dname, angles)


