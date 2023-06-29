# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 디버그 메시지 끄기

import tensorflow as tf

# gpu 사용 확인
print(tf.test.gpu_device_name())

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

from tensorflow import keras
from keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Dense, Embedding, Dropout, GRU
import matplotlib.pyplot as plt # 데이터 시각화

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle

import natsort
import json

from tqdm import tqdm

from angle_out import out
import numpy as np

# tf.keras.backend.clear_session()

def make_lstm():
    model = keras.Sequential()
    model.add(layers.LSTM(4, activation='relu', input_shape=(4, 1), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(14, activation='relu', return_sequences=False))
    # model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

def make_gru():
    model = keras.Sequential()
    model.add(layers.GRU(4, activation='relu', input_shape=(4, 1), dropout = 0.2, return_sequences=True))
    # model.add(layers.Dropout(0.2))
    model.add(layers.GRU(14, activation='relu', return_sequences=False))
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

def make_random_forest(n_estimators =100, max_depth=10, min_samples_split=2, random_state=0):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    return model

# %%
# meta_data = []
data = []
label = []
window = 4

path = "./data"
label_path = "./label"


keypoint = [
    "nose",
    "forehead",
    "mouth",
    "under_mouth",
    "neck",
    "right_front_start",
    "left_front_start",
    "right_front_ankle",
    "left_front_ankle",
    "right_thigh",
    "left_thigh",
    "right_back_ankle",
    "left_back_ankle",
    "tail_start",
    "tail_end"
]

label_parts = [
    ["under_mouth", "nose", "forehead", "neck"],
    ["nose", "forehead", "neck", "tail_start"],
    ["forehead", "neck", "tail_start", "tail_end"],
    ["forehead", "neck", "right_front_start", "right_front_ankle"],
    ["forehead", "neck", "left_front_start", "left_front_ankle"],
    ["neck", "tail_start", "right_thigh", "right_back_ankle"],
    ["neck", "tail_start", "left_thigh", "left_back_ankle"],
]

iii = 0

# print("data")

for data_list in natsort.natsorted(os.listdir(path)):
    if data_list == "TEST":
        continue
    elif "_DLC" in data_list:
        continue 
    elif "_bak" in data_list:
        continue
    elif not os.path.isdir(f"{path}/{data_list}"):
        continue
    print(f"{path}/{data_list} > ")
        
    delta = f"{path}/{data_list}"
    for label_list in natsort.natsorted(os.listdir(delta)):
        iii += 1
        # _json_ = label_list
        # delta_np = f"{delta}/{delta_list}/{delta_list}.npy"
        delta_label = f"{label_path}/{data_list}/{label_list}.json"
        # delta_label = f"{delta}/{delta_list}/{delta_list}.json"
        
        # print(f"{delta_label} > ")
        
        keypoints = []
        
        if not os.path.exists(delta_label):
            continue
        
        with open(delta_label, "r") as label_json:
            label_tmp = json.load(label_json)
            for annotation in label_tmp['annotations']:
                # frame = annotation['frame_number']
                # timestamp = annotation['timestamp']
                key = annotation['keypoints']
                # print(keypoint)
                # if keypoint == None:
                    # keypoint['x'] = 0
                    # keypoint['y'] = 0
                # print(annotation['keypoints'])
                # if None in key:
                # print(key)
                for val in key:
                    # print(val)
                    if key[val] == None:
                        key[val] = {'x' : 0 , 'y' : 0}
                keypoints.append(key)
                # print(keypoints) 
            # x_tmp = json.load(label_json)["annotations"][]["keypoints"]
            
        # print(len(data))
        # print(len(label))
        # print(label_tmp)
    
                
        # np_tmp = np.load(delta_np)
        tmp = np.array(keypoints)
        # print(tmp[0])
        
        # print(tmp[0].shape)
        meta_data = []
        # angle_arr = []
        for data_index in tmp:
            # print(data_index)
            data_tmp = []
            for index in data_index:
                # print(data_index[index])
                data_tmp.append(data_index[index])
            # angle = out(inputs = data_index, keypoint= keypoint, label_parts = label_parts)
            # angle_arr.append(angle)
        meta_data.append(np.array(data_tmp))
            # print(data_tmp)
        
        angle_arr = []
        # print(meta_data)
        for i in range(len(meta_data)):
            angle = out(inputs = meta_data[i], keypoint= keypoint, label_parts = label_parts)
            angle_arr.append(angle)
        
        # print(angle_arr)
        # print(f"window {window}")
        # print(f"len {len(meta_data)}")
        # print(f"data > {meta_data[0]}")
        # print(f"angle > {angle_arr}")
        np_tmp = []
        for index in range(window,len(angle_arr[0])):
            tmp_wind = angle_arr[0][index-window:index]
            
            data.append(tmp_wind)
            np_tmp.append(tmp_wind)
            label.append(data_list)
        
        np_save = f"{label_path}/{data_list}/{label_list}.npy"
        np.save(np_save, np_tmp)
        # label.append(data_list)
            
            
# print(f"count = {iii}")
            
# print(f" {len(data)}")
# print(len(label))

# angle_arr = []
# data_t = []


    # print(meta_data[index])
    # print(label[index])
    # for i in range(len(data[index])):
    # print(data[index])
        # angle = out(inputs = meta_data[index], keypoint= keypoint, label_parts = label_parts)
        # data_t.append(angle)
    
    # for index in range(window,len(np_tmp)):
    #     data.append(angle_arr[index-window:index])
    #     label.append(label_tmp)
    
# print(f"shape > {np.shape(data)}")
print(f"data sample > {data[0]}")
        # angle_arr = []
        # for i in range(len(np_tmp)):
        #     angle = out(inputs = np_tmp[i])
        #     angle_arr.append(angle)

        #     for index in range(window,len(np_tmp)):
        #         data.append(angle_arr[index-window:index])
        #         label.append(label_tmp)

# print(np.shape(data))
x_train = np.array(data)
# x_train = x_train.reshape(-1,4,10)
y_train = np.array(label)

print(f"x shape > {x_train.shape}")
print(f"y shape > {y_train.shape}")

# %%
word_to_index = {"BODYLOWER" : 0, "BODYSCRATCH" : 1, "BODYSHAKE" : 2, "FOOTUP" : 3, "HEADING" : 4, "LYING" : 5, "MOUNTING" : 6, "SIT" : 7, "TURN" : 8}

def convert_word_to_index(word_to_index, sentences):
    arr = []
    for i in range(len(sentences)):
        arr.append(word_to_index[sentences[i]])
    arr = np.array(arr)
    return arr
    
def one_hot_encoding(words, word_to_index):
    ohv = []
    for word in words:
        one_hot_vector = [0]*(len(word_to_index))
        index = word_to_index[word]
        one_hot_vector[index] = 1
        ohv.append([one_hot_vector])
    ret = np.array(ohv)
    return ret


# y_tmp = tf.one_hot(y_train, 3, on_value=1.0, off_value=0.0)

# y_tmp = one_hot_encoding(y_train, word_to_index)
y_tmp = convert_word_to_index(word_to_index, y_train)
# y_tmp = y_tmp.reshape(-1,1,3)

# print(y_train.shape)
print(y_tmp.shape)
# print(x_train[0])
# print(y_tmp[1000])

# %%
# np_data = np.load('./out/coco_train02.npy')
# print(np_data.shape)
# print(np_data[19])
# print(data[0])

# window = 4
# arr =[]
 
# model = lstm()
max_depth = 45
n_estimators = 25

x_train = x_train.reshape(-1, 4)

def random_forest():
    model = make_random_forest(max_depth=max_depth, n_estimators=n_estimators, random_state=0)
    print(x_train.shape, y_tmp.shape)
    model.fit(x_train, y_tmp)
    model_params = model.get_params()
    print(model_params)
    pickle.dump(model, open(f"./out/rf_{max_depth}_{n_estimators}.pkl", 'wb'))
    
    return model
    
def lstm():
    model = make_lstm()
    
    # model.compile(loss='binary_crossentropy',
                # optimizer='rmsprop',
                # metrics=['accuracy'])
    # model.compile(loss='mse', optimizer=Adam(0.01))

    print(model.summary())
    
    model.fit(x_train, y_tmp, batch_size=1 , epochs=40, verbose=1)

    return model

def gru():
    model = make_gru()
    
    model.fit(x_train, y_tmp, batch_size=1 , epochs=40, verbose=1)
    
    return model

# x_train=np.array(x_train) # (16,4,5,2) -> (16,4,10,1)

# x_train = x_train.reshape(-1, x_train.shape[1],x_train.shape[2], 1)

# print(x_train[0])
# print(y_tmp[0])

# model.fit(x_train, y_tmp, batch_size=1 , epochs=40, verbose=1)

# model.save('lstm_model.h5')
# model.save
# print("모델 저장 완려") #려~

# pred = model.predict(x_train) # 테스트 데이터 예측

# fig = plt.figure(facecolor='white')
# ax = fig.add_subplot(111)
# ax.plot(y_train, label='True')
# ax.plot(pred, label='Prediction')
# ax.legend()
# plt.show()

# model = random_forest()
# model = lstm()
model = gru()

# %%
# with open(delta_label, "r") as label_json:
    # label_tmp = json.load(label_json)[0]["pose"]
    
# model = keras.models.load_model('lstm_model.h5')
# lenght = len(x_train)
# faild = 0
# fail_list = []
# for i in tqdm(range(lenght), desc=f"오차 {faild/lenght}", mininterval=1):
#     x_tmp = [x_train[i]]
#     y_pred = model.predict(x_tmp)
#     if y_pred != y_tmp[i]:
#         faild += 1
#         print(f"실패 > {faild}/{i}")
#         fail_list.append(i)

# print(f"성공률 > {(1-faild/lenght)*100}")

def score_check():

    score = model.score(x_train, y_tmp)

    print(f"score > {score}")
    print(f"model accuracy score > {accuracy_score(y_tmp, model.predict(x_train))}")

    import time

    start = time.time()
    y_pred = model.predict([x_train[0]])
    # model.
    run_time = time.time() - start
    print(f"predict time > {run_time}")

    params = model.get_params()

    params["score"] = score
    params["pred_time"] = run_time
    # json_data = json.dumps(params)
    with open(f"./model/depth_{max_depth}_{n_estimators}.json", "w") as json_file:
        json.dump(params, json_file)
# print(f"실패한 갯수 > {faild/len(x_train)}")
# print(f"{x_train[0]}")


# y_pred = model.predict(x_tmp)
# print(f"예상치 > {y_pred}")
# print(f"실제값 > {y_tmp[0]}")

# score_check()

# %%



