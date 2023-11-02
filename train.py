# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 디버그 메시지 끄기

import tensorflow as tf

# gpu 사용 확인
print(tf.test.gpu_device_name())

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

import json
import pickle

import matplotlib.pyplot as plt  # 데이터 시각화
import natsort
import numpy as np
from keras.layers import (
    LSTM,
    GRU,
    Conv1D,
    MaxPooling1D,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense,
)
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.auto import tqdm

from angle_out import out


body_parts = [
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

label_parts = [
    ["Nose", "MouthCorner", "LowerLip"],
    ["LowerLip", "Nose", "Forehead"],
    ["Nose", "Forehead", "Neck"],
    ["Forehead", "Neck", "TailStart"],
    ["Neck", "TailStart", "TailTip"],
    ["Forehead", "Neck", "RightArm"],
    ["Neck", "RightArm", "RightWrist"],
    ["Forehead", "Neck", "LeftArm"],
    ["Neck", "LeftArm", "LeftWrist"],
    ["Neck", "TailStart", "RightFemur"],
    ["Neck", "RightFemur", "RightAnkle"],
    ["Neck", "TailStart", "LeftFemur"],
    ["Neck", "LeftFemur", "LeftAnkle"],
]


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


def make_lstm(window=10):
    model = Sequential()
    model.add(LSTM(256, input_shape=(window, len(label_parts)), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(13, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )

    print(model.summary())

    return model


def make_gru(window=10):
    model = Sequential()
    model.add(
        GRU(
            256,
            input_shape=(window, len(label_parts)),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.3))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=False))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(13, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )

    print(model.summary())

    return model


def make_gru_old(window=10):
    model = Sequential()
    model.add(
        GRU(
            256,
            input_shape=(window, len(label_parts)),
            dropout=0.3,
            return_sequences=True,
        )
    )
    model.add(GRU(128, dropout=0.25, return_sequences=True))
    model.add(GRU(64, dropout=0.25, return_sequences=False))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(13, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )
    print(model.summary())
    return model


def make_cnn2d(window=10):
    model = Sequential()
    model.add(
        Conv2D(
            192,
            kernel_size=(6, 6),
            activation="relu",
            input_shape=(window, len(label_parts), 1),
        )
    )
    model.add(MaxPooling2D(pool_size=4))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(13, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )

    print(model.summary())

    return model


def make_cnn1d(window=10):
    model = Sequential()
    model.add(
        Conv1D(
            192,
            kernel_size=4,
            activation="relu",
            input_shape=(window, len(label_parts)),
        )
    )
    # model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(128, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(13, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )

    print(model.summary())

    return model


# %%
# 윈도우 사이즈에 맞추어서 저장할 공간
# 초당 5개 제한 2초동안 10개

label_path = "label"


def make_data(window=10):
    print("make data")
    print(f"label_path > {label_path}")
    out_dir = f"./label{window}_{len(label_parts)}"
    for list_ in tqdm(
        natsort.natsorted(os.listdir(label_path)), desc="label", leave=True, position=0
    ):
        if list_[:5] == "label":
            label_list = f"{label_path}/{list_}/json"
            # print(label_list)
            for l in tqdm(
                natsort.natsorted(os.listdir(label_list)),
                desc=list_[5:],
                leave=True,
                position=1,
            ):
                keypoints = []

                with open(f"{label_list}/{l}") as label_json:
                    label_tmp = json.load(label_json)
                    for annotation in label_tmp["annotations"]:
                        key = annotation["keypoints"]
                        for val in key:
                            # print(val)
                            if key[val] == None:
                                key[val] = {"x": 0, "y": 0}
                        keypoints.append(key)
                tmp = np.array(keypoints)
                meta_data = []
                for data_index in tmp:
                    data_tmp = []
                    for index in data_index.values():
                        data_tmp.append(index)
                    meta_data.append(np.array(data_tmp))

                angle_arr = []
                for m_d in meta_data:
                    angle = out(
                        inputs=m_d,
                        body_parts=body_parts,
                        label_parts=label_parts,
                    )
                    angle_arr.append(angle)
                np_tmp = []
                for index in range(window, len(angle_arr)):
                    tmp_wind = angle_arr[index - window : index]

                    np_tmp.append(tmp_wind)
                os.makedirs(f"{out_dir}/{list_[5:]}", exist_ok=True)
                file_name = l.split(".")[0]
                np_save = f"{out_dir}/{list_[5:]}/{file_name}.npy"
                np.save(np_save, np_tmp)


# %%


def load_data(window=10):
    print("load data")

    out_dir = f"./label{window}_{len(label_parts)}"

    if not os.path.isdir(out_dir):
        make_data(window)

    x_data = []
    y_data = []

    for label_list in tqdm(
        natsort.natsorted(os.listdir(out_dir)), desc="label", position=0
    ):
        # print(label_list)
        for np_list in tqdm(
            natsort.natsorted(os.listdir(f"{out_dir}/{label_list}")),
            desc=label_list,
            position=1,
        ):
            np_tmp = np.load(f"{out_dir}/{label_list}/{np_list}")
            for tmp in np_tmp:
                x_data.append(tmp)
                y_data.append(label_list)

    x_data = np.array(x_data) / 360.0
    y_data = np.array(y_data)

    if not os.path.exists("model/onehot_encoder.pkl"):
        encoder = OneHotEncoder(sparse=False)
        y_one_hot = encoder.fit_transform(y_data.reshape(-1, 1))
        pickle.dump(encoder, open("model/onehot_encoder.pkl", "wb"))
    else:
        encoder = pickle.load(open("model/onehot_encoder.pkl", "rb"))
        y_one_hot = encoder.transform(y_data.reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_one_hot, test_size=0.3, shuffle=True
    )

    return x_train, x_test, y_train, y_test


# %%


def score_check(model_name="LSTM", window=10, verbose=1):
    """
    학습 및 점수 체크
    """
    print("score check")
    try:
        model_name = model_name.upper()
        model_list = ["LSTM", "GRU", "CNN1D", "CNN2D"]
        if model_name not in model_list:
            raise ValueError("model_name must be LSTM, GRU, CNN1D, CNN2D")
    except ValueError as e:
        print(e)
        return

    x_train, x_test, y_train, y_test = load_data(window)

    verbose = verbose
    dir_name = f"model/{model_name}_{window}_{len(label_parts)}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=verbose,
        mode="auto",
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        f"{dir_name}/best_model.h5",
        monitor="val_loss",
        verbose=verbose,
        save_best_only=True,
    )
    callbacks = [early_stopping, model_checkpoint]

    try:
        if model_name == "LSTM":
            model = make_lstm(window)
        elif model_name == "GRU":
            model = make_gru(window)
        elif model_name == "CNN2D":
            model = make_cnn2d(window)
        elif model_name == "CNN1D":
            model = make_cnn1d(window)
            # x_train = x_train.reshape(-1, window * len(label_parts), 1)
            # x_test = x_test.reshape(-1, window * len(label_parts), 1)

        history = model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=200,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=(x_test, y_test),
        )
        model.save(f"{dir_name}/model.h5")

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        return
    except Exception as e:
        print(e)
        return

    finally:
        pickle.dump(history.history, open(f"{dir_name}/history.pkl", "wb"))

        fig, ax1 = plt.subplots()
        (line1,) = ax1.plot(
            history.history["accuracy"], label="train_acc", color="blue"
        )
        (line2,) = ax1.plot(
            history.history["val_accuracy"], label="val_acc", color="green"
        )
        ax1.set_ylabel("accuracy")
        ax2 = ax1.twinx()

        (line3,) = ax2.plot(history.history["loss"], label="train_loss", color="red")

        (line4,) = ax2.plot(
            history.history["val_loss"], label="val_loss", color="orange"
        )
        ax2.set_ylabel("loss")

        plt.title("model history")
        plt.xlabel("epochs")

        legend1 = plt.legend(handles=[line1, line2], loc="upper right")
        plt.gca().add_artist(legend1)
        legend2 = plt.legend(handles=[line3, line4], loc="lower right")
        plt.gca().add_artist(legend2)
        plt.savefig(f"{dir_name}/model_history.png")


for w in [10, 15, 20]:
    for model in ["LSTM", "GRU", "CNN1D"]:
        score_check(model_name=model, window=w)

# for w in [15]:
# score_check("lstm", window=w)
