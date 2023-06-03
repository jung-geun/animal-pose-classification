import os

import json
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree

from cal_rad import cal_rad


def get_data():
    data = pd.read_json("pose.json")
    print(data.info())
    meta_x = []

    # x = data.iloc[:,:4].values
    # 인식 가능한 자료형으로 변환
    for row in data.iloc:
        # print(row["arm_left"])
        deg = [
            cal_rad(row["arm_left"])[0],
            cal_rad(row["arm_left"])[1],
            cal_rad(row["arm_right"])[0],
            cal_rad(row["arm_right"])[1],
            cal_rad(row["leg_left"])[0],
            cal_rad(row["leg_left"])[1],
            cal_rad(row["leg_right"])[0],
            cal_rad(row["leg_right"])[1],
        ]
        # deg.append(cal_rad(row["arm_right"]))
        # deg.append(cal_rad(row["leg_left"]))
        # deg.append(cal_rad(row["leg_right"]))

        meta_x.append(deg)

    x = np.array(meta_x)
    # print(x.shape)

    y = data.iloc[:, 4].values
    # print(y.shape)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.1
    )
    # print(x_train.shape)
    # print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, x_test, y_train, y_test


# get_data()


def make_model(sel="forest"):
    if sel == "svc":
        model = svm.SVC()
    elif sel == "tree":
        model = tree.DecisionTreeClassifier()
    elif sel == "forest":
        model = RandomForestClassifier()
    elif sel == "sgd":
        model = SGDClassifier()
    elif sel == "logistic":
        model = LogisticRegression()

    print(model)

    return model


# make_model()


def do_train(self=None):
    # 데이터 호출
    try:
        x_train, x_test, y_train, y_test = get_data()
    except:
        return -2
    # print(x_train.shape)
    # print(y_train.shape)

    self.progressbar_pre.setMaximum(len(x_train))

    try:
        model = make_model(sel=self.model)
        # gram_train = np.dot(x_train, x_train.T)
    except:
        return -3

    try:
        model.fit(x_train, y_train)

        print("학습 완료")

        scores = model.score(x_test, y_test)

        self.Text_train.appendPlainText(f"정확도 : {scores}")
        # gram_test = np.dot(x_test, x_train.T)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(accuracy)
        # print(scores)
    except:
        return -4

    try:
        if not os.path.exists("./model"):
            os.mkdir("./model")
    except:
        print("Error: 폴더 생성 실패")

    try:
        with open(f"./model/model_{self.model}.pkl", "wb") as f:
            pickle.dump(model, f)
            self.Text_train.appendPlainText(f"{self.model} 모델 저장 완료")
    # pickle.dump(model,open("model_linear.m","wb"))
    except:
        print("모델 저장 실패")
        return -1


# do_train()
