import os
import video_Preprocessing as vp
import video_train as vt
import video_inference as vi
from threading import Thread

import time
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QLabel,
    QProgressBar,
    QMainWindow,
    QMessageBox,
    QToolTip,
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5 import uic

form_class = uic.loadUiType("./gui.ui")[0]


class MyApp(QMainWindow, form_class):
    model = "forest"
    flag = 0
    cam = 0

    def __init__(self):
        QMainWindow.__init__(self)
        # 연결한 Ui를 준비한다.
        self.setupUi(self)
        # 연결한 UI 설정 변경
        self.initUI()
        # 화면을 보여준다.
        self.show()

    def initUI(self):
        QToolTip.setFont(QFont("SansSerif", 10))
        self.setWindowTitle("yoga posture detection")

        self.tabWidget.setTabText(0, "Preprocessing")
        self.tabWidget.setTabText(1, "Training")
        self.tabWidget.setTabText(2, "Testing")

        self.btn_get_data.clicked.connect(self.dataClick)
        self.btn_pre.clicked.connect(self.preClick)

        self.btn_train.clicked.connect(self.trainClick)
        self.btn_model_1.clicked.connect(self.model1Click)
        self.btn_model_2.clicked.connect(self.model2Click)
        self.btn_model_3.clicked.connect(self.model3Click)
        self.btn_model_4.clicked.connect(self.model4Click)
        self.btn_model_5.clicked.connect(self.model5Click)

        self.video_start.clicked.connect(self.video_start_click)
        self.video_stop.clicked.connect(self.video_stop_click)

        self.radio_cam_0.clicked.connect(self.cam0Click)
        self.radio_cam_1.clicked.connect(self.cam1Click)

        self.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Message",
            "Are you sure to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.flag = 1
            event.accept()
        else:
            event.ignore()

    def dataClick(self):
        vp.get_csv()

    def preClick(self):
        print("버튼이 클릭되었습니다.")

        self.progressbar_pre.reset()
        result = vp.img_media(self=self)
        print(result)

    def trainClick(self):
        print("버튼이 클릭되었습니다.")

        self.progressbar_train.reset()
        result = vt.do_train(self=self)
        print(result)

    def model1Click(self):
        self.model = "svc"

    def model2Click(self):
        self.model = "tree"

    def model3Click(self):
        self.model = "forest"

    def model4Click(self):
        self.model = "sgd"

    def model5Click(self):
        self.model = "logistic"

    def video_start_click(self):
        print("video start")
        self.flag = 0
        Thread(target=vi.real_infereance(self=self)).start()

    def video_stop_click(self):
        self.flag = 1
        print("video stop")
        # self.label.clear()

    def cam0Click(self):
        self.cam = 0

    def cam1Click(self):
        self.cam = 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    app.exec_()
