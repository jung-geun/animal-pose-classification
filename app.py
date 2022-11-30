from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip, QDesktopWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QProgressBar, QMainWindow, QMessageBox
from PyQt5.QtGui import QFont, QIcon, QPixmap
# from PyQt5.QtCore import QCoreApplication, QBasicTimer, Qt
from PyQt5 import uic

form_class = uic.loadUiType("./app.ui")[0]


import sys
# import os


class MyApp(QMainWindow, form_class):
    pixmap1_path = ""
    pixmap2_path = ""
    siml_img_path = ""

    def __init__(self):
        QMainWindow.__init__(self)
        # 연결한 Ui를 준비한다.
        self.setupUi(self)
        # 연결한 UI 설정 변경
        self.initUI()
        # 화면을 보여준다.
        self.show()

    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setToolTip('This is a <b>Face unlock</b> widget')
        self.setWindowTitle('Face Project!!')
        # self.setWindowIcon(QIcon('deepface-icon.png'))

        # self.button_cmp.clicked.connect(self.compare_btn_clicked)
        # self.button_search.clicked.connect(self.img_siml)
        # self.reset.clicked.connect(self.reset_clecked)

        # self.setimg()
        # self.center()
        self.show()

    def Preprocessing(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def ref_FileLoad(self):
        img1_fname = QFileDialog.getOpenFileName(self)
        self.pixmap1_path = img1_fname[0]
        pixmap1 = QPixmap(img1_fname[0])
        pixmap1 = pixmap1.scaledToHeight(261)
        self.img_ref.setPixmap(pixmap1)

    def cmp_FileLoad(self):
        img2_fname = QFileDialog.getOpenFileName(self)
        self.pixmap2_path = img2_fname[0]
        pixmap2 = QPixmap(img2_fname[0])
        pixmap2 = pixmap2.scaledToHeight(261)
        self.img_cmp.setPixmap(pixmap2)

    def sim_FildLoad(self):
        img_sim_fname = QFileDialog.getOpenFileName(self)
        self.siml_img_path = img_sim_fname[0]
        pixmap3 = QPixmap(img_sim_fname[0])
        pixmap3 = pixmap3.scaledToHeight(231)
        self.img_main.setPixmap(pixmap3)

    def reset_clecked(self):
        pixmap = QPixmap('deepface-icon.png')
        pixmap = pixmap.scaledToHeight(261)
        self.img_cmp.setPixmap(pixmap)
        self.img_ref.setPixmap(pixmap)

    # def img_siml(self):
    #     df = DeepFace.find(img_path=self.siml_img_path, db_path="cmp_db")
    #
    #     count = 0
    #
    #     for value in df['identity']:
    #         pixmap4 = QPixmap(value)
    #         pixmap4 = pixmap4.scaledToHeight(231)
    #         if (count == 0):
    #             self.img_sim1.setPixmap(pixmap4)
    #         elif (count == 1):
    #             self.img_sim2.setPixmap(pixmap4)
    #         elif (count == 2):
    #             self.img_sim3.setPixmap(pixmap4)
    #         elif (count == 3):
    #             self.img_sim4.setPixmap(pixmap4)
    #
    #         count = count + 1
    #     if count == 0:
    #         QMessageBox.about(self, 'Error!!', '이미지 검색에 실패했습니다')

    # def compare_btn_clicked(self):
    #     if (self.pixmap1_path != "" and self.pixmap2_path != ""):
    #         result = DeepFace.verify(img1_path=self.pixmap1_path, img2_path=self.pixmap2_path)
    #         # QMessageBox.NoButton(self, )
    #         verified = result['verified']
    #
    #         self.suc.setText("일치율 : " + str(verified))
    #     else:
    #         QMessageBox.about(self, 'Warnnig!!', '이미지를 선택하시오')

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def setimg(self):
        self.sel_ref.clicked.connect(self.ref_FileLoad)
        self.sel_cmp.clicked.connect(self.cmp_FileLoad)
        self.sel_sim.clicked.connect(self.sim_FildLoad)

        pixmap1 = QPixmap('deepface-icon.png')
        pixmap1 = pixmap1.scaledToHeight(261)
        self.img_ref.setPixmap(pixmap1)

        pixmap2 = QPixmap('deepface-icon.png')
        pixmap2 = pixmap2.scaledToHeight(261)
        self.img_cmp.setPixmap(pixmap2)

        pixmap3 = QPixmap('deepface-icon.png')
        pixmap3 = pixmap3.scaledToHeight(231)
        self.img_main.setPixmap(pixmap3)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    app.exec_()
