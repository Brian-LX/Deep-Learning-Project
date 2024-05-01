
import sys
from gui import Ui_MainWindow
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
from predict import predict

class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        self.cwd = os.getcwd()
        # self.cwd = os.environ['HOME']
        super(MyMainWindow, self).__init__(parent)
        self.center()

        self.file_list = []
        self.num = 0

        # palette = QPalette()
        # palette.setBrush(QPalette.Background, QBrush(QPixmap("./img/3.png")))
        # self.setPalette(palette)

        self.setupUi(self)
        # Select picture file
        self.btn_seletefile = self.seletefile
        self.btn_seletefile.setObjectName("btn_seletefile")
        self.btn_seletefile.setText("Select Picture")
        self.btn_seletefile.clicked.connect(self.select_image)
        # Select Picture folder
        self.btn_chooseDir = self.selectdir
        self.btn_chooseDir.setObjectName("btn_chooseDir")
        self.btn_chooseDir.setText("Select Picture")
        self.btn_chooseDir.clicked.connect(self.choose_dir)
        # Start recognition button
        self.btn_start = self.startlpr
        self.btn_start.setObjectName("start_train")
        self.btn_start.setText("SingleClassification")
        self.btn_start.clicked.connect(self.start)
        # Proceed to the next recognition of the images in the folder
        self.btn_nextimg = self.nextimg
        self.btn_nextimg.setObjectName("btn_nextimg")
        self.btn_nextimg.setText("Batch Classification")
        self.btn_nextimg.clicked.connect(self.next_img)
        # Exit button
        self.btn_exit = self.exit
        self.btn_exit.setObjectName("exit_system")
        self.btn_exit.setText("Exit system")
        # self.btn_exit.clicked.connect(self.onClick_Button())
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)

    def process(self, path):
        image, labels = predict(path)
        # print(self.lineEdit.displayText())
        # self.lineEdit.displayText()
        # image = image[:,:,[2,1,0]]
        # print(image,labels)
        self.lineEdit_3.setText(labels)
        # Convert the image to QImage
        temp_imgSrc = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3,
                             QImage.Format_RGB888)
        # Convert images to QPixmap for easy display
        pixmap_imgSrc = QtGui.QPixmap.fromImage(temp_imgSrc).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(pixmap_imgSrc)
        self.label_3.setScaledContents(True)
        QtWidgets.QApplication.processEvents()



if __name__== "__main__":
    app = QApplication(sys.argv)
    Mywin = MyMainWindow()
    Mywin.show()
    sys.exit(app.exec_())
