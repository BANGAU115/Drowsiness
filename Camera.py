import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from imutils import face_utils
from imutils.video import VideoStream
import imutils
from threading import Thread
import os
#import datetime
from datetime import datetime,date
from imutils.video import WebcamVideoStream
import shutil

path_sd="/home/bang/Bangau/sdcard"
path_sd_cam="/home/bang/Bangau/sdcard/camera/"
date_object = date.today()
time_object = datetime.now().time()
time_object = time_object.strftime("%H-%M-%S")
if not os.path.exists(path_sd_cam):
    os.mkdir(path_sd_cam)
if not os.path.exists(path_sd_cam+str(date_object)):
    os.mkdir(path_sd_cam+str(date_object))
out = cv2.VideoWriter(path_sd_cam+str(date_object)+'/'+str(time_object)+'.avi',cv2.VideoWriter_fourcc('X','2','6','4'),20, (352,288))
i=0
class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=3, parent=None):
        super().__init__(parent)
        self.camera = WebcamVideoStream(src=camera_port).start()

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        data = self.camera.read()
        data = imutils.resize(data, width = 460)
        try:
            self.image_data.emit(data)
        except:{}
    


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()

    def image_data_slot(self, image_data):
        global out
        global i
        try:
                total, used, free = shutil.disk_usage(path_sd)
                total=total // (2**30)
                free=free // (2**30)
                if total>28 and free <7:
                    list_of_files = os.listdir(path_sd_cam)
                    full_path = [path_sd_cam+"{0}".format(x) for x in list_of_files]
                    if len([name for name in list_of_files]) >1:
                        oldest_file = min(full_path, key=os.path.getctime)
                        os.rmdir(oldest_file)
                elif((total>28)and(free>7)):
                    image_save = cv2.resize(image_data,(352,288))
                    out.write(image_save)
                    i+=1
                    if(i>3000):
                        i=0
                        date_object = date.today()
                        time_object = datetime.now().time()
                        time_object = time_object.strftime("%H-%M-%S")
                        if not os.path.exists(path_sd_cam):
                            os.mkdir(path_sd_cam)
                        if not os.path.exists(path_sd_cam+str(date_object)):
                            os.mkdir(path_sd_cam+str(date_object))
                        out = cv2.VideoWriter(path_sd_cam+str(date_object)+'/'+str(time_object)+'.avi',cv2.VideoWriter_fourcc('X','2','6','4'),                                20, (352,288))
        except:{}

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()
    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_detection_widget = FaceDetectionWidget()

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()
		
        label = QtWidgets.QLabel()
        label.setPixmap(QtGui.QPixmap('/root/Bangau/BN.jpg'))
		#label.setFont(QtGui.QFont("Times", 16, QtGui.QFont.Bold))
        layout.addWidget(label)
  
        layout.addWidget(self.face_detection_widget)
        layout.setAlignment(Qt.AlignCenter)
        #self.run_button = QtWidgets.QPushButton('Start')
        #layout.addWidget(self.run_button)

        self.record_video.start_recording()
        self.setLayout(layout)


def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget()
    #main_window.move(490, 0)
    main_window.setGeometry(485, 0, 370, 400)
    main_window.setWindowTitle('Camera') 
    main_window.setWindowIcon(QtGui.QIcon('/root/Bangau/logo.jpg'))
    main_window.setCentralWidget(main_widget)
    
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
