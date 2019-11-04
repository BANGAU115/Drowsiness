import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

from tflite_runtime.interpreter import Interpreter
#from keras.models import model_from_json
from imutils import face_utils
from imutils.video import VideoStream
import imutils
from pygame import mixer 
import os
#import datetime
from datetime import datetime,date
#from keras.applications.imagenet_utils import preprocess_input
from imutils.video import WebcamVideoStream
import dlib
from imutils import face_utils
import shutil
filename = '/root/Bangau/alarm_vn.mp3'
path_sd="/home/bang/Bangau/sdcard"
path_sddr=path_sd+"/drowsiness/"
def sound_alarm(path):
    # play an alarm sound
	mixer.init()
	mixer.music.load(path)
	mixer.music.play()

def set_input_tensor(interpreter, image):
	tensor_index = interpreter.get_input_details()[0]['index']
	input_tensor = interpreter.tensor(tensor_index)()[0]
	input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=0):
	set_input_tensor(interpreter, image)
	interpreter.invoke()
	output_details = interpreter.get_output_details()[0]
	output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
	if output_details['dtype'] == np.uint8:
		scale, zero_point = output_details['quantization']
		output = scale * (output - zero_point)
  #ordered = np.argpartition(-output, top_k)
	return output
def prepare_image(image):
	img = image / 255.0
	img = np.expand_dims(img, axis=2)
	return img

def getAllFaceBoundingBoxes( grayImg,detector):

	try:
		return detector(grayImg, 0)
	except Exception as e:
		print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
	return []

def getLargestFaceBoundingBox( grayImg,detector, skipMulti=False):

	faces = getAllFaceBoundingBoxes(grayImg,detector)
	if (not skipMulti and len(faces) > 0) or len(faces) == 1:
		return max(faces, key=lambda rect: rect.width() * rect.height())
	else:
		return None
class RecordVideo(QtCore.QObject):
	image_data = QtCore.pyqtSignal(np.ndarray)

	def __init__(self, camera_port=1, parent=None):
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
	def __init__(self, model, parent=None):
		super().__init__(parent)
		self.model = model
		self.detector = dlib.get_frontal_face_detector()
		self.image = QtGui.QImage()
		self._red = (0, 0, 255)
		self._width = 2
		self._min_size = (30, 30)

	def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_image = cv2.equalizeHist(gray_image)
		faces = getLargestFaceBoundingBox(gray_image,self.detector)
		return faces,gray_image
	def image_data_slot(self, image_data):
		try:
			global COUNTER
			global MAX_FRAME
			global Flag_detect_face
			faces,gray_image = self.detect_faces(image_data)
			if faces:
				Flag_detect_face=1
				(startX, startY, endX, endY) =faces.left(),faces.top(),faces.right(),faces.bottom()
				roi = gray_image[startY:endY, startX:endX]
				shape = cv2.resize(roi,(100, 100))
				image_pre=prepare_image(shape)
				results = classify_image(interpreter, image_pre)
				print(results)
				if results<0.5:
					COUNTER+=1
					if COUNTER >= MAX_FRAME:
						if(mixer.music.get_busy()==False):
							sound_alarm(filename)
						try:
							total, used, free = shutil.disk_usage(path_sd)
							total=total // (2**30)
							free=free // (2**30)
							if total>28 and free <1:
								list_of_files = os.listdir(path_sddr)
								full_path = [path_sddr+"{0}".format(x) for x in list_of_files]
								if len([name for name in list_of_files]) >1:
									oldest_file = min(full_path, key=os.path.getctime)
									os.rmdir(oldest_file)
							elif((total>28)and(free>1)):
								date_object = date.today()
								time_object = datetime.now().time()
								time_object = time_object.strftime("%H-%M-%S")
								if not os.path.exists(path_sddr):
									os.mkdir(path_sddr)
								if not os.path.exists(path_sddr+str(date_object)):
									os.mkdir(path_sddr+str(date_object))
								cv2.imwrite(path_sddr+str(date_object)+"/" + str(time_object) + ".jpg", roi)
						except:{}
				else:
					COUNTER=0
					if(mixer.music.get_busy()==True):
						mixer.music.stop()
                
				cv2.rectangle(image_data,
                          (startX, startY),
                          (endX, endY),
                          self._red,
                          self._width)
			elif Flag_detect_face==1:
				COUNTER += 1
				if COUNTER >= (MAX_FRAME+3):
					if(mixer.music.get_busy()==False):
						sound_alarm(filename)
			
			self.image = self.get_qimage(image_data)
			if self.image.size() != self.size():
				self.setFixedSize(self.image.size())

			self.update()
		except:{}

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

class MainWidget(QtWidgets.QWidget):
	def __init__(self, model, parent=None):
		super().__init__(parent)
		self.face_detection_widget = FaceDetectionWidget(model)

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

		self.record_video.start_recording()
		self.setLayout(layout)

COUNTER = 0
MAX_FRAME = 4
Flag_detect_face=0
interpreter = Interpreter("/root/Bangau/drowsiness_v1.tflite")
interpreter.allocate_tensors()

sound_alarm(filename)
mixer.music.pause()

def main():
	app = QtWidgets.QApplication(sys.argv)

	main_window = QtWidgets.QMainWindow()
	main_widget = MainWidget(interpreter)
    #main_window.move(0, 0)
	main_window.setGeometry(0, 0, 480, 400)
	main_window.setWindowTitle('Drowsiness') 
	main_window.setWindowIcon(QtGui.QIcon('/root/Bangau/logo.jpg'))
	main_window.setCentralWidget(main_widget)
    
	main_window.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()
