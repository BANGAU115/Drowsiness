#!/usr/bin/python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import time
import numpy as np
import imutils
import os
from imutils.video import WebcamVideoStream
from tflite_runtime.interpreter import Interpreter
import cv2
from pygame import mixer
from datetime import datetime,date
import dlib
from imutils import face_utils
import shutil

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=0):
  """Returns a sorted array of classification results."""
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
def sound_alarm(path):
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()

detector = dlib.get_frontal_face_detector()
def getAllFaceBoundingBoxes( grayImg):

	try:
		return detector(grayImg, 0)
	except Exception as e:
		print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
		return []

def getLargestFaceBoundingBox( grayImg, skipMulti=False):

	faces = getAllFaceBoundingBoxes(grayImg)
	if (not skipMulti and len(faces) > 0) or len(faces) == 1:
		return max(faces, key=lambda rect: rect.width() * rect.height())
	else:
		return None
def main():


  COUNTER = 0
  MAX_FRAME = 4
  Flag_detect_face=0
  interpreter = Interpreter("model/drowsiness_v1.tflite")
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
    # Load model
  
    
  camera =WebcamVideoStream(src=1).start()
  time.sleep(5)
  cv2.namedWindow("Drowsiness");
  cv2.moveWindow('Drowsiness',550,0)
  sound_alarm("alarm_vn.mp3")
  mixer.music.pause()
  while True:
    try:
      image = camera.read()
      image = imutils.resize(image, width = 460)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      bb = getLargestFaceBoundingBox(gray)
      if bb:
        Flag_detect_face=1
        sx, sy, tx, ty =bb.left()-10,bb.top()-20,bb.right()+10,bb.bottom()+10
        roi = gray[sy:ty,sx:tx]
        roi = cv2.resize(roi,(width, height), interpolation = cv2.INTER_AREA)
        image_pre=prepare_image(roi)
        results = classify_image(interpreter, image_pre)
        if results<0.5:
          COUNTER += 1
          if COUNTER >= MAX_FRAME:
            if(mixer.music.get_busy()==False):
              filename = 'alarm_vn.mp3'
              sound_alarm(filename)
            try:
              total, used, free = shutil.disk_usage("/home/bang/Bangau/sdcard")
              total=total // (2**30)
              free=free // (2**30)
              if total>28 and free <1:
                list_of_files = os.listdir('sdcard/drowsiness/')
                full_path = ["sdcard/drowsiness/{0}".format(x) for x in list_of_files]
                if len([name for name in list_of_files]) >1:
                  oldest_file = min(full_path, key=os.path.getctime)
                  os.rmdir(oldest_file)
              elif((total>28)and(free>1)):
                date_object = date.today()
                time_object = datetime.now().time()
                time_object = time_object.strftime("%H-%M-%S")
                if not os.path.exists("sdcard/drowsiness/"+str(date_object)):
                #os.system('sudo mkdir sdcard/'+str(date_object))
                  os.mkdir("sdcard/drowsiness/"+str(date_object))
              #cv2.imwrite(os.path.join("/home/bang/Bangau/sdcard/2019-10-23" , 'waka.jpg'), roi)
                cv2.imwrite("sdcard/drowsiness/"+str(date_object)+"/" + str(time_object) + ".jpg", roi)
            except:{}
          #cv2.putText(image,"Drowsiness Alert",(50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv2.LINE_AA)
        else:
          COUNTER =0
          mixer.music.stop()
        cv2.rectangle(image, (sx, sy), (tx, ty), (255, 255, 255), 2)
      elif Flag_detect_face==1:
        COUNTER += 1
        if COUNTER >= (MAX_FRAME+3):
          if(mixer.music.get_busy()==False):
            filename = 'alarm_vn.mp3'
            sound_alarm(filename)
      cv2.imshow("Drowsiness", image)
      key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
      if key == ord("q"):
        break
    except:{}
  cv2.destroyAllWindows()
  camera.stop()

if __name__ == '__main__':
  main()
