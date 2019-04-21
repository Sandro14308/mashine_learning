#http://qaru.site/questions/2933/simple-digit-recognition-ocr-in-opencv-python

import cv2
import numpy as np
import KNN

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = KNN.KNearest()
model.train(samples,responses)