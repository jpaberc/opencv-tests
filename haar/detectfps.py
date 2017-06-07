import numpy as np
import cv2
import time
import argparse
import imutils
from imutils import paths

fb_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

#for imagePath in paths.list_images('../pos'):
cap = cv2.VideoCapture(0)
 
if cap.isOpened() == False:
	print('Unable to open the camera')
else:
	#~ cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
	#~ cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
	start_time = time.time()
	for i in range(100):
		ret,frame = cap.read()
		if ret==False:
			print('Unable to grab from the camera')
			break
	 
		#image = cv2.imread(imagePath)
		#frame = imutils.resize(frame, width=min(400, frame.shape[1]))
	 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	 
		fbs = fb_cascade.detectMultiScale(gray, 1.5, 5)
		for (x,y,w,h) in fbs:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	
		filename = "profimgs/" + str(i) + ".jpg"
		cv2.imwrite(filename, frame)
	
	elapsed = time.time() - start_time
	fps = i / elapsed

	print "avg fps = " + str(fps)
