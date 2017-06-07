import numpy as np
import cv2
import time
import argparse
from imutils import paths

# initialize classifiers
fb_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
ub_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
lb_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

#  if --images <path> loop over the image paths
i=0
for imagePath in paths.list_images('../pos/'):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = cv2.imread(imagePath)
	#image = imutils.resize(image, width=min(400, image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	# draw the original bounding boxes
	fbs = fb_cascade.detectMultiScale(gray, 1.05, 5)
	for (x,y,w,h) in fbs:
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
	
	#~ ubs = ub_cascade.detectMultiScale(gray, 1.05, 5)
	#~ for (x,y,w,h) in ubs:
		#~ cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

	#~ lbs = lb_cascade.detectMultiScale(gray, 1.05, 5)
	#~ for (x,y,w,h) in lbs:
		#~ cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

	filename = "profimgs/" + str(i) + ".jpg"
	cv2.imwrite(filename, image)
	i += 1
