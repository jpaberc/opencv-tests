from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#  if --images <path> loop over the image paths
i = 0
for imagePath in paths.list_images('../pos/'):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	filename = "profimgs/" + str(i) + ".jpg"
	cv2.imwrite(filename, image)
	i += 1

#else:
#~ cap = cv2.VideoCapture(0)
#~ 
#~ if cap.isOpened() == False:
	#~ print('Unable to open the camera')
#~ else:
	#~ cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
	#~ cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
	#~ for i in range(100):
		#~ ret,frame = cap.read()
		#~ if ret==False:
			#~ print('Unable to grab from the camera')
			#~ break
 #~ 
		#~ frame = imutils.resize(frame, width=min(400, frame.shape[1]))
	 #~ 
		#~ # detect people in the image
		#~ (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
			#~ padding=(8, 8), scale=1.3)
	 #~ 
		#~ # apply non-maxima suppression to the bounding boxes using a
		#~ # fairly large overlap threshold to try to maintain overlapping
		#~ # boxes that are still people
		#~ rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		#~ pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	 #~ 
		#~ # draw the final bounding boxes
		#~ for (xA, yA, xB, yB) in pick:
			#~ cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	 #~ 
		#~ filename = "profimgs/frame_" + str(i) + ".jpg"
		#~ cv2.imwrite(filename, frame)
 #~ 
#~ cap.release()
#~ cv2.destroyAllWindows()
#~ quit()

