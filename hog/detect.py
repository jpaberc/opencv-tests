# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False, help="path to images directory")
ap.add_argument("-v", "--video", required=False, help="capture video feed")
args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#  if --images <path> loop over the image paths
if args["images"]:
	for imagePath in paths.list_images(args["images"]):
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		orig = image.copy()
	 
		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)
	 
		# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	 
		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	 
		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	 
		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes, {} after suppression".format(
			filename, len(rects), len(pick)))
	 
		# show the output images
		cv2.imshow("Before NMS", orig)
		cv2.imshow("After NMS", image)
		cv2.waitKey(0)

else:
	cap = cv2.VideoCapture(int(args["video"]))
 
	if cap.isOpened() == False:
		print('Unable to open the camera')
	else:
		print('Start grabbing, press a key on Live window to terminate')
		cv2.namedWindow('Live');
		cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
		oldclock = time.clock()
		while( cap.isOpened() ):
			# timing mechanism for fps measurement
			newclock = time.clock()
			elapsed = newclock - oldclock
			fps = 1 / elapsed
			print("{0:0.2f}".format(fps))
			oldclock = time.clock()
			
			ret,frame = cap.read()
			if ret==False:
				print('Unable to grab from the camera')
				break
	 
			frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		 
			# detect people in the image
			(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
				padding=(8, 8), scale=1.02)
		 
			# apply non-maxima suppression to the bounding boxes using a
			# fairly large overlap threshold to try to maintain overlapping
			# boxes that are still people
			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
			pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		 
			# draw the final bounding boxes
			for (xA, yA, xB, yB) in pick:
				cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
		 
			# show some information on the number of bounding boxes
			#filename = imagePath[imagePath.rfind("/") + 1:]
			#print("[INFO] {}: {} original boxes, {} after suppression".format(
			#	filename, len(rects), len(pick)))
	 
			cv2.imshow('Live',frame)
			#cv2.waitKey(0);
			key = cv2.waitKey(5)
			if key==255: key=-1 #Solve bug in 3.2.0
			if key >= 0:
				break
		print('Closing the camera')
	 
	cap.release()
	cv2.destroyAllWindows()
	print('bye bye!')
	quit()

