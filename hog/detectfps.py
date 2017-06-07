from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


cap = cv2.VideoCapture(0)
 
if cap.isOpened() == False:
	print('Unable to open the camera')
else:
	start_time = time.time()
	for i in range(100):
		# timing mechanism for fps measurement
		
		ret,frame = cap.read()
		if ret==False:
			print('Unable to grab from the camera')
			break
 
		frame = imutils.resize(frame, width=min(400, frame.shape[1]))
	 
		# detect people in the image
		(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
			padding=(8, 8), scale=1.5)
	 
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
		filename = "profimgs/" + str(i) + ".jpg"
		cv2.imwrite(filename, frame)
		
	elapsed = time.time() - start_time
	fps = i / elapsed
	print "fps " + str(fps)
	print('Closing the camera')
 
cap.release()
cv2.destroyAllWindows()
print('bye bye!')
quit()
