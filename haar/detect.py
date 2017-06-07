import numpy as np
import cv2
import time
import argparse
import imutils
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False, help="path to images directory")
ap.add_argument("-v", "--video", required=False, help="capture video feed")
args = vars(ap.parse_args())

# initialize classifiers
fb_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

#  if --images <path> loop over the image paths
if args["images"]:
	for imagePath in paths.list_images(args["images"]):
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	 
		# detect people in the image
		fbs = fb_cascade.detectMultiScale(gray, 1.1, 5)
	 
		# draw the original bounding boxes
		for (x, y, w, h) in fbs:
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	 
		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes".format(
			filename, len(fbs)))
	 
		# show the output images
		cv2.imshow("Before NMS", image)
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
			print "{0:0.2f}".format(fps)
			oldclock = time.clock()
			
			ret,frame = cap.read()
			if ret==False:
				print('Unable to grab from the camera')
				break
	 
	 		#image = cv2.imread(imagePath)
			#frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		 
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		 
			fbs = fb_cascade.detectMultiScale(gray, 1.01, 5)
			for (x,y,w,h) in fbs:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			
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

#img = cv2.imread('sachin.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x,y,w,h) in faces:
#    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
