#Saving a Video#

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv.cv.CV_FOURCC(*'XVID')
out = cv.VideoWriter('testevideo.avi', fourcc, 20.0, (640,  480))
for t in range(400):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()

