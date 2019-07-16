import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

cap = cv.VideoCapture('testevideo.avi')  
while cap.isOpened():
    ret, frame = cap.read()    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        cont=1 ##contador
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if cont==1:
                ext=ex
                eyt=ey
                ewt=ew
                eht=eh
            cont+=1  
        cv.rectangle(roi_color,(int(ext+ewt*0.25),eyt),(int(ext+ewt*0.5),eyt+int((eyt-y)*0.6)),(0,0,255),2)
        #v.rectangle(roi_color,(int(ext+ewt*10),eyt),(int(ext+ewt*0.5),eyt+int((eyt-y)*1)),(0,0,255),2)


    cv.imshow('Deteccao',frame)
    if cv.waitKey(1)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()