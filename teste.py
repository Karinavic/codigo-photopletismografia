import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') # cria o classificador
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
img = cv.imread('lena.jpg') #ler a imagem
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #converter preto e branco

faces = face_cascade.detectMultiScale(gray, 1.3, 5  )
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray )

    #for (ex,ey,ew,eh) in eyes:
        #cv.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)

    cont=1 ##contador
    for (ex,ey,ew,eh) in eyes:
        if cont==1
            ext=ex
            eyt=ey
            ewt=ew
            eht=eh
        cont+=1
    xeye = ext
    yeye = eyt
    heighteye = ex +eh
    widtheye = ey + ew
    cv.rectangle(roi_color,(xeye,yeye),(heighteye,widtheye),(0,255,0),2)
    
    #cv.rectangle(roi_color,(int(ext+ewt*0.25),eyt),(int(ext+ewt*0.5),eyt+int((eyt-y)*0.6)),(0,0,255),2)# oficial
    #cv.rectangle(roi_color,(int(ext+ewt*0.25),eyt),(int (2*(ext+ewt*0.5)),eyt+int((eyt-y)*0.6)),(0,0,255),2)
    #cv.rectangle(roi_color,(int(ext+ewt*0.25),eyt),(int (2*(ext+ewt*0.5)),eyt+int((eyt-y)*0.6)),(0,0,255),2)
    print(y)
    print(eyt)
    print(int(y+eyt*0.7))
    cv.rectangle(roi_color,(int(xeye+widtheye*0.25),y),(int(yeye*0.5),int((yeye-y)*0.5)),(0,0,255),2)

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()