import numpy as np
import cv2 as cv
import sys

def ler_video(caminho=None):
    face_cascade = 'haarcascade_frontalface_default.xml'
    eye_cascade = 'haarcascade_eye.xml'
    faceCascade1 = cv2.CascadeClassifier(face_cascade) #classificador para o rosto
    faceCascade2 = cv2.CascadeClassifier(eye_cascade) #classificador para os olhos

    if caminho is None:
        cap = cv.VideoCapture(0) #ler a partir da webcam
    else:
        cap = cv.VideoCapture(caminho) #ler a partir do caminho(video local)
    try:
        while True:
            ret, frame = cap.read() #ler o quadro da imagem do vídeo
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #converte o quadro para tons de cinza 
            detecta_face(gray)
    except cv2.error:
        sys.exit()

def detecta_face(gray):
    faces = faceCascade1.detectMultiScale(
        gray,
        minNeighbors=20,
        minSize=(30, 30),
	maxSize=(300,300)
    )
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        detecta_olho()

def detecta_olho(roi_gray):
    olhos = faceCascade2.detectMultiScale(
            roi_gray,
            minNeighbors=20,
            minSize=(10, 10),
            maxSize=(90,90)
        )
    cont=1
    for (ex,ey,ew,eh) in olhos:
        if cont==1:
            ext=ex
            eyt=ey
            ewt=ew
            eht=eh
            pht = eyt + eht # ponto inferior do olho + altura 
        cont+=1
    ph = ey+eh
    try: #lidando com possiveis erros- sem os dois olhos
        if ext < ex:
            pw = ex + ew #ponto superior do olho + largura
            ex = ext
        else:
            pw = ext + ewt
            ew = ewt
        if pht > ph:
            ph = pht
        if eyt < ey:
            ey=eyt
    except NameError:
        pass
    