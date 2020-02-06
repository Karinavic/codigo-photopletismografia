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
    except cv2.error:
        sys.exit()