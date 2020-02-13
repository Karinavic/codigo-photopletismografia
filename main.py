import numpy as np
import cv2 as cv
import sys, argparse
import easygui

def ler_video(caminho=None):
    if caminho is None:
        cap = cv.VideoCapture(0) #ler a partir da webcam
    else:
        cap = cv.VideoCapture(caminho) #ler a partir do caminho(video local)
    try:
        while True:
            frame = cap.read()[1] #ler o quadro da imagem do vídeo
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #converte o quadro para tons de cinza 
            detecta_face(frame, gray)
            cv.imshow('Video', frame) #mostra a imagem capturada na janela

            #o trecho seguinte e apenas para parar o codigo e fechar a janela
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    except cv.error:
        sys.exit()

def detecta_face(frame, gray):
    face_cascade = 'haarcascade_frontalface_default.xml'
    faceCascade1 = cv.CascadeClassifier(face_cascade) #classificador para o rosto
    faces = faceCascade1.detectMultiScale(
        gray,
        minNeighbors=20,
        minSize=(30, 30),
	maxSize=(300,300)
    )
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)#retangulo da face
        detecta_olho(roi_gray, roi_color)

def detecta_olho(roi_gray, roi_color):
    eye_cascade = 'haarcascade_eye.xml'
    faceCascade2 = cv.CascadeClassifier(eye_cascade) #classificador para os olhos
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
    
    try: #lidando com possiveis erros- sem os dois olhos
        ph = ey+eh
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
    try: #lidando com possiveis erros-nao achar nenhum olho
            cv.rectangle(roi_color,(ex,ey),(pw,ph), (255,0,0),2)# retangulo dos olhos
    except:
        pass
    cv.rectangle(roi_color,(ex+int(0.5*ew),ey-eh),(pw-int(0.5*ew),ph-int(eh*1.4)), (255,0,255),2)# retangulo da testa

def gravar_video(nome):
    cap = cv.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    arquivo = "testevideo_" + nome + ".avi"
    out = cv.VideoWriter(arquivo, fourcc, 20.0, (640,  480))
    for __ in range(400):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        out.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release
    out.release
    cv.destroyAllWindows

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--captura", help="captura o video da webcam", action="store_true")
args = parser.parse_args()
if args.captura:
    gravar_video(inqput("Digite o nome para o video: "))
else:
    ler_video(easygui.fileopenbox())#chama a funcao ler video
