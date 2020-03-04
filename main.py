import numpy as np
import cv2 as cv
import sys, argparse
import easygui

def ler_video(caminho=None):#variavel vazia
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
    faceCascade1 = cv.CascadeClassifier(face_cascade) #classificador para a face
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
        detecta_olho(roi_gray, roi_color, w)

def detecta_olho(roi_gray, roi_color, w):
    global ex, ey, pw, ph,ew,eh
    eye_cascade = 'haarcascade_eye.xml'
    faceCascade2 = cv.CascadeClassifier(eye_cascade) #classificador para os olhos
    olhos = faceCascade2.detectMultiScale(
            roi_gray,
            minNeighbors=20,
            minSize=(10, 10),
            maxSize=(90,90)
        )
    cont=0
    for (ex,ey,ew,eh) in olhos:
        if cont==0:
            ext=ex
            eyt=ey
            ewt=ew
            eht=eh
            pht = eyt + eht # ponto inferior do olho + altura 
        cont+=1
    if cont == 2:
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
    elif cont == 1:
        ph = ey+eh
        ponto_medio = w/2 #ponto medio da largura da face
        if ex < ponto_medio:  # caso encontre so o olho esquerdo
            pw = w-ex #coordenada 
        else:
            pw = ex+ew
            l = w-(ex+ew) #largura entre canto direito do olho e a borda da face
            ex = l
    try: #lidando com possiveis erros-nao encontrando nenhum olho
        cv.rectangle(roi_color,(ex,ey),(pw,ph), (255,0,0),2)# retangulo dos olhos
        cv.rectangle(roi_color,(ex+int(0.5*ew),ey-eh),(pw-int(0.5*ew),ph-int(eh*1.4)), (255,0,255),2)# retangulo da testa
        roi_testa = roi_color[ey-eh:ph-int(eh*1.4), ex+int(0.5*ew):pw-int(0.5*ew)]
    except:
        pass
    
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

def processamento(roi_testa): 
    hsv = cv.cvtColor(roi_testa, cv.COLOR_BGR2HSV) # Converte BGR em HSV
    vetor_matiz = np.empty([0])
    for linha in range(0, hsv.shape[0]):#percorre linha do frame
        for coluna in range(0, hsv.shape[1]):#percorre linha do frame
            if hsv[linha,coluna,0] < 18: # definicao do limite da matiz=18
                vetor_matiz = np.append(vetor_matiz,hsv[linha,coluna,0])
    media_matiz = np.mean(vetor_matiz)
    return media_matiz

parser = argparse.ArgumentParser() #
parser.add_argument("-c", "--captura", help="captura o video da webcam", action="store_true")
args = parser.parse_args()
if args.captura:
    gravar_video(input("Digite o nome para o video: "))
else:
    ler_video(easygui.fileopenbox())#chama a funcao ler video na pasta do arquivo 
