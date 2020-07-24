#!/usr/bin/env python

"""
Fotopletismografia com Python e OpenCV.

Este código calcula a variabilidade da frequência cardíaca de um
indivíduo a partir do vídeo da sua face, utilizando técnicas de visão
computacional e processamento digital de imagens e vídeos.

"""

import sys
import argparse
import numpy as np
import cv2 as cv
import easygui


def main(caminho=None):  # variavel vazia
    """Lê o vídeo a partir do arquivo ou da webcam."""
    if caminho is None:
        cap = cv.VideoCapture(0)  # ler a partir da webcam
    else:
        cap = cv.VideoCapture(caminho)  # ler a partir do caminho
    try:
        raw_ppg = np.empty([0])
        while True:
            frame = cap.read()[1]  # ler o quadro da imagem do vídeo
            gray = cv.cvtColor(frame,  # converte o quadro para tons de cinza
                               cv.COLOR_BGR2GRAY)
            media_matiz = detecta_face(frame, gray)
            raw_ppg = np.append(raw_ppg, media_matiz)
            cv.imshow('Video', frame)  # mostra a imagem capturada na janela

            # o trecho seguinte e apenas para parar o codigo e fechar a janela
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        print(raw_ppg)
    except cv.error:
        print(raw_ppg)
        sys.exit()


def detecta_face(frame, gray):
    """Detecta a face no quadro."""
    face_cascade_name = 'haarcascade_frontalface_default.xml'
    face_cascade = cv.CascadeClassifier(face_cascade_name)  # classificador para face
    faces = face_cascade.detectMultiScale(gray, minNeighbors=20,
                                          minSize=(30, 30),
                                          maxSize=(300, 300))
    for (x_coord, y_coord, width, height) in faces:
        roi_gray = gray[y_coord : y_coord+height, x_coord : x_coord+width]
        roi_color = frame[y_coord : y_coord+height, x_coord : x_coord+width]
        cv.rectangle(frame, (x_coord, y_coord),  # retangulo da face
                     (x_coord + width, y_coord + height), (0, 255, 0), 4)
        return detecta_olho(roi_gray, roi_color, width)


def detecta_olho(roi_gray, roi_color, width):
    """Detecta o olho no quadro."""
    eye_cascade_name = 'haarcascade_eye.xml'
    eye_cascade = cv.CascadeClassifier(eye_cascade_name)  # classificador para os olhos
    olhos = eye_cascade.detectMultiScale(roi_gray, minNeighbors=20,
                                         minSize=(10, 10), maxSize=(90, 90))
    cont = 0
    eye_x = None
    eye_y = None
    eye_wd = None
    eye_hg = None
    eyes_wd = None
    eyes_hg = None

    for (eye_x, eye_y, eye_wd, eye_hg) in olhos:
        if cont == 0:
            eye_xt = eye_x
            eyt = eye_y
            ewt = eye_wd
            eht = eye_hg
            pht = eyt + eht  # ponto inferior do olho + altura
        cont += 1

    if cont == 2:  # quando os dois olhos são encontrados no quadro
        eyes_hg = eye_y + eye_hg
        if eye_xt < eye_x:
            eyes_wd = eye_x + eye_wd  # ponto superior do olho + largura
            eye_x = eye_xt
        else:
            eyes_wd = eye_xt + ewt
            eye_wd = ewt
        if pht > eyes_hg:
            eyes_hg = pht
        if eyt < eye_y:
            eye_y = eyt

    elif cont == 1:  # quando só um olho é encontrado no quadro
        eyes_hg = eye_y + eye_hg
        ponto_medio = width / 2  # ponto medio da largura da face
        if eye_x < ponto_medio:  # caso encontre so o olho esquerdo
            eyes_wd = width - eye_x  # coordenada
        else:
            eyes_wd = eye_x + eye_wd
            eye_x = width - (eye_x + eye_wd)

    try:  # lidando com possiveis erros-nao encontrando nenhum olho
        cv.rectangle(roi_color, (eye_x, eye_y), (eyes_wd, eyes_hg),
                     (255, 0, 0), 2)  # retangulo dos olhos
        cv.rectangle(roi_color, (eye_x + int(0.5*eye_wd), eye_y - eye_hg),
                     (eyes_wd - int(0.5*eye_wd), eyes_hg - int(eye_hg*1.4)),
                     (255, 0, 255), 2)
        roi_testa = roi_color[eye_y-eye_hg : eyes_hg-int(eye_hg*1.4),
                              eye_x+int(0.5*eye_wd) : eyes_wd-int(0.5*eye_wd)]
        return calcular_media_matiz(roi_testa)
    except:
        pass


def gravar_video(nome):
    """Grava o vídeo a partir da webcam."""
    cap = cv.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    arquivo = "testevideo_" + nome + ".avi"
    out = cv.VideoWriter(arquivo, fourcc, 20.0, (640, 480))
    for __ in range(400):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        out.write(frame)
        cv.imshow('frame', frame)()
    cap.release()
    out.release()
    cv.destroyAllWindows()


def calcular_media_matiz(roi_testa):
    """Calcula a média de matiz da região da testa."""
    hsv = cv.cvtColor(roi_testa, cv.COLOR_BGR2HSV)  # Converte BGR em HSV
    vetor_matiz = np.empty([0])
    for linha in range(0, hsv.shape[0]):  # percorre linha do frame
        for coluna in range(0, hsv.shape[1]):  # percorre linha do frame
            if hsv[linha, coluna, 0] < 18:  # definicao do limite da matiz=18
                vetor_matiz = np.append(vetor_matiz, hsv[linha, coluna, 0])
    media_matiz = np.mean(vetor_matiz)
    return media_matiz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--captura", help="captura o video da webcam",
                        action="store_true")  # argumento para capturar video
    args = parser.parse_args()
    if args.captura:
        gravar_video(input("Digite o nome para o video: "))
    else:
        main(easygui.fileopenbox())  # escolhe o video nas pastas locais
