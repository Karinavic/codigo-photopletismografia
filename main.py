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
    try:
        if caminho is None:
            cap = cv.VideoCapture(0)  # ler a partir da webcam
            if not cap.isOpened():
                raise IOError
        else:
            cap = cv.VideoCapture(caminho)  # ler a partir do caminho
            if not cap.isOpened():
                raise NameError
    except IOError:
        print("Impossível abrir a webcam, verifique a conexão.")
        sys.exit()
    except NameError:
        print("Erro na leitura, você checou se é um arquivo de vídeo?")
        sys.exit()
    raw_ppg = np.empty([0])
    while True:
        frame = cap.read()[1]  # ler o quadro da imagem do vídeo
        if frame is None:  # fim do vídeo
            break
        roi_gray, roi_color, width = detecta_face(frame)
        if roi_gray is not []:  # se a face foi detectada
            roi_testa = detecta_olho(roi_gray, roi_color, width)
        else:
            roi_testa = None
        if roi_testa is not None:  # se encontrou a região da testa
            media_matiz = calcular_media_matiz(roi_testa)
            raw_ppg = np.append(raw_ppg, media_matiz)
        cv.imshow('Video', frame)  # mostra a imagem capturada na janela

        # o trecho seguinte e apenas para parar o codigo e fechar a janela
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    print("Done!")


def detecta_face(frame):
    """Detecta a face no quadro."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
    if len(faces) == 0:
        return [], [], []
    else:
        return roi_gray, roi_color, width


def detecta_olho(roi_gray, roi_color, width):
    """Detecta o olho no quadro."""
    eye_cascade_name = 'haarcascade_eye.xml'
    eye_cascade = cv.CascadeClassifier(eye_cascade_name)  # classificador para os olhos
    olhos = eye_cascade.detectMultiScale(roi_gray, minNeighbors=20,
                                         minSize=(10, 10), maxSize=(90, 90))
    contador = 0  # conta a quantidade de olhos encontrados na face
    eye_x = None  # coordenada x de um dos olhos
    eye_y = None  # coordenada y de um dos olhos
    eye_wd = None  # largura de um dos olhos
    eye_hg = None  # altura de um dos olhos
    eyes_wd = None  # largura de ambos os olhos
    eyes_hg = None  # altura de ambos os olhos

    for (eye_x, eye_y, eye_wd, eye_hg) in olhos:
        if contador == 0:  # armazena os dados de um olho antes do próximo
            eye_x_temp = eye_x
            eye_y_temp = eye_y
            eye_width_temp = eye_wd
            eye_height_temp = eye_hg
            eye_height_bottom = eye_y_temp + eye_height_temp
        contador += 1

    if contador == 2:  # quando os dois olhos são encontrados no quadro
        if eye_x_temp < eye_x:
            eyes_wd = eye_x + eye_wd - eye_x_temp
            eye_x = eye_x_temp
        else:
            eyes_wd = eye_x_temp + eye_width_temp - eye_x
        if eye_height_bottom < eye_y + eye_hg:
            eye_height_bottom = eye_y + eye_hg
        if eye_y_temp < eye_y:
            eyes_hg = eye_height_bottom - eye_y_temp
            eye_y = eye_y_temp
        else:
            eyes_hg = eye_height_bottom - eye_y
    elif contador == 1:  # quando só um olho é encontrado no quadro
        eyes_hg = eye_hg
        ponto_medio = width / 2  # ponto medio da largura da face
        if eye_x > ponto_medio:  # caso encontre só o olho direito do quadro
            eye_x = width - (eye_x + eye_wd)
        eyes_wd = width - 2*eye_x

    if contador == 0:
        roi_testa = None
    else:
        testa_x = eye_x + int(0.5*eye_wd)
        testa_y = 0
        testa_w = eyes_wd - eye_wd
        testa_h = int(0.7*eyes_hg)
        cv.rectangle(roi_color, (eye_x, eye_y),  # retangulo dos olhos
                     (eye_x + eyes_wd, eye_y + eyes_hg), (255, 0, 0), 2)
        cv.rectangle(roi_color, (testa_x, testa_y),  # retângulo da testa
                     (testa_x + testa_w, testa_y + testa_h), (255, 0, 255), 2)
        roi_testa = roi_color[testa_y : testa_y+testa_h,
                              testa_x : testa_x+testa_w]
    return roi_testa


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
        for coluna in range(0, hsv.shape[1]):  # percorre coluna do frame
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
