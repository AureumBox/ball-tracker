from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time

# Configuración de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Rango de color para el verde
greenLower = (68, 94, 68)
greenUpper = (255, 255, 255)

# Inicializa la cámara
if not args.get("video", False):
    camera = cv2.VideoCapture(1)
else:
    camera = cv2.VideoCapture(args["video"])

# Variables para controlar el tiempo de impresión
last_print_time = time.time()
print_interval = 5  # Intervalo de impresión en segundos

# calulo de distancia. A partir de (4, 95) y (1, 25)
def get_distance(diameter):
    meters = -0.04285714 * diameter + 5.07142857
    return meters

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear la máscara para el rango de color
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # Obtener el centro de la imagen
    height, width = frame.shape[:2]
    camera_center = (width // 2, height // 2)

    # Dibujar el centro de la cámara
    cv2.circle(frame, camera_center, 5, (255, 0, 0), -1)  # Centro de la cámara en azul

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            # Calcular el diámetro
            diameter = radius * 2

            # Dibujar el círculo alrededor de la pelota
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Dibujar el centro de la pelota
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Centro de la pelota en verde

            # Mostrar el diámetro en el marco
            cv2.putText(frame, f"Diameter: {int(diameter)} px", (int(x) - 50, int(y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Mostrar distancia
            cv2.putText(frame, f"Distance: {(get_distance(diameter))} m", (int(x) - 70, int(y) - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Determinar la posición de la pelota en relación al centro de la cámara
            if time.time() - last_print_time > print_interval:
                if x < camera_center[0] - 50:  # Si la pelota está muy a la izquierda
                    print("Girar a izquierda")
                elif x > camera_center[0] + 50:  # Si la pelota está muy a la derecha
                    print("Girar a derecha")
                else:  # Si está más o menos centrada
                    print("Adelante")
                last_print_time = time.time()

    # Mostrar los resultados
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()