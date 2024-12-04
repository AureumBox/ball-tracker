import cv2
import numpy as np
import imutils
import time

# Rango de color para el verde (ajustar según sea necesario)
greenLower = (40, 108, 80)
greenUpper = (255, 255, 255)

# Inicializa la cámara
camera = cv2.VideoCapture(0)

# Parámetros de calibración
REAL_DIAMETER = 0.1  # Diámetro real de la pelota en metros (ajustar según el objeto)
FOCAL_LENGTH = 600  # Longitud focal en píxeles (ajustar según calibración)

# Calculo de distancia
def get_distance(diameter_in_pixels):
    if diameter_in_pixels > 0:
        distance = (REAL_DIAMETER * FOCAL_LENGTH) / diameter_in_pixels
        return distance
    return float('inf')  # Retorna infinito si el diámetro es cero o negativo

# Variables para controlar el tiempo de impresión
last_print_time = time.time()
print_interval = 2  # Intervalo de impresión en segundos

# Tamaño del cuadrado
min_square_size = 140 
max_square_size = 250 

while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear la máscara para el rango de color
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Obtener el centro de la imagen
    height, width = frame.shape[:2]
    camera_center = (width // 2, height // 2)

    # Dibujar el centro de la cámara
    cv2.circle(frame, camera_center, 5, (255, 0, 0), -1)  # Centro de la cámara en azul

    # Dibujar el cuadrado en el centro de la imagen
    top_left_min = (camera_center[0] - min_square_size // 2, camera_center[1] - min_square_size // 2)
    bottom_right_min = (camera_center[0] + min_square_size // 2, camera_center[1] + min_square_size // 2)
    cv2.rectangle(frame, top_left_min, bottom_right_min, (255, 255, 255), 2)  # Cuadrado pequeño en blanco

    top_left_max = (camera_center[0] - max_square_size // 2, camera_center[1] - max_square_size // 2)
    bottom_right_max = (camera_center[0] + max_square_size // 2, camera_center[1] + max_square_size // 2)
    cv2.rectangle(frame, top_left_max, bottom_right_max, (255, 255, 255), 2)  # Cuadrado grande en blanco

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            # Calcular el diámetro
            diameter_in_pixels = radius * 2

            # Dibujar el círculo alrededor de la pelota
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Dibujar el centro de la pelota
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Centro de la pelota en verde

            # Mostrar el diámetro en el marco
            cv2.putText(frame, f"Diameter: {int(diameter_in_pixels)} px", (int(x) - 50, int(y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Mostrar distancia
            distance = get_distance(diameter_in_pixels)
            cv2.putText(frame, f"Distance: {distance:.2f} m", (int(x) - 70, int(y) - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Determinar la posición de la pelota en relación al centro de la cámara
            if time.time() - last_print_time > print_interval:

                if x < camera_center[0] - 50 or x > camera_center[0] + 50:

                    if x < camera_center[0] - 50:  # Si la pelota está muy a la izquierda
                        print("Girar a izquierda")
                    if x > camera_center[0] + 50:  # Si la pelota está muy a la derecha
                        print("Girar a derecha")
                else:
                    if diameter_in_pixels < max_square_size and diameter_in_pixels > min_square_size:
                        print("Listo para responder al color")

                    if diameter_in_pixels > max_square_size:
                        print("Atrás")

                    if diameter_in_pixels < min_square_size:
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