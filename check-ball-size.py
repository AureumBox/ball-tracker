import cv2
import numpy as np
import imutils

# Cargar la imagen
image = cv2.imread('ball_size/4m.jpg')  # Cambia esto a la ruta de tu imagen
frame = imutils.resize(image, width=600)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Rango de color para el verde
greenLower = (68, 94, 68)
greenUpper = (225, 255, 255)

# Crear la máscara para el rango de color
mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# Encontrar contornos
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

if len(cnts) > 0:
    # Obtener el contorno más grande
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

# Mostrar los resultados
cv2.imshow("Frame", frame)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()