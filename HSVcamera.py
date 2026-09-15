import cv2
import numpy as np

# Deschidem camera
cap = cv2.VideoCapture(0)

while True:
    #Citim un frame de la cameră
    ret, frame = cap.read()
    # Dacă nu am primit imagine, ieșim
    if not ret:
        print("Nu pot accesa camera!")
        break
    
    # Convertim imaginea în spațiul de culoare HSV din BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("Camera mea", frame)
    cv2.imshow("Camera mea HSV", hsv)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()

