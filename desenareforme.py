import cv2
import numpy as np
# Imagine neagră 500x500
img = np.zeros((500, 500), dtype=np.uint8)

# Pătrat
cv2.rectangle(img, (50, 50), (100, 100), 255, 10)

# Cerc
cv2.circle(img, (300, 300), 25, 255, 10)

# Triunghi-OpenCV are funcții de desen pentru anumite forme de bază, dar pentru un triunghi îl construiești din 3 linii/puncte.
#Cele trei coordonate sunt cele trei vârfuri.
#Definim un triunghi folosind un array de puncte
triangle = np.array([
    [200, 200],
    [250, 300],
    [150, 300]
], np.int32)
#Il desenăm folosind polylines. Parametrul True indică faptul că triunghiul este închis.
cv2.polylines(img, [triangle], True, 255, 10)

# Găsim contururile
contours, hierarchy = cv2.findContours(
    img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contururi găsite:", len(contours))

result = img.copy()

# Desenăm contururile detectate
cv2.drawContours(
    result,
    contours,
    -1,
    255,
    3
)

cv2.imshow("Imagine", result)

cv2.waitKey(0)
cv2.destroyAllWindows()