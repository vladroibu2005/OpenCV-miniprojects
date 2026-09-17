import cv2
import numpy as np

img = cv2.imread("imag.jpg")

# Convertim imaginea în grayscale și HSV
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Limităm valorile pentru a detecta verdele
# H: 40-80
# S: 40-255
# V: 40-255
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Păstrăm doar pixelii care se încadrează în interval
mask = cv2.inRange(hsv, lower_green, upper_green)

cv2.imshow("Imagine HSV", hsv)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()