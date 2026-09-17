import cv2
import numpy as np

img = cv2.imread("dog.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Threshold
_, threshold = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# Kernel = o mică matrice folosită pentru operația morfologică
kernel = np.ones((5, 5), np.uint8)
#eroded = cv2.erode(
#Erodarea este o operație morfologică care micșorează zonele albe dintr-o imagine binară. Aceasta poate fi utilă pentru a elimina zgomotul sau pentru a separa obiectele apropiate.
# Dilatare
# Dilatarea este o operație morfologică care mărește zonele albe dintr-o imagine binară. Aceasta poate fi utilă pentru a conecta obiecte apropiate sau pentru a evidenția caracteristicile unei imagini.
dilated = cv2.dilate(
    threshold,
    kernel,
    iterations=1
)

cv2.imshow("Threshold", threshold)
cv2.imshow("Dilated", dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()