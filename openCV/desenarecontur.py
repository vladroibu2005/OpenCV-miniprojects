import cv2
import numpy as np

# Imagine neagră 500x500
img = np.zeros((500, 500), dtype=np.uint8)

# Pătrat alb
#cv2.rectangle(image, start_point, end_point, color, thickness)
#Thickness of rectangle border in pixels. Use -1 to fill the rectangle with color.
cv2.rectangle(img, (100, 100), (400, 400), 255, 10)

# Găsim contururile
contours, hierarchy = cv2.findContours(
    img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
# Afișăm contururile detectate
print("Contururi găsite:", len(contours))

# Copiem imaginea
result = img.copy()

# Desenăm conturul 	cv.drawContours(	image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]	)
cv2.drawContours(
    result,
    contours,
    -1,
    255,
    3
)
# Afișăm imaginea cu conturul
cv2.imshow("Imagine", result)

cv2.waitKey(0)
cv2.destroyAllWindows()