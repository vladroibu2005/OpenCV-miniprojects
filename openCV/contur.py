import cv2

img = cv2.imread("dog.jpg")
# Convertim imaginea în tonuri de gri
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Aplicăm detecția marginilor folosind algoritmul Canny
edges = cv2.Canny(gray, 100, 200)
# Afișăm imaginea cu marginile detectate
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
# Afișăm contururile detectate
result = img.copy()
# Desenăm contururile pe imaginea originală
cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imshow("Contururi", result)

cv2.waitKey(0)
cv2.destroyAllWindows()