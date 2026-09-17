import cv2
img=cv2.imread("dog.jpg")
# Convertim imaginea în tonuri de gri
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Aplicăm detecția marginilor folosind algoritmul Canny
#cv2.Canny(imagine, threshold1, threshold2)
                 #   ↑           ↑
                  #  inferior    superior

edges = cv2.Canny(gray, 100, 100)
# Afișăm imaginea cu marginile detectate
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()