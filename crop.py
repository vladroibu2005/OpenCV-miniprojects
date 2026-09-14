import cv2
img = cv2.imread("dog.jpg")
# Tăiem imaginea 
crop=img[100:400, 200:600]
# Afișăm imaginea tăiată
cv2.imshow("Imaginea mea",crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
