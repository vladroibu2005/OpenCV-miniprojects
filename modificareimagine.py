import cv2
img=cv2.imread("dog.jpg")
# Redimensionăm imaginea la 200x200 pixeli
small=cv2.resize(img,(200,200))
# Afișăm imaginea redimensionată
cv2.imshow("Imaginea mea",small)
cv2.waitKey(0)
cv2.destroyAllWindows()