import cv2
# Încărcăm imaginea-img e doar numele variabilei care conține imaginea
img = cv2.imread("dog.jpg")
# Afișăm imaginea
cv2.imshow("Imaginea mea", img)
# Așteptăm până când apăsăm o tastă
cv2.waitKey(0)
# Închidem fereastra
cv2.destroyAllWindows()