import cv2

# Citim imaginea
img = cv2.imread("dog.jpg")

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive Threshold
# Folosim metoda de thresholding adaptiv pentru a obține o imagine binară. Aceasta va ajuta la detectarea contururilor în pasul următor.
adaptive = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,#dimensiunea zonei locale analizate.
    2#o valoare scăzută din calculul pragului.
)

# Găsim contururile
contours, hierarchy = cv2.findContours(
    adaptive,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contururi găsite:", len(contours))

# Copie pentru desenare
result = img.copy()

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignorăm contururile foarte mici
    if area < 100:
        continue

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

# Afișăm
cv2.imshow("Original", img)
cv2.imshow("Adaptive Threshold", adaptive)
cv2.imshow("Contururi", result)

cv2.waitKey(0)
cv2.destroyAllWindows()