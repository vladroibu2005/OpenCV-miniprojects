import cv2

# ==========================================
# 1. CITIM IMAGINEA
# ==========================================

img = cv2.imread("dog.jpg")

# ==========================================
# 2. GRAYSCALE
# ==========================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==========================================
# 3. BLUR
# ==========================================

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ==========================================
# 4. THRESHOLD
# ==========================================
#Folosim metoda de thresholding pentru a obține o imagine binară. Aceasta va ajuta la detectarea contururilor în pasul următor.
_, threshold = cv2.threshold(
    blur,
    50, #O să vezi că modificând un singur număr îi spui lui OpenCV ce consideră „alb” și ce consideră „negru”
    255,
    cv2.THRESH_BINARY
)

# ==========================================
# 5. GĂSIM CONTURURILE
# ==========================================
#Acest pas implică detectarea contururilor în imaginea binară obținută anterior. Contururile sunt curbe care unesc punctele cu aceeași intensitate, fiind utile pentru a identifica formele obiectelor din imagine.
contours, hierarchy = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contururi găsite:", len(contours))

# ==========================================
# 6. DESENĂM CONTURURILE
# ==========================================

result = img.copy()

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignorăm obiectele foarte mici
    if area < 100:
        continue

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

# ==========================================
# 7. AFIȘĂM
# ==========================================

cv2.imshow("Original", img)
cv2.imshow("Threshold", threshold)
cv2.imshow("Contururi", result)

cv2.waitKey(0)
cv2.destroyAllWindows()