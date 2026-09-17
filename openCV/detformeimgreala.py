import cv2

# ==========================================
# 1. CITIM IMAGINEA
# ==========================================

img = cv2.imread("dog.jpg")

# ==========================================
# 2. TRANSFORMĂM ÎN GRAYSCALE
# ==========================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==========================================
# 3. REDUCEM ZGOMOTUL
# ==========================================

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ==========================================
# 4. DETECTĂM MUCHIILE
# ==========================================

edges = cv2.Canny(blur, 100, 200)

# ==========================================
# 5. GĂSIM CONTURURILE
# ==========================================

contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contururi găsite:", len(contours))

# ==========================================
# 6. FACEM O COPIE PENTRU REZULTAT
# ==========================================

result = img.copy()

# ==========================================
# 7. ANALIZĂM FIECARE CONTUR
# ==========================================

for contour in contours:

    # Calculăm perimetrul
    perimeter = cv2.arcLength(contour, True)

    # Simplificăm conturul
    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    # Numărăm punctele / colțurile
    corners = len(approx)

    # Desenăm conturul
    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # Afișăm numărul de puncte
    x, y, w, h = cv2.boundingRect(contour)

    cv2.putText(
        result,
        str(corners),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

# ==========================================
# 8. AFIȘĂM REZULTATELE
# ==========================================

cv2.imshow("Imagine originala", img)
cv2.imshow("Muchii", edges)
cv2.imshow("Contururi", result)

cv2.waitKey(0)
cv2.destroyAllWindows()