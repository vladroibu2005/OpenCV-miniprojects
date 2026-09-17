import cv2
import numpy as np

# ==========================================
# 1. CREĂM IMAGINEA
# ==========================================

# Imagine neagră 500x500
img = np.zeros((500, 500), dtype=np.uint8)

# Dreptunghi
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# Cerc
cv2.circle(img, (350, 100), 50, 255, -1)

# Triunghi
triangle = np.array([
    [250, 250],
    [350, 400],
    [150, 400]
], np.int32)
#setăm triunghiul ca fiind închis
cv2.fillPoly(img, [triangle], 255)


# ==========================================
# 2. GĂSIM CONTURURILE
# ==========================================
#Găsim contururile folosind cv2.findContours. Această funcție returnează o listă de contururi și o ierarhie. Contururile sunt reprezentate ca o listă de puncte care definesc forma obiectului.
contours, hierarchy = cv2.findContours(
    img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contururi găsite:", len(contours))


# ==========================================
# 3. ANALIZĂM FIECARE CONTUR
# ==========================================

# Facem o copie color pentru a putea
# desena contururile cu verde
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#Pentru fiecare contur găsit, vom calcula perimetrul, vom simplifica conturul, vom număra colțurile și vom determina forma. Apoi, vom desena conturul și vom scrie numele formei pe imagine.
for contour in contours:

    # --------------------------------------
    # Calculăm perimetrul
    # --------------------------------------

    perimeter = cv2.arcLength(contour, True)

    # --------------------------------------
    # Simplificăm conturul
    # --------------------------------------
#Aproximarea conturului se face folosind funcția cv2.approxPolyDP, care reduce numărul de puncte din contur, păstrând forma generală. Parametrul 0.02 * perimeter specifică cât de mult să fie aproximat conturul.
    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    # --------------------------------------
    # Numărăm colțurile
    # --------------------------------------

    corners = len(approx)

    print("Colțuri:", corners)

    # --------------------------------------
    # Stabilim forma
    # --------------------------------------

    if corners == 3:
        shape = "Triunghi"

    elif corners == 4:
        shape = "Dreptunghi"

    elif corners == 5:
        shape = "Pentagon"

    else:
        shape = "Cerc"

    print("Forma:", shape)

    # --------------------------------------
    # Desenăm conturul
    # --------------------------------------

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    # --------------------------------------
    # Găsim poziția unde vom scrie textul
    # --------------------------------------
#Găsim un dreptunghi care încadrează conturul folosind cv2.boundingRect. Aceasta ne oferă coordonatele x și y ale colțului din stânga sus, precum și lățimea (w) și înălțimea (h) dreptunghiului. Vom folosi aceste coordonate pentru a poziționa textul deasupra formei.
    x, y, w, h = cv2.boundingRect(contour)

    # --------------------------------------
    # Scriem numele formei
    # --------------------------------------
#Scriem numele formei pe imagine folosind cv2.putText. Parametrii includ imaginea, textul, poziția, fontul, dimensiunea fontului, culoarea și grosimea textului.
    cv2.putText(
        result,
        shape,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ==========================================
# 4. AFIȘĂM REZULTATUL
# ==========================================

cv2.imshow("Shape Detection", result)

cv2.waitKey(0)
cv2.destroyAllWindows()