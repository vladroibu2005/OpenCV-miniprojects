import cv2
import numpy as np

# ==========================================
# 1. CREĂM IMAGINEA
# ==========================================

img = np.zeros((600, 800, 3), dtype=np.uint8)

# Cele 4 colțuri ale foii deformate
points = np.array([
    [150, 100],
    [650, 100],
    [550, 500],
    [200, 450]
], np.int32)

# Desenăm foaia albă
cv2.fillPoly(
    img,
    [points],
    (255, 255, 255)
)

# ==========================================
# 2. TRANSFORMĂM ÎN GRAYSCALE
# ==========================================

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# ==========================================
# 3. DETECTĂM CONTURURILE
# ==========================================

contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# ==========================================
# 4. LUĂM CONTURUL CEL MAI MARE
# ==========================================

contour = max(
    contours,
    key=cv2.contourArea
)

# ==========================================
# 5. APROXIMĂM CONTURUL
# ==========================================

epsilon = 0.02 * cv2.arcLength(
    contour,
    True
)

approx = cv2.approxPolyDP(
    contour,
    epsilon,
    True
)

print("Număr de colțuri:", len(approx))

# ==========================================
# 6. VERIFICĂM DACĂ AVEM 4 COLȚURI
# ==========================================

if len(approx) == 4:

    # Extragem punctele
    pts = approx.reshape(4, 2)

    print("Colțurile găsite:")
    print(pts)

    # ======================================
    # 7. DESENĂM COLȚURILE
    # ======================================

    for i, point in enumerate(pts):

        x, y = point

        cv2.circle(
            img,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            img,
            str(i),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ======================================
    # 8. DEFINIM CUM VREM SĂ ARATE
    #    FOAIA DUPĂ TRANSFORMARE
    # ======================================

    width = 500
    height = 700

    destination = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # ======================================
    # 9. CALCULĂM TRANSFORMAREA
    # ======================================

    matrix = cv2.getPerspectiveTransform(
        np.float32(pts),
        destination
    )

    # ======================================
    # 10. APLICĂM TRANSFORMAREA
    # ======================================

    result = cv2.warpPerspective(
        img,
        matrix,
        (width, height)
    )

    # Afișăm rezultatul
    cv2.imshow(
        "Imagine initiala",
        img
    )

    cv2.imshow(
        "Perspectiva corectata",
        result
    )

else:

    print("Nu am găsit 4 colțuri!")

cv2.waitKey(0)
cv2.destroyAllWindows()