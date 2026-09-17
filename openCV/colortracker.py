import cv2
import numpy as np

img = cv2.imread("imag.jpg")

# -----------------------------
# 1. BGR -> HSV
# -----------------------------

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -----------------------------
# 2. Selectăm culoarea galbenă
# -----------------------------



#Pentru a detecta galbenul, putem folosi următoarele valori pentru H, S și V:,unde variabilele reprezinta limitele inferioare și superioare pentru fiecare canal.
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

mask = cv2.inRange(
    hsv,
    lower_yellow,
    upper_yellow
)

# -----------------------------
# 3. Curățăm puțin masca
# -----------------------------

kernel = np.ones((5, 5), np.uint8)


mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# -----------------------------
# 4. Căutăm contururile
# -----------------------------

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------------
# 5. Dacă am găsit obiecte
# -----------------------------

if contours:

    # Alegem cel mai mare contur
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest_contour)

    # Ignorăm obiectele foarte mici
    if area > 500:

        # Bounding box
        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        # Desenăm dreptunghiul
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Afișăm aria
        cv2.putText(
            img,
            f"Area: {int(area)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

# -----------------------------
# 6. Afișăm rezultatul
# -----------------------------

cv2.imshow("Imagine", img)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

