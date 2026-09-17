import cv2
import numpy as np

# Deschidem videoclipul
cap = cv2.VideoCapture("video1.mp4")

# Vom păstra aici pozițiile anterioare ale obiectului
trajectory = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # 1. HSV
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Modifică valorile dacă obiectul tău are altă culoare
    lower_color = np.array([35, 80, 50])
    upper_color = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # =========================
    # 2. Curățăm masca
    # =========================
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # =========================
    # 3. Căutăm obiectul
    # =========================
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        # Alegem cel mai mare obiect detectat
        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)

        if area > 500:

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centrul obiectului
            cx = x + w // 2
            cy = y + h // 2

            # =========================
            # 4. Adăugăm poziția în traiectorie
            # =========================
            trajectory.append((cx, cy))

            # =========================
            # 5. Desenăm obiectul
            # =========================
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                6,
                (0, 0, 255),
                -1
            )

            # Coordonatele
            cv2.putText(
                frame,
                f"({cx}, {cy})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # =========================
    # 6. Desenăm traiectoria
    # =========================
    for i in range(1, len(trajectory)):

        cv2.line(
            frame,
            trajectory[i - 1],
            trajectory[i],
            (255, 0, 0),
            2
        )

    # Afișăm
    cv2.imshow("Tracker + Traiectorie", frame)
    cv2.imshow("Mask", mask)

    # Q = ieșire
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()