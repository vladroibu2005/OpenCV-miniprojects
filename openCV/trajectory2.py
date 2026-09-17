import cv2
import numpy as np
import math

# Deschidem videoclipul
cap = cv2.VideoCapture("video1.mp4")

# Lista în care păstrăm traiectoria
trajectory = []

# Poziția obiectului în cadrul anterior
prev_cx = None
prev_cy = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # 1. Transformăm imaginea în HSV
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Culoarea obiectului
    lower_color = np.array([35, 80, 50])
    upper_color = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # =========================
    # 2. Curățăm masca
    # =========================
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

    # =========================
    # 3. Găsim obiectul
    # =========================
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        # Alegem cel mai mare contur
        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)

        if area > 500:

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centrul obiectului
            cx = x + w // 2
            cy = y + h // 2

            # Salvăm poziția
            trajectory.append((cx, cy))

            # =========================
            # 4. Calculăm mișcarea
            # =========================
            if prev_cx is not None:

                dx = cx - prev_cx
                dy = cy - prev_cy

                # Distanța parcursă între cele două cadre
                speed = math.sqrt(dx**2 + dy**2)

                # =========================
                # 5. Determinăm direcția
                # =========================

                if abs(dx) < 3 and abs(dy) < 3:
                    direction = "STATIONAR"

                elif abs(dx) > abs(dy):

                    if dx > 0:
                        direction = "DREAPTA"
                    else:
                        direction = "STANGA"

                else:

                    if dy > 0:
                        direction = "JOS"
                    else:
                        direction = "SUS"

                # Afișăm direcția
                cv2.putText(
                    frame,
                    direction,
                    (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # Afișăm viteza
                cv2.putText(
                    frame,
                    f"Viteza: {speed:.1f} px/frame",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # Actualizăm poziția precedentă
            prev_cx = cx
            prev_cy = cy

            # =========================
            # 6. Desenăm obiectul
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

            # Coordonate
            cv2.putText(
                frame,
                f"({cx}, {cy})",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # =========================
    # 7. Desenăm traiectoria
    # =========================

    for i in range(1, len(trajectory)):

        cv2.line(
            frame,
            trajectory[i - 1],
            trajectory[i],
            (255, 0, 0),
            2
        )

    # =========================
    # 8. Afișăm rezultatul
    # =========================

    cv2.imshow("Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()