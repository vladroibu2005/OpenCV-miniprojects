import cv2
import numpy as np
cap = cv2.VideoCapture("video1.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Verde
    lower_green = np.array([35, 80, 50])
    upper_green = np.array([85, 255, 255])

    # Masca
    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Curățăm masca
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

    # Găsim contururile
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        # Cel mai mare obiect
        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(contour)

        if area > 500:

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centrul
            cx = x + w // 2
            cy = y + h // 2

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # Centrul
            cv2.circle(
                frame,
                (cx, cy),
                7,
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

    cv2.imshow("Tracker", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()