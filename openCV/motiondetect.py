import cv2

cap = cv2.VideoCapture("video1.mp4")

# Primul frame
ret, prev_frame = cap.read()

if not ret:
    print("Nu pot deschide videoclipul!")
    exit()

# Transformăm primul frame în grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:

    # Citim următorul frame
    ret, frame = cap.read()

    if not ret:
        break

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Diferența dintre frame-ul anterior și cel actual
    diff = cv2.absdiff(prev_gray, gray)

    # Transformăm diferența în alb/negru
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Curățăm zgomotul
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # Găsim zonele în care există diferențe
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Procesăm fiecare zonă detectată
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignorăm schimbările foarte mici
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Desenăm zona în care s-a detectat mișcare
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "MISCARE",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Afișăm
    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Diferenta", thresh)

    # Frame-ul actual devine frame-ul anterior
    prev_gray = gray

    # Q pentru iesire
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()