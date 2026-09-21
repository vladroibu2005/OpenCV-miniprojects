import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# MediaPipe Gesture Recognizer
# =========================

MODEL_PATH = "gesture_recognizer.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.GestureRecognizer.create_from_options(options)


# =========================
# Camera
# =========================

cap = cv2.VideoCapture(0)

print("Apasă tasta 'q' pentru a închide fereastra.")


# =========================
# Main loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Nu pot accesa camera!")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertim imaginea pentru MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Gesture recognition
    result = detector.recognize(mp_image)


    # =========================
    # Draw hand landmarks
    # =========================

    h, w, _ = frame.shape

    for hand in result.hand_landmarks:

        # Draw landmarks
        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw connections between landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for start, end in connections:

            x1 = int(hand[start].x * w)
            y1 = int(hand[start].y * h)

            x2 = int(hand[end].x * w)
            y2 = int(hand[end].y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # =========================
    # Display recognized gesture
    # =========================

    if result.gestures:

        gesture = result.gestures[0][0]

        gesture_name = gesture.category_name
        confidence = gesture.score

        text = f"{gesture_name} ({confidence:.2f})"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    # =========================
    # Display frame
    # =========================

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()