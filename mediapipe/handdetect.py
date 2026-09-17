import cv2
import mediapipe as mp

# ============================================================
# 1. Importăm componentele MediaPipe Tasks
# ============================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 2. Calea către modelul Hand Landmarker
# ============================================================

MODEL_PATH = "hand_landmarker.task"


# ============================================================
# 3. Configurăm modelul
# ============================================================
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,

    # Numărul maxim de mâini pe care vrem să le detectăm
    num_hands=2,

    # Pragul minim pentru detectarea unei mâini
    min_hand_detection_confidence=0.5,

    # Pragul minim pentru urmărirea mâinii
    min_tracking_confidence=0.5
)


# ============================================================
# 4. Creăm Hand Landmarker-ul
# ============================================================

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# 5. Citim imaginea
# ============================================================

image = cv2.imread("hand.jpg")


# Verificăm dacă imaginea a fost încărcată
if image is None:
    print("Eroare: nu am putut încărca hand.jpg")
    exit()


# ============================================================
# 6. OpenCV folosește BGR.
#    MediaPipe folosește RGB.
# ============================================================

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# 7. Transformăm imaginea OpenCV într-un obiect MediaPipe Image
# ============================================================

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# ============================================================
# 8. Detectăm mâinile
# ============================================================

result = detector.detect(mp_image)


# ============================================================
# 9. Verificăm dacă au fost detectate mâini
# ============================================================

if result.hand_landmarks:

    # Parcurgem fiecare mână detectată
    for hand_index, hand in enumerate(result.hand_landmarks):

        print(f"\nMâna {hand_index + 1}")

        # ----------------------------------------------------
        # Fiecare mână are 21 de landmark-uri
        # ----------------------------------------------------

        for landmark_id, point in enumerate(hand):

            # Coordonatele landmark-ului sunt normalizate
            # între 0 și 1.
            #
            # point.x → poziția pe orizontală
            # point.y → poziția pe verticală

            h, w, c = image.shape

            x = int(point.x * w)
            y = int(point.y * h)

            # Afișăm în terminal ID-ul și coordonatele
            print(
                f"Landmark {landmark_id}: "
                f"x={x}, y={y}"
            )

            # ------------------------------------------------
            # Desenăm punctul pe imagine
            # ------------------------------------------------

            cv2.circle(
                image,
                (x, y),
                8,
                (255, 0, 255),
                cv2.FILLED
            )


# ============================================================
# 10. Afișăm imaginea
# ============================================================

cv2.imshow("Hand Landmarker", image)


# Așteptăm apăsarea unei taste
cv2.waitKey(0)


# Închidem toate ferestrele OpenCV
cv2.destroyAllWindows()
