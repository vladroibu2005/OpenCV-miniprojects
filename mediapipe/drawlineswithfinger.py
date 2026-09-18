import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)  

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
h, w, c = frame.shape

canvas = frame.copy()
canvas[:] = 0
print("Apasă tasta 'q' pentru a închide fereastra.")


punct_anterior = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Nu pot accesa camera!")
        break

    frame = cv2.flip(frame, 1)

    # Toată logica trebuie să fie în interiorul buclei while (indentată la dreapta)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_index, hand in enumerate(result.hand_landmarks):
            coordonate_puncte = {}
            h, w, c = frame.shape
            #print(f"Rezoluție: {w} x {h}")
            
            for landmark_id, point in enumerate(hand):
                x = int(point.x * w)
                y = int(point.y * h)
                
                coordonate_puncte[landmark_id] = (x, y)

                cv2.circle(
                    frame,                     
                    (x, y),                    
                    5,                         
                    (0, 255, 0),                 
                    cv2.FILLED                       
                )
            if 8 in coordonate_puncte:
                x8, y8 = coordonate_puncte[8]
                if punct_anterior is not None:
                    cv2.line(
                        canvas,
                        punct_anterior ,
                        (x8, y8), 
                        (0, 0, 255), 
                        2
                    )
                punct_anterior = (x8, y8)
                    
                    
            

    # Afișarea imaginii trebuie să se facă la fiecare iterație (cadru)
    frame = cv2.add(frame, canvas)
    cv2.imshow("Imaginea", frame)
    
    # waitKey(1) permite afișarea video-ului fluent (așteaptă 1 ms)
    # Ieșim din buclă dacă apăsăm tasta 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberăm camera la final
cap.release()
cv2.destroyAllWindows()