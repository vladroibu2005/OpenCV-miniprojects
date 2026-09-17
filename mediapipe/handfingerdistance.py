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


image=cv2.imread("hand.jpg")
if image is None:
    print("Nu am putut încărca imaginea!")
    exit()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

result = detector.detect(mp_image)

if result.hand_landmarks:
    for hand_index, hand in enumerate(result.hand_landmarks):
        print(f"Mână {hand_index + 1}:")
        coordonate_puncte = {}
        for landmark_id, point in enumerate(hand):
            h,w,c=image.shape


            x=int(point.x * w)
            y=int(point.y * h)

            coordonate_puncte[landmark_id] = (x, y)

            print(
                f"Landmark {landmark_id}: "
                f"(x={x}, y={y})"
            )
            cv2.circle(
                image,                     
                (x, y),                    
                5,                         
                (0, 255, 0),                 
                cv2.FILLED                       
            )
#Test sa vad daca functioneaza corect si afiseaza coordonatele landmark-urilor            
#(coordonate_puncte)
(x4, y4) = coordonate_puncte[4]
(x8, y8) = coordonate_puncte[8]


distanta = ((x8 - x4) ** 2 + (y8 - y4) ** 2) ** 0.5
cv2.line(
    image,                     
    (x4, y4),                    
    (x8, y8),                                       
    (255, 0, 0),                 
    2
)
#Algorimul poate fi modificat pt a calcula distanta dintre orice alte degete,deoarece avem deja coordonatele tuturor landmark-urilor in coordonate_puncte sub forma de dictionar,iar pentru a calcula distanta dintre doua puncte
distanta_mijlocx=(x4 + x8) // 2
distanta_mijlocy=(y4 + y8) // 2
cv2.putText(
    image, 
    f"Distanta: {distanta:.2f} pixeli", 
    (distanta_mijlocx, distanta_mijlocy), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1, 
    (0, 0, 255), 
    2
)
cv2.imshow("Imaginea", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
