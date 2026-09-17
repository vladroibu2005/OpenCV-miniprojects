import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

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
#Unghiul dintre 8,5,12-semnul pacii :))) (punctele A,B,C ale unui triunghi)
(x8, y8) = coordonate_puncte[8]
(x5, y5) = coordonate_puncte[5]
(x12, y12) = coordonate_puncte[12]
#Calculam cele 2 pante
panta1=(y8-y5)/(x8-x5)
panta2=(y12-y5)/(x12-x5)
#Calculam tangenta unghiului dintre cele 2 pante si apoi unghiul in radiani si in grade
tangenta_unghiului=(panta2-panta1)/(1+panta1*panta2)
unghi=math.atan(tangenta_unghiului)
unghi_grade=math.degrees(unghi)

cv2.putText(
    image,
    f"Unghi: {unghi_grade:.2f} grade",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)
cv2.imshow("Imaginea", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
