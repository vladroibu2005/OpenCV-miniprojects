import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#Pentru a comunica cu ESP32
import serial
#Deschidem portul serial pentru a comunica cu ESP32
esp32 = serial.Serial("COM5", 115200)

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
print("Apasă tasta 'q' pentru a închide fereastra.")

x_min,y_min=100,30 #coordonatele nu sunt carteziene clasice,originea e la (0,0),stanga sus, x creste spre dreapta, y creste in jos
x_max,y_max=150,60
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
                cv2.rectangle(
                    frame, 
                    (x_min, y_min), 
                    (x_max, y_max), 
                    (255, 0, 0), 
                    -1, #ca sa fie umplut
                    )
                if x_min < x8 < x_max and y_min < y8 < y_max:
                    print("Buton apasat!")
                    esp32.write(b"1\n")  #Pt a trimite un semnal către ESP32 când butonul este apăsat
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), cv2.FILLED)
                    #cv2.putText(frame, "Apasat!", (x_min + 35, y_min + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    print("Buton neapasat!")
                    esp32.write(b"0\n")  #Pt a trimite un semnal către ESP32 când butonul nu este apăsat

            

    # Afișarea imaginii trebuie să se facă la fiecare iterație (cadru)
    cv2.imshow("Imaginea", frame)
    
    # waitKey(1) permite afișarea video-ului fluent (așteaptă 1 ms)
    # Ieșim din buclă dacă apăsăm tasta 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberăm camera la final
cap.release()
cv2.destroyAllWindows()