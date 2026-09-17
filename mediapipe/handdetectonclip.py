import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# 1. Configurare model și detector
# ============================================================
MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# ============================================================
# 2. Pornim camera web
# ============================================================
cap = cv2.VideoCapture(0)

print("Apasă tasta 'q' pentru a închide fereastra.")

# Bucla principală - rulează continuu până la apăsarea tastei 'q'
while True:
    
    # Citim un cadru (o imagine) de la cameră
    ret, frame = cap.read()

    # Dacă a apărut o problemă cu camera, oprim bucla
    if not ret:
        print("Nu pot accesa camera!")
        break

    # (OPȚIONAL) Întoarcem imaginea în oglindă pentru a fi mai naturală
    frame = cv2.flip(frame, 1)

    # ============================================================
    # 3. Procesăm imaginea
    # ============================================================
    # Convertim culorile pentru MediaPipe
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Detectăm mâinile în cadrul curent
    result = detector.detect(mp_image)

    # ============================================================
    # 4. Desenăm pe imagine
    # ============================================================
    if result.hand_landmarks:
        for hand_index, hand in enumerate(result.hand_landmarks):
            for landmark_id, point in enumerate(hand):
                
                # Aici folosim 'frame.shape' în loc de 'image.shape'
                h, w, c = frame.shape
                
                x = int(point.x * w)
                y = int(point.y * h)

                # Desenăm punctul pe 'frame'
                cv2.circle(
                    frame, 
                    (x, y), 
                    6, # Am făcut cercul puțin mai mic pentru claritate
                    (255, 0, 255), 
                    cv2.FILLED
                )
                
                # Desenăm textul (ID-ul) pe 'frame'
                cv2.putText(
                    frame,                      
                    str(landmark_id),           
                    (x + 10, y + 10),           
                    cv2.FONT_HERSHEY_SIMPLEX,   
                    0.5,                        
                    (0, 255, 0),                
                    1                           
                )
    # Harta conexiunilor: ce punct se leagă cu ce punct
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # Degetul mare
        (0, 5), (5, 6), (6, 7), (7, 8),         # Arătător
        (5, 9), (9, 10), (10, 11), (11, 12),    # Mijlociu
        (9, 13), (13, 14), (14, 15), (15, 16),  # Inelar
        (13, 17), (17, 18), (18, 19), (19, 20), # Degetul mic
        (0, 17)                                 # Baza palmei (încheietura spre degetul mic)
    ]

    # ============================================================
    # 4. Desenăm pe imagine
    # ============================================================
    if result.hand_landmarks:
        for hand_index, hand in enumerate(result.hand_landmarks):
            
            # Creăm un dicționar pentru a memora coordonatele calculate (x, y)
            # Format: { id_punct: (x, y) }
            coordonate_puncte = {}

            # --- PASUL 1: Calculăm și desenăm punctele ---
            for landmark_id, point in enumerate(hand):
                h, w, c = frame.shape
                
                x = int(point.x * w)
                y = int(point.y * h)

                # Salvăm coordonatele acestui punct în dicționar ca să le folosim mai târziu la linii
                coordonate_puncte[landmark_id] = (x, y)

                # Desenăm punctul (cercul)
                cv2.circle(frame, (x, y), 6, (255, 0, 255), cv2.FILLED)
                
                # Desenăm textul (ID-ul)
                cv2.putText(
                    frame, str(landmark_id), (x + 10, y + 10),           
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1                           
                )

            # --- PASUL 2: Desenăm liniile (scheletul / stick-urile) ---
            # Parcurgem lista cu perechile de puncte definite mai sus
            for start_id, end_id in HAND_CONNECTIONS:
                
                # Extragem coordonatele (x, y) pentru capetele liniei
                punct_start = coordonate_puncte[start_id]
                punct_end = coordonate_puncte[end_id]

                # Tragem linia pe 'frame' între cele două puncte
                cv2.line(
                    frame, 
                    punct_start,    # De unde începe linia
                    punct_end,      # Unde se termină linia
                    (0, 255, 255),  # Culoarea liniei în BGR (aici e Galben)
                    2               # Grosimea liniei
                )

    # ============================================================
    # 5. Afișăm cadrul live și setăm butonul de ieșire
    # ============================================================
    # ... (restul codului de dedesubt rămâne la fel) ...
    # ============================================================
    # 5. Afișăm cadrul live și setăm butonul de ieșire
    # ============================================================
    # Afișăm imaginea procesată
    cv2.imshow("Hand Landmarker - Live", frame)

    # cv2.waitKey(1) așteaptă 1 milisecundă. Dacă în acel timp apeși 'q', bucla se rupe.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Închidem camera...")
        break

# ============================================================
# 6. Curățenie (Eliberăm resursele)
# ============================================================
# Oprește accesul la camera web
cap.release()

# Închide toate ferestrele OpenCV
cv2.destroyAllWindows()