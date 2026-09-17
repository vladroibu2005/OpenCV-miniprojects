import cv2

# Deschidem camera
cap = cv2.VideoCapture(0)

while True:

    # Citim un frame de la cameră
    ret, frame = cap.read()

    # Dacă nu am primit imagine, ieșim
    if not ret:
        print("Nu pot accesa camera!")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray, 100, 200)
    # Afișăm imaginea
    cv2.imshow("Camera mea", gray)
    

    # Apăsăm Q pentru a ieși
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Eliberăm camera
cap.release()

# Închidem ferestrele
cv2.destroyAllWindows()