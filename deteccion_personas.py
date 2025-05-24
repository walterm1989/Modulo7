from ultralytics import YOLO
import cv2

# Carga el modelo YOLOv8 pre‑entrenado
model = YOLO('yolov8n.pt')

# Abre la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Aquí cambiamos: sólo clase 0 (persona) ----
    results = model(frame, classes=[0], conf=0.5)  
    # -----------------------------------------------

    # Visualiza los resultados (sólo personas)
    annotated_frame = results[0].plot()
    cv2.imshow('Sólo Personas con YOLOv8', annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()