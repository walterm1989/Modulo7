from ultralytics import YOLO
import cv2

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8n.pt')

# Abre la WebCam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza inferencia en la imagen
    results = model(frame)

    # Visualiza los resultados en la imagen
    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8 Detección de Personas', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27: # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()