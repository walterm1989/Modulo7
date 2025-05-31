from ultralytics import YOLO
import cv2

# Cargar modelo YOLOv11 (debe estar en el mismo directorio)
model = YOLO("yolo11n.pt")

# Inicializar cámara (0 = cámara por defecto del portátil)
cap = cv2.VideoCapture(0)
aforo = 2  # Límite de personas
merged_tracks = {}  # Trayectorias

# Parámetros del video de salida
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS estimado si es 0

out = cv2.VideoWriter("person_tracking_output.avi",
                      cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

print("🎥 Cámara iniciada. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar predicción + tracking sobre el frame actual
    results = model.track(frame, persist=True, classes=[0], conf=0.6, verbose=False)[0]
    boxes = results.boxes

    unique_ids = set()
    for box in boxes:
        if int(box.cls[0]) != 0 or box.id is None:
            continue

        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        unique_ids.add(track_id)

        # Guardar trayectoria
        if track_id not in merged_tracks:
            merged_tracks[track_id] = []
        merged_tracks[track_id].append(center)

        # Dibujar bounding box e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar trayectoria
        #for pt in merged_tracks[track_id]:
        #    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

    # Control de aforo
    person_count = len(unique_ids)
    if person_count > aforo:
        msg = "Aforo superado: aplicar protocolo 66"
        color = (0, 0, 255)  # rojo
    else:
        msg = f"Personas detectadas: {person_count} / {aforo}"
        color = (0, 255, 0)  # verde

    cv2.putText(frame, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mostrar en ventana
    cv2.imshow("Detección en vivo", frame)

    # Guardar frame en video de salida
    out.write(frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("🛑 Finalizado. Video guardado como person_tracking_output.avi")


