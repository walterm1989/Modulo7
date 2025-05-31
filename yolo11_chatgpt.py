from ultralytics import YOLO
import cv2

# Cargar modelo YOLOv11
model = YOLO("yolo11n.pt")

# Ruta del video de entrada
video_path = "./Video_personas_caminando1.mp4"

# Ejecutar tracking
results = model.track(source=video_path, persist=True, stream=True, show=False)

# Almacena las trayectorias por ID
track_paths = []
merged_tracks = []

# Guardar también las cajas para dibujarlas más tarde
all_boxes = []

for result in results:
    frame_boxes = []
    boxes = result.boxes
    frame_tracks = {}

    for box in boxes:
        if int(box.cls[0]) != 0:  # Solo personas
            continue
        if box.id is None:
            continue

        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        frame_tracks[track_id] = center
        frame_boxes.append((track_id, x1, y1, x2, y2))

        # Guardar trayectoria para este ID
        while len(merged_tracks) <= track_id:
            merged_tracks.append([])
        merged_tracks[track_id].append(center)

    track_paths.append(frame_tracks)
    all_boxes.append(frame_boxes)

# Preparar video de salida
cap = cv2.VideoCapture(video_path)
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("person_tracking_output.avi",
                      cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(track_paths):
        break

    frame_tracks = track_paths[frame_idx]
    frame_boxes = all_boxes[frame_idx]

    for track_id, x1, y1, x2, y2 in frame_boxes:
        # Dibuja bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Mostrar el ID encima
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar trayectoria
        if track_id < len(merged_tracks):
            for pt in merged_tracks[track_id]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print("✅ Video de salida generado con cajas e IDs: person_tracking_output.avi")
