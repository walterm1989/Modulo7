import torch
import torchvision
import cv2
import numpy as np

# Cargar el modelo Keypoint R-CNN preentrenado
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Usa la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar la imagen
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Dibujar los keypoints detectados
    for keypoints in outputs['keypoints']:
        for keypoint in keypoints:
            x, y, v = keypoint
            if v > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Keypoint R-CNN Detección de Personas', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()