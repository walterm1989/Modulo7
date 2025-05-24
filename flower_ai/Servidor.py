import flwr as fl
import cv2
import numpy as np

from ultralytics import YOLO

class FlowerServer(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov8n.pt')

    def aggregate_fit(self, rnd, results, failures):
        for client_idx, (params, num_examples, _)in enumerate(results):
            if params:  # params es una lista de bytes, por ejemplo [img_bytes]
                img_bytes = params[0]
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Solo detecta personas (Clase 0)
                    results = self.model(frame, classes=[0], conf=0.5)
                    annotated_frame = results[0].plot()
                    cv2.imshow(f'Persona detecta Cliente {client_idx + 1}', annotated_frame)
                    cv2.waitKey(1)
        return [], {}
    
def main():
    strategy = FlowerServer()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=9999),
        strategy=strategy,
    )
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()