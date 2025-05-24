import flwr as fl
import cv2
import pickle

SERVER_IP = "xxx.xx.xx.xx" # Reemplaza con la IP del servidor
SERVER_PORT = 8080  # Puerto del servidor

class CameraClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return []
    
    def fit(self, parameters, config):
        # Captura de un solo frame por ronda, se puede adaptar más frames si es necesario
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("No se pudo capturar imagen.")
            return [], 0, {}
        
        # Codifica la imagen para enviar
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        # Enviar la imagen como parte del historial federado
        return [img_bytes], 1, {}
    
    def evaluate(self, parameters, config):
        # No se usa en este ejemplo
        return 0.0, 0, {}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=f"{SERVER_IP}:{SERVER_PORT}",
        client=CameraClient()
    )