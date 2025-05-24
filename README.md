# Modulo7 - Detección de personas por webcam

Proyecto del módulo 7: detección de personas en tiempo real utilizando YOLOv8 y Keypoint RCNN con Python.

## Novedad: Arquitectura Distribuida con Flower AI

Ahora, además del script clásico de detección local, el proyecto incluye una solución distribuida usando [Flower AI](https://flower.dev/). Esto permite que múltiples clientes capturen imágenes desde sus propias webcams y las envíen a un servidor central, donde se realiza la detección de personas con YOLOv8 y se visualizan los resultados.

### Estructura distribuida

- **Servidor (Servidor.py):**  
  Recibe imágenes de los clientes, ejecuta la detección de personas y muestra los resultados.
- **Clientes (Cliente_Marina.py y Cliente_Sergio.py):**  
  Capturan imágenes desde la webcam local y las envían automáticamente al servidor mediante Flower AI.

---

## Instalación de dependencias

Instala todas las dependencias necesarias (Flower, OpenCV, Ultralytics, etc.):

```bash
pip install -r requirements.txt


**Autores:**  
Walter Mosqueira
Sergio Pizarro
Marina Pariente