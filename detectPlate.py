from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo YOLOv8 (asegúrate de que el archivo 'best.pt' es el que entrenaste con 26 clases)
model = YOLO("./bestPlateCar.pt")

# Ruta del video de entrada y de salida
video_path = "./carVideos/full_video.mov"
output_video_path = "predicted_video_full.mp4"

# Abrir el video
cap = cv2.VideoCapture(video_path)

# Obtener la resolución del video original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# # Crear un objeto para escribir el video con las predicciones
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para el video de salida
out = cv2.VideoWriter(output_video_path, fourcc, fps, (320, 200))

# Procesar cada frame del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Salir si no quedan más frames

    # Realizar la predicción en el frame actual, omitiendo la clase "Plate"
    results = model.predict(
        source=frame, conf=0.25, verbose=False
    )  # Excluir la clase 0 ("Plate")

    # Extraer las cajas delimitadoras (bounding boxes) y dibujar rectángulos
    for result in results:
        # print(result.boxes)
        for box in result.boxes:  # Para cada caja delimitadora
            # Obtener las coordenadas de la caja delimitadora (formato xyxy)
            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )  # Convertir las coordenadas a enteros
            class_id = int(box.cls[0])  # ID de la clase
            confidence = box.conf[0]  # Confianza de la predicción
            # print("confidence is: ", confidence)
            if confidence < 0.85:
                continue
            # Dibujar el rectángulo alrededor del objeto detectado
            cv2.rectangle(
                frame, (x1 - 5, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 2
            )  # Color verde y grosor 2
            cv2.imshow("img roi", cv2.resize(frame, (320, 200)))
            label = f"Class {class_id} ({confidence:.2f})"

            # Poner la etiqueta con el ID de la clase y la confianza
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            # print(f"{x1=} {x2=}\n{y1=} {y2=}")
            imgRoi = frame[y1 + 8 : y2 - 12, x1 + 3 : x2 - 5]
            print(f"{imgRoi.shape=}")
            if imgRoi.shape[0] < 20 or imgRoi.shape[1] < 60:
                continue
            out.write(cv2.resize(imgRoi, (320, 200)))
            if cv2.waitKey(10) & 0xFF == ord("q"):
                continue
            # cv2.imwrite("placa.png", imgRoi)

# Liberar los recursos
cap.release()
out.release()

print(f"Video procesado y guardado en {output_video_path}")
