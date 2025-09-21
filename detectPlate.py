from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo YOLOv8 (asegúrate de que el archivo 'best.pt' es el que entrenaste con 26 clases)
model = YOLO("./bestPlateCar.pt")

# Ruta del video de entrada y de salida
video_path = "./carVideos/2.mp4"
output_video_path = "predicted_video.mp4"

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
        print(result.boxes)
        for box in result.boxes:  # Para cada caja delimitadora
            # Obtener las coordenadas de la caja delimitadora (formato xyxy)
            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )  # Convertir las coordenadas a enteros
            class_id = int(box.cls[0])  # ID de la clase
            confidence = box.conf[0]  # Confianza de la predicción
            print("confidence is: ", confidence)
            if confidence > 0.5:
                # Dibujar el rectángulo alrededor del objeto detectado
                cv2.rectangle(
                    frame, (x1 - 5, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 2
                )  # Color verde y grosor 2
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
                print(f"{x1=} {x2=}\n{y1=} {y2=}")
                imgRoi = frame[y1 : y2 + 3, x1 : x2 + 3]
                # cv2.imshow("img roi", cv2.resize(imgRoi, (320, 200)))
                out.write(cv2.resize(imgRoi, (320, 200)))

        # # Dibujar las máscaras de segmentación y extraer el ROI
        # if hasattr(result, 'masks') and result.masks is not None:
        #     masks = result.masks  # Extraer las máscaras de la predicción
        #     for mask in masks.data:
        #         # Convertir la máscara a formato OpenCV
        #         mask = mask.cpu().numpy().astype(np.uint8)
        #         mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Redimensionar la máscara

        #         # Encontrar los contornos de la máscara
        #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #         if contours:
        #             # Encontrar el contorno más grande (por si hay varios)
        #             largest_contour = max(contours, key=cv2.contourArea)

        #             # Crear una máscara en blanco para almacenar el ROI segmentado
        #             mask_blank = np.zeros_like(frame)

        #             # Dibujar el contorno en la máscara en blanco (solo el área segmentada)
        #             cv2.drawContours(mask_blank, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        #             # Extraer el ROI utilizando la máscara segmentada
        #             roi = cv2.bitwise_and(frame, mask_blank)

        #             # Dibujar la máscara sobre el frame original (opcional)
        #             color_mask = np.zeros_like(frame)
        #             color_mask[mask > 0] = (255, 0, 0)  # Azul para las máscaras segmentadas

        #             # Superponer la máscara en el frame original
        #             frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

        #             # Mostrar la región de interés (ROI) recortada
        #             cv2.imshow("Segmented ROI", roi)
        #             cv2.waitKey(1)

    # Escribir el frame anotado en el video de salida
    # out.write(frame)
    # cv2.imshow("preddict video", frame)
    # cv2.waitKey(30)

# Liberar los recursos
cap.release()
out.release()

print(f"Video procesado y guardado en {output_video_path}")
