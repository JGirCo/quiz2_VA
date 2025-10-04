# Quiz 2 Visión artificial

---

## TODO

- [x] Identificar placas
  - [x] Adaptar el código del profesor para tener imágenes óptimas
- [/] Segmentar letras
  - [x] Preprocesamiénto de imágenes
  - [x] Filtrado de imágenes
  - [x] Binarización
  - [/] Detección de contornos
  - [ ] Filtrado de contornos
  - [ ] Segmentación de letras
- [?] Integrar machine learning
  - [?] Vectorización de letras
  - [?] Integración de machine learning
- [?] Diseñar Interfaz de Usuario

## Arquitectura de software

> [!TODO]

## Segmentación

### Cómo correr

1. Descargar de Teams de la carpeta Quiz_2 bestPlateCar.pt y moverlo a la carpeta del repositorio
2. Descargar video de los carros y añadir a una nueva carpeta llamada carVideos como 1.mp4 **hasta ahora solo se ha probado el video de Kevin**
3. Ejecutar detectPlate.py para obtener predicted_video.mp4
4. Ejecutar binarization.py para obtener binary_video.mp4

En el futuro tanto detectPlate como binarization van a estructurarse como clases que pueden correr en hilos
