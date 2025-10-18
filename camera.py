import threading #Para que siga procesando la cámara independientemente de la GUI
import cv2
from logger import Logger
import time
import joblib
from ultralytics import YOLO
import numpy as np
from extract import extract_features_from_image, binarize
# Acá en camera.py se hace el procesamiento de imágenes:

class Placas(): # Obtener frames de la cámara en paralelo al funcionamiento de GUI
    def __init__(self, src=0, name="Placas_1"):
        self.loggerReport = Logger("logCamera")
        try:
            # Variables globales
            self.name = name
            self.src = src
            self.ret = None
            self.frame = None
            self.stopped = False
            self.predict = None
            self.frame_detected = np.zeros((200, 300, 3), dtype=np.uint8)
            self.yolo = YOLO(r"C:\Users\varga\OneDrive - Universidad EIA\Documents\Universidad EIA\Sexto Semestre\Vision_Artificial-Electiva\VA_Codes\Quiz_2\bestPlateCar.pt")
            self.scaler = joblib.load(r"C:\Users\varga\OneDrive - Universidad EIA\Documents\Universidad EIA\Sexto Semestre\Vision_Artificial-Electiva\VA_Codes\Quiz_2\pca_scaler_rois.pkl")
            self.pca = joblib.load(r"C:\Users\varga\OneDrive - Universidad EIA\Documents\Universidad EIA\Sexto Semestre\Vision_Artificial-Electiva\VA_Codes\Quiz_2\pca_model_rois.pkl")
            self.svm_model = joblib.load(r"C:\Users\varga\OneDrive - Universidad EIA\Documents\Universidad EIA\Sexto Semestre\Vision_Artificial-Electiva\VA_Codes\Quiz_2\rois_svm_model_pca.pkl")
            self.loggerReport.logger.info("Init constructor Placas") #Para poner mensajes
            self.plaque_in_frame = False
            self.plaque_processed = False
        except Exception as e:
            self.loggerReport.logger.error("Error in Placas " + str(e)) #Poner el error en string
    
    def start(self):
        try:
            self.stream = cv2.VideoCapture(self.src)
            time.sleep(1) #Por si se demora en arrancar la cámara
            self.ret, self.frame = self.stream.read() #Guardar los frames del video
            if(self.stream.isOpened()):
                self.loggerReport.logger.info("Creating Thread in start camera")
                self.my_thread = threading.Thread(target=self.get, name=self.name, daemon=True)
                self.my_thread.start()
            else:
                self.loggerReport.logger.warning("start camera no initialized")
        except Exception as e:
            self.loggerReport.logger.error("Error in start Camera " + str(e)) #Poner el error en string
        
        
    def processPlates(self, frame):
        
        results = self.yolo.predict(
        source=frame, conf=0.25, verbose=False)  # Excluir la clase 0 ("Plate")

        # Extraer las cajas delimitadoras (bounding boxes) y dibujar rectángulos
        if len(results) < 1:
            self.plaque_in_frame = False
            return
        for result in results:
            # print(result.boxes)
            boxes = [x for x in result.boxes if x.conf[0] > 0.85]
            if not boxes:
                self.plaque_in_frame = False
                continue
            if self.plaque_in_frame:
                continue
            for box in boxes:  # Para cada caja delimitadora
                # Obtener las coordenadas de la caja delimitadora (formato xyxy)
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )  # Convertir las coordenadas a enteros
                if x1 < 20:
                    continue
                self.plaque_in_frame = True
                self.plaque_processed = False
                # Dibujar el rectángulo alrededor del objeto detectado
                cv2.rectangle(
                    frame, (x1 - 5, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 2
                )  # Color verde y grosor 2
                imgRoi = frame[y1 + 8 : y2 - 12, x1 + 3 : x2 - 5]
                if imgRoi.shape[0] < 20 or imgRoi.shape[1] < 60:
                    continue
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    continue
                self.frame_detected = imgRoi

                # Obtener las características de la placa:
                imgRoi_gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
                imgRoi_bin = binarize(imgRoi_gray)
                clean = cv2.bitwise_not(imgRoi_bin)
                contours, _ = cv2.findContours(
                clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                )
                sizeable_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
                valid_contours = []
                frame_contours = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
                for cnt in sizeable_contours: 
                    x, y, w, h = cv2.boundingRect(cnt)

                    whRatio = h / w
                    if not (1.5 < whRatio < 5):
                        continue
                    valid_contours.append([cnt,(x,y,w,h)])
                    cv2.rectangle(frame_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("contorno",frame_contours)
                
                char_regions = sorted(valid_contours, key=lambda r: r[1][0])
                charas = []
                for i, cnt in enumerate(char_regions):
                    x, y, w, h = cnt[1]
                    charas.append(cv2.resize(clean[y:y+h, x:x+w], (28,28)))
                    cv2.imshow(str(i),cv2.resize(clean[y:y+h, x:x+w], (28,28)))
               
                features_list = []
                for ch in charas:
                    result = extract_features_from_image(0, ch, '?')
                    if result is None:
                        continue
                    _, feat = result
                    features_list.append(feat)

                if not features_list:
                    print("⚠️ No se extrajeron características válidas")
                    return

                predicted_chars = []  # ← Guardará los caracteres detectados

                for feat in features_list:
                    feat = feat.reshape(1, -1)

                    # Ajustar longitud según el scaler
                    expected = self.scaler.mean_.shape[0]
                    if feat.shape[1] > expected:
                        feat = feat[:, :expected]
                    elif feat.shape[1] < expected:
                        pad = np.zeros((1, expected - feat.shape[1]), dtype=np.float32)
                        feat = np.hstack((feat, pad))

                    # Escalar y reducir con PCA
                    feat_scaled = self.scaler.transform(feat)
                    feat_pca = self.pca.transform(feat_scaled)

                    # Predecir carácter
                    pred = self.svm_model["model"].predict(feat_pca)[0]
                    predicted_chars.append(str(pred))

                # === Unir caracteres para formar la placa completa ===
                self.predict = "".join(predicted_chars)
                print("Placa detectada:", self.predict)
            
        

    def get(self):
        while not self.stopped:
            if not self.ret:
                pass
            else:
                try:
                    self.ret, self.frame = self.stream.read()
                    if not self.ret or self.frame is None:
                        self.loggerReport.logger.warning("Frame vacío en Placas.get()")
                        return None
                    
                    self.processPlates(self.frame) # Función donde esté el código para el procesamiento del video
                    
                except Exception as e:
                    self.loggerReport.logger.error("Error in get Camera", exc_info=True)

