from logger import Logger
import tkinter as tk
from PIL import Image, ImageTk #pip install pillow
import tkinter.font as font
from tkinter import ttk
import cv2 #pip install opencv-python
import camera
import numpy as np


class Application(ttk.Frame):
    def __init__(self, master=None): #Todo lo gráfico irá en el master
        super().__init__(master)
        # variables globales:
        self.logReport = Logger("logGUI")
        self.logReport.logger.info("init GUI")
        self.frame = None
        self.imgTk = None # Video
        self.imgTk2 = None # Placa
        self.master = master

        # Resolución o tamaño de la imagen:
        self.width = 1080 
        self.height = 720
        self.master.geometry("%dx%d" % (self.width, self.height))
        
        self.createFrame(1, 80, 100, 500, 400) 
        self.createButton(80, 510)
        self.widgetText("Placa detectada:", 620, 405, 13)
        self.createFrame(2, 620, 190, 300, 200) 

        self.master.mainloop()
    
    def createFrame(self, video, xpos, ypos, width, height): # Video
        match (video):
            case 1:
                self.labelVideo_1 = tk.Label(
                    self.master,
                    borderwidth = 2,
                    relief="solid" #Borde continuo
                )
                self.labelVideo_1.place(x=xpos, y=ypos)
                self.createImageZeros(1, width, height) #Imagen de ceros negra
                self.labelVideo_1.configure(image=self.imgTk) #Imágenes a color del mismo tamaño de imgTk
                self.labelVideo_1.image = self.imgTk
            case 2:
                self.labelVideo_2 = tk.Label(
                    self.master,
                    borderwidth = 2,
                    relief="solid" #Borde continuo
                )
                self.labelVideo_2.place(x=xpos, y=ypos)
                self.createImageZeros(2, width, height) #Imagen de ceros negra
                self.labelVideo_2.configure(image=self.imgTk2) #Imágenes a color del mismo tamaño de imgTk
                self.labelVideo_2.image = self.imgTk2
    
    def createImageZeros(self, video, w, h):
        self.frame = np.zeros([h, w, 3], dtype=np.uint8)  # imagen negra
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(self.frame)
        match (video):
            case 1:
                self.imgTk = ImageTk.PhotoImage(image=imgArray)
            case 2:
                self.imgTk2 = ImageTk.PhotoImage(image=imgArray)
    
    def widgetText(self, title, xpos, ypos, size):
        self.fontLabelText = font.Font(
            family='Helvetica', size = size, weight = 'bold' #Fuente para el texto
        )
        self.labelNameCamera = tk.Label(
            self.master, text = title, fg = '#000000'
        )
        self.labelNameCamera['font'] = self.fontLabelText
        self.labelNameCamera.place(x=xpos, y=ypos)

    def createButton(self, xpos, ypos):
        self.btnInitCamera = tk.Button(
            self.master,
            text="START",
            bg = '#007A39',
            fg = '#ffffff',
            width = 12,
            command = self.initCamera
        )
        self.btnInitCamera.place(x=xpos, y=ypos) #Lugar del botón dependiendo del tamaño de la resolución dada width and height

    def initCamera(self):
        self.Classifier = camera.Placas(r"C:\Users\varga\OneDrive - Universidad EIA\Documents\Universidad EIA\Sexto Semestre\Vision_Artificial-Electiva\VA_Codes\Quiz_2\VideoPlacasQuiz2.mov") # Ruta del video
        self.Classifier.start() #start de camera_actividad5.py
        self.showVideo()
        print("Start Classifier")
    
        
    def showVideo(self): #Actualiza cada frame para mostrar video continuo
        if(self.Classifier.frame is not None):
            #Video:
            imgresize =cv2.resize(self.Classifier.frame, (500, 400))
            imgTk = self.convertToFrameTk(1, imgresize) # Ventana convertida a Tkinter 
            self.labelVideo_1.configure(image=imgTk)
            self.labelVideo_1.image = imgTk
            
            if self.Classifier.frame_detected is not None and self.Classifier.frame_detected.size != 0:
                imgresize2 = cv2.resize(self.Classifier.frame_detected, (300, 200))
            else:
                return 
            imgTk2 = self.convertToFrameTk(1, imgresize2) # Ventana convertida a Tkinter 
            self.labelVideo_2.configure(image=imgTk2)
            self.labelVideo_2.image = imgTk2
            
            self.widgetText(self.Classifier.predict, 750, 405, 13)
        self.labelVideo_1.after(10, self.showVideo) #Para actualizarse cada 10ms
            
    
    def convertToFrameTk(self, img, frame): #1, BGR; 2, Binary and GrayScale
        if(img==1):    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Tkinter solo recibe imágenes en RGB
            imgArray = Image.fromarray(frame) #Y Tkinter lo tiene como un array
        else:
            framebin = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Tkinter solo recibe imágenes en RGB
            imgArray = Image.fromarray(framebin) #Y Tkinter lo tiene como un array
        return ImageTk.PhotoImage(image = imgArray)
    
def main():
    root = tk.Tk()
    root.title("Detector de placas")
    appRunCamera = Application(master=root)
