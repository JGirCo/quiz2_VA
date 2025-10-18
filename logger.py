import logging
from colorlog import ColoredFormatter # pip install colorlog

class Logger:
    def __init__(self, name, log_file = r"C:\Users\varga\OneDrive - Universidad EIA\Documents\Universidad EIA\Sexto Semestre\Vision_Artificial-Electiva\VA_Codes\GUI\logs\app.log"): 
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.log_file = log_file 
        
        if not self.logger.hasHandlers():
            #Asignar un handler de logs a la consola
            console_handler = logging.StreamHandler() #Imprime en la consola en todo momento lo que vaya sucediendo
            console_handler.setLevel(logging.DEBUG)
            console_formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S", #Le pondrá los colores respectivos a los mensajes que aparezcan:
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                },
            )
            console_handler.setFormatter(console_formatter)

            #Asignar un handler de logs a un archivo
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)

            #Agregar los handlers al logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def debug(self, message):
        self.logger.debug(message)
 
    def info(self, message):
        self.logger.info(message)
 
    def warning(self, message):
        self.logger.warning(message)
 
    def error(self, message):
        self.logger.error(message)
   
    def critical(self, message):
        self.logger.critical(message)
