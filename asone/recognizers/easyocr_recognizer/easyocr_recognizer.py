import easyocr
import numpy as np
from PIL import *


class EasyOCRRecognizer:
    
    def __init__ (self, languages: list = ['en'], detect_network="craft", 
                    recog_network='standard', gpu=True):    
        self.detect_network = detect_network
        self.gpu = gpu
        self.model = easyocr.Reader(languages, detect_network=self.detect_network, gpu=self.gpu)    
    
    def detect(self, img):
        horizontal_list, free_list = self.model.detect(img)   
        return   horizontal_list, free_list
    
    def recognize(self, img, horizontal_list=None, free_list=None):   
        
        horizontal_list = np.array(horizontal_list)
        horizontal_list = horizontal_list.astype(int)
        horizontal_list = horizontal_list.tolist()
        reformated_input = []
        for bbx in horizontal_list:
            bbx[1], bbx[2] = bbx[2], bbx[1]
            reformated_input.append(bbx[:4])
        horizontal_list = reformated_input
        
        free_list_format = []
        if horizontal_list!=[]:
           for text in horizontal_list:
                xmin, xmax, ymin, ymax = text
                free_list_format.append([[xmin,ymin], [xmax,ymin], [xmax,ymax] , [xmin,ymax]])
        
        free_list.extend(free_list_format)
        results = self.model.recognize(img, horizontal_list=[], free_list=free_list)

        formated_output = []
        for data in results:
            x_list = []
            y_list = []
            for bbx in data[0]:
                x_list.append(int(bbx[0]))
                y_list.append(int(bbx[1]))
            formated_output.append([min(x_list), min(y_list), max(x_list), max(y_list), data[1], data[2]])

        return formated_output
