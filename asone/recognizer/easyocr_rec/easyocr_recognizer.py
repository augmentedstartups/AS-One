import easyocr


class EasyocrRecognizer:
    
    def __init__ (self, languages: list = ['en'], detect_network="craft", 
                    recog_network='standard', gpu=True):    
        self.detect_network = detect_network
        self.gpu = gpu
        self.model = easyocr.Reader(languages, detect_network=self.detect_network, gpu=self.gpu)    
    
    def detect(self, img):
        horizontal_list, free_list = self.model.detect(img)   
        return   horizontal_list, free_list
    
    def recognize(self, img, horizontal_list=None, free_list=None):    
        dets = self.model.recognize(img, horizontal_list=horizontal_list[0], free_list=free_list[0])
        return dets