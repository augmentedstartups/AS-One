import easyocr


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
        
        results = self.model.recognize(img, horizontal_list=horizontal_list[0], free_list=free_list[0])
        formated_output = []
        
        for result in results:
            formated_output.append((result[0][0][0], result[0][0][1], result[0][2][0], 
                    result[0][2][1], result[1], result[2]))
        return formated_output