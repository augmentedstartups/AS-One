import easyocr
import numpy as np


class TextDetector:
    def __init__(self, detect_network, languages: list = ['en'], use_cuda=True):
        self.use_cuda = use_cuda
        self.detect_network = detect_network
        self.reader = easyocr.Reader(languages, detect_network=self.detect_network ,gpu=self.use_cuda)
        
    def detect(self, image: list,  freelist: bool=False, return_image=False, **config) -> list:
        """_summary_
        Args:
            image : Image 
            languages (list, optional): List of languages. Defaults to ['en'].
        Returns:
            list: numpy array of extracted text and img info(heigh, width)
        """
        
        h, w = image.shape[0:2]
        horizontal_list, free_list = self.reader.detect(image) 

        if horizontal_list[0] == [] and free_list[0] == []:
            if return_image:
                return horizontal_list, image
            return np.empty((0, 6)), {'width': w, 'height': h}
        
        if freelist:
            return horizontal_list, free_list, {'width': w, 'height': h}
        
        x_list = []
        y_list = []
        new_points = []
        if free_list[0] != []:
            bbox_list = np.array(free_list[0]).astype(int)
            xmin= bbox_list[:, :, 0].min(axis=1, keepdims=True)
            xmax= bbox_list[:, :, 0].max(axis=1, keepdims=True)
            ymin= bbox_list[:, :, 1].min(axis=1, keepdims=True)
            ymax= bbox_list[:, :, 1].max(axis=1, keepdims=True)
            new_points = np.hstack((xmin, xmax, ymin,  ymax)).tolist()

        if len(horizontal_list[0]) < 1:
            horizontal_list = [new_points]
        else:
            horizontal_list = [horizontal_list[0] + new_points]

        horizontal_list = np.array(horizontal_list[0])
        horizontal_list[:, [1, 2]] = horizontal_list[:, [2, 1]]
        horizontal_list = np.hstack((horizontal_list, np.array([[0.7, 80]]*len(horizontal_list))))
        
        if return_image:
            return horizontal_list, image
            
        return  horizontal_list, {'width': w, 'height': h}
