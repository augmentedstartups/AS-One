from asone.recognizers import EasyOCRRecognizer
from asone.recognizers.utils.recognizer_name import get_recognizer_name


class TextRecognizer:
        
    def __init__(self,
                 model_flag: int,
                 languages: list=['en'],
                 use_cuda: bool=True):

        self.model = self._select_recognizer(model_flag, use_cuda, languages)

    def _select_recognizer(self, model_flag, use_cuda, languages):
        recognizer_name = get_recognizer_name(model_flag)
        if recognizer_name == 'easyocr':
            _recognizer = EasyOCRRecognizer(gpu=use_cuda, languages=languages)
        return _recognizer

    def get_recognizer(self):
        return self.model

    def recognize(self,
               image: list,
               horizontal_list=None, free_list=None):
        print(horizontal_list)
        return self.model.recognize(image, horizontal_list=horizontal_list, free_list=free_list)