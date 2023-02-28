import os 

recognizers = {
    '200': 'easyocr'
}

def get_recognizer_name(model_flag):
    
    if model_flag == 200:
        recognizer = recognizers[str(model_flag)]
    
    return recognizer