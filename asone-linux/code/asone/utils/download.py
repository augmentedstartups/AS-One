
import gdown
import os
import zipfile


def exractfile(file, dest):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def download_weights(weights):

    outputpath = os.path.dirname(weights)
    model = os.path.splitext(os.path.basename(weights))[0]
    filename = f'{model}.zip'

    if model == 'yolov5s':
        model_key = '13Agcwy0yFxPn6nujHIEB8_KCYg5y-t4a'
    elif model == 'yolov7':
        model_key = '10XNOpBAmMrYqmXOsJLl79MGtuGWY2zAl'
    elif model == 'yolov7-tiny':
        model_key = '1ut2doFvtQSKGjiHGPBsEItZlTTj-7_rF'
    elif model == 'ckpt':
        model_key = '1VZ05gzg249Q1m8BJVQxl3iHoNIbjzJf8'
    else:
        raise ValueError(f'No model named {model} found.')

    url = f'https://drive.google.com/uc?id={model_key}'
    gdown.download(url, output=filename, quiet=False)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    
    exractfile(filename, outputpath)
    os.remove(filename)