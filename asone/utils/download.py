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
        model_key = '1H7G8ryDXs6bKlK2Qot7-2uIkjEYoYook'
    elif model == 'yolov5x6':
        model_key = '161bThpOB4HDqrh2FXvbFZJmiSKFwS_Wb'
    elif model == 'yolov5n':
        model_key = '1zI4f0AUHAz-fTE_fP7UyiFSRGBYYXd7x'
    elif model == 'yolov5m':
        model_key = '1vy8S68wbUzKSHMhsTuLN-VA7lMzKchAa'
    elif model == 'yolov5l':
        model_key = '1pQL9s0o3v6CycAgAX8SkxCfordUl5IxZ'
    elif model == 'yolov5x':
        model_key = '1iB7MQ1IP3MVKLMF8TIJ44vtv9cjWC2qH'
    elif model == 'yolov5n6':
        model_key = '1YxnRYlPcCqXGbX20kPlfSimNfROKwoJH'
    elif model == 'yolov5s6':
        model_key = '1mm5zY6IpPtM7IZh_X5x0kAxuO7INKyte'
    elif model == 'yolov5m6':
        model_key = '1qv_uan5oNq9skcg1UThfaFs0xMs2mSE2'
    elif model == 'yolov5l6':
        model_key = '1eaM51cIh8i_EXmg6Nf0Sx2uW53pT7wZR'
    elif model == 'yolov6n':
        model_key = '1NA_u4BkPE_N8HcPmZrd7HyLmvFHOk8qd'
    elif model == 'yolov6t':
        model_key = '16OWncBp-vh-sLDMOR58th3WOGv4envQ1'
    elif model == 'yolov6s':
        model_key = '14BE0j654ClLxMq2ySWZNhTCmf48mLyXi'
    elif model == 'yolov6l_relu':
        model_key = '14UfY057QUQoAj6q39PX_qE7U1bBIrIGi'
    elif model == 'yolov6l':
        model_key = '1HdRIs0uMPbqs5E2aEX8O3d3dJTh-KBTf'
    elif model == 'yolov6m':
        model_key = '1t_w9SCwbZAW7icwX_z97-SQz-plXzBgM'
    elif model == 'yolov6s_repopt':
        model_key = '1L_1Crxx-4059xDDUZEf_asWRBVd3PF05'
    elif model == 'yolov7-e6e':
        model_key = '1rQR5KiSJiWtpHEniAyeBQdpXFb7Wv1UT'
    elif model == 'yolov7-d6':
        model_key = '1idAyjdq9pVsgkDCCfADbGOjxGq4TPulB'
    elif model == 'yolov7':
        model_key = '10XNOpBAmMrYqmXOsJLl79MGtuGWY2zAl'
    elif model == 'yolov7-tiny':
        model_key = '1ut2doFvtQSKGjiHGPBsEItZlTTj-7_rF'
    elif model == 'yolov7-e6':
        model_key = '1E9pow2PFcvil0iqRx2tRCI4HLduh9gp0'
    elif model == 'yolov7-w6':
        model_key = '1B8j9XMZxGxz8kpsqJhKXuk1TE_244n6t'
    elif model == 'yolov7x':
        model_key = '1FiGLXG6_3He21ean4bFET471Wrj-3oc3'
    elif model == 'yolor_csp':
        model_key = '1G3FBZKrznW_64mGfs6b3nAJiJv6GmmV0'
    elif model == 'yolor_csp_star':
        model_key = '15WDl46ZthFGZfpOyI3qXx6gC9FQLH_wH'
    elif model == 'yolor_csp_x':
        model_key = '1LU2ckh7eSpVD0nyPSdq1n34lKmNAX39T'
    elif model == 'yolor_csp_x_star':
        model_key = '1jheqFDm7BpHQpR60wuWSBpbuyK5SoKdV'
    elif model == 'yolor_p6':
        model_key = '1XKREKdxQCO8OXiW2IWGFhczolIaIr9sm'
    elif model == 'yolox_l':
        model_key = '1jX1KHerOdZ5-dmXh6cWcRAn80aKD-7sP'
    elif model == 'yolox_nano':
        model_key = '1783Os6uTpYjunL-MfK0WE1Wcwk58fIUi'
    elif model == 'yolox_tiny':
        model_key = '1Lcv1ITvfPdWsu6Kb8Hq6cOSfJE7lbbf2'
    elif model == 'yolox_darknet':
        model_key = '17f4UI06TWJ25Oqo2OoQGu8AoGVX1lPta'
    elif model == 'yolox_s':
        model_key = '1IUAOv62XuwkwwTCVD2y3xJg7KUA3M0-M'
    elif model == 'yolox_m':
        model_key = '1ktHj8UEwl0V8Qz9G74E-yj-o13FLeD0-'
    elif model == 'yolox_x':
        model_key = '13HNnlILCx_XamNJWwJ1MG5x0XfP6HL1U'
    elif model == 'ckpt':
        model_key = '1VZ05gzg249Q1m8BJVQxl3iHoNIbjzJf8'
    else:
        raise ValueError(f'No model named {model} found.')

    url = f'https://drive.google.com/uc?id={model_key}&confirm=t'
    gdown.download(url, output=filename, quiet=False)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    exractfile(filename, outputpath)
    os.remove(filename)
