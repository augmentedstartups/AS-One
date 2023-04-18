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
        model_key = '1W5ypZmrYE4_Aqu3Jqsl-IDvLK6SOtJCK'
    elif model == 'yolov5x6':
        model_key = '1CTYtGC8VFZD0uJbU4fcSPSjcYb8Be0jU'
    elif model == 'yolov5n':
        model_key = '1q9_e76T_b353QmG5xGi3zkselGQrxuBk'
    elif model == 'yolov5m':
        model_key = '1Vv3VEkgYd7WB-3e2MPo0QUKq_-F7biBP'
    elif model == 'yolov5l':
        model_key = '1Wr4S7BTqqCOCP14T_aDgVxGV5h9pMM-n'
    elif model == 'yolov5x':
        model_key = '18g_pjpwsnOlBKbhApuaTsbQUIDO75BSt'
    elif model == 'yolov5n6':
        model_key = '1gOAZ90nKcvo7bhNZCZM-JRuY8FBeEkhl'
    elif model == 'yolov5s6':
        model_key = '12W1Z0esjFc9UhiWWxCESjhm5KA3nzSwt'
    elif model == 'yolov5m6':
        model_key = '1O-bbJ8WcqLig40IUyJ8ulKwj-J_KXvHr'
    elif model == 'yolov5l6':
        model_key = '1sPZ-YpenYojZSIB5G1SMv2hPWX5oRwlb'
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
        model_key = '1zh0WsfX9cHJnQ2smmHosgphuebe6ivCd'
    elif model == 'yolov7-d6':
        model_key = '1_Ybtx7EAXnBwIZ59Vgo0FkXsCdjRuq2s'
    elif model == 'yolov7':
        model_key = '17iFeNfq5hKZVpQLQEgZzxF9Da5o5llLG'
    elif model == 'yolov7-tiny':
        model_key = '18Fels44wVJ1vG7yDuDPuWwuiAqeFxKI7'
    elif model == 'yolov7-e6':
        model_key = '1g_2nYpeJ28cLYcOAUeztHUA5R-stk3Cm'
    elif model == 'yolov7-w6':
        model_key = '1wv3M23RFo0MhaujegBPY6gZ30IA894CO'
    elif model == 'yolov7x':
        model_key = '1zZskyvdgU45Ke8TtCA6csdCzqLrhmFYx'
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

    elif model == 'yolov8s':
        model_key = '1hUhjQWw1cJL7TtBG0zD3MO52iCYh0DEn'
    elif model == 'yolov8n':
        model_key = '1x6zHzsEcyhuyWy2xY3swAQ4vIQvBYrsr'
    elif model == 'yolov8l':
        model_key = '1xQxHTEIpoiP4d73F6dtedU7hIZpFTqY2'
    elif model == 'yolov8m':
        model_key = '1_FoKnqkaoWchVy4B24Hn2PanEKfh-eSp'
    elif model == 'yolov8x':
        model_key = '1s60fsjiyDlPQ1L5H_GAoWahHONLbvz7T'
        
    else:
        raise ValueError(f'No model named {model} found.')

    url = f'https://drive.google.com/uc?id={model_key}&confirm=t'
    gdown.download(url, output=filename, quiet=False)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    exractfile(filename, outputpath)
    os.remove(filename)

# def download_weights(weights):
    
#     f = open('asone/utils/weights.json')
#     data = json.load(f)
    
#     outputpath = os.path.dirname(weights)
#     model = os.path.splitext(os.path.basename(weights))[0]
#     filename = f'{model}.zip'
#     if model in data:
#         model_key = data[model]
#     else:
#         raise ValueError(f'No model named {model} found.')

#     url = f'https://drive.google.com/uc?id={model_key}&confirm=t'
#     gdown.download(url, output=filename, quiet=False)

#     if not os.path.exists(outputpath):
#         os.makedirs(outputpath)

#     exractfile(filename, outputpath)
#     os.remove(filename)
