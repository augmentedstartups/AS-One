import os

weights = { '0': os.path.join('yolov5','weights','yolov5x6.pt'),
            '1': os.path.join('yolov5','weights','yolov5x6.onnx'),
            '2': os.path.join('yolov5','weights','yolov5s.pt'),
            '3': os.path.join('yolov5','weights','yolov5s.onnx'),
            '4': os.path.join('yolov5','weights','yolov5n.pt'),
            '5': os.path.join('yolov5','weights','yolov5n.onnx'),
            '6': os.path.join('yolov5','weights','yolov5m.pt'),
            '7': os.path.join('yolov5','weights','yolov5m.onnx'),
            '8': os.path.join('yolov5','weights','yolov5l.pt'),
            '9': os.path.join('yolov5','weights','yolov5l.onnx'),
            '10': os.path.join('yolov5','weights','yolov5x.pt'),
            '11': os.path.join('yolov5','weights','yolov5x.onnx'),
            '12': os.path.join('yolov5','weights','yolov5n6.pt'),
            '13': os.path.join('yolov5','weights','yolov5n6.onnx'),
            '14': os.path.join('yolov5','weights','yolov5s6.pt'),
            '15': os.path.join('yolov5','weights','yolov5s6.onnx'),
            '16': os.path.join('yolov5','weights','yolov5m6.pt'),
            '17': os.path.join('yolov5','weights','yolov5m6.onnx'),
            '18': os.path.join('yolov5','weights','yolov5l6.pt'),
            '19': os.path.join('yolov5','weights','yolov5l6.onnx'),
            
            '120': os.path.join('yolov5','weights','yolov5n.mlmodel'),
            '121': os.path.join('yolov5','weights','yolov5s.mlmodel'),
            '122': os.path.join('yolov5','weights','yolov5x6.mlmodel'),
            '123': os.path.join('yolov5','weights','yolov5m.mlmodel'),
            '124': os.path.join('yolov5','weights','yolov5l.mlmodel'),
            '125': os.path.join('yolov5','weights','yolov5x.mlmodel'),
            '126': os.path.join('yolov5','weights','yolov5n6.mlmodel'),
            '127': os.path.join('yolov5','weights','yolov5s6.mlmodel'),
            '128': os.path.join('yolov5','weights','yolov5m6.mlmodel'),
            '129': os.path.join('yolov5','weights','yolov5l6.mlmodel'),
            
            # YOLOv6
            '20': os.path.join('yolov6','weights','yolov6n.pt'),
            '21': os.path.join('yolov6','weights','yolov6n.onnx'),
            '22': os.path.join('yolov6','weights','yolov6t.pt'),
            '23': os.path.join('yolov6','weights','yolov6t.onnx'),
            '24': os.path.join('yolov6','weights','yolov6s.pt'),
            '25': os.path.join('yolov6','weights','yolov6s.onnx'),
            '26': os.path.join('yolov6','weights','yolov6m.pt'),
            '27': os.path.join('yolov6','weights','yolov6m.onnx'),
            '28': os.path.join('yolov6','weights','yolov6l.pt'),
            '29': os.path.join('yolov6','weights','yolov6l.onnx'),
            '30': os.path.join('yolov6','weights','yolov6l_relu.pt'),
            '31': os.path.join('yolov6','weights','yolov6l_relu.onnx'),
            '32': os.path.join('yolov6','weights','yolov6s_repopt.pt'),
            '33': os.path.join('yolov6','weights','yolov6s_repopt.onnx'),
            # YOLOv7
            '34': os.path.join('yolov7','weights','yolov7-tiny.pt'),
            '35': os.path.join('yolov7','weights','yolov7-tiny.onnx'),
            '36': os.path.join('yolov7','weights','yolov7.pt'),
            '37': os.path.join('yolov7','weights','yolov7.onnx'),
            '38': os.path.join('yolov7','weights','yolov7x.pt'),
            '39': os.path.join('yolov7','weights','yolov7x.onnx'),
            '40': os.path.join('yolov7','weights','yolov7-w6.pt'),
            '41': os.path.join('yolov7','weights','yolov7-w6.onnx'),
            '42': os.path.join('yolov7','weights','yolov7-e6.pt'),
            '43': os.path.join('yolov7','weights','yolov7-e6.onnx'),
            '44': os.path.join('yolov7','weights','yolov7-d6.pt'),
            '45': os.path.join('yolov7','weights','yolov7-d6.onnx'),
            '46': os.path.join('yolov7','weights','yolov7-e6e.pt'),
            '47': os.path.join('yolov7','weights','yolov7-e6e.onnx'),
            
            '130': os.path.join('yolov7','weights','yolov7-tiny.mlmodel'),
            '131': os.path.join('yolov7','weights','yolov7.mlmodel'),
            '132': os.path.join('yolov7','weights','yolov7x.mlmodel'),
            '133': os.path.join('yolov7','weights','yolov7-w6.mlmodel'),
            '134': os.path.join('yolov7','weights','yolov7-e6.mlmodel'),
            '135': os.path.join('yolov7','weights','yolov7-d6.mlmodel'),
            '136': os.path.join('yolov7','weights','yolov7-e6e.mlmodel'),
            # YOLOR
            '48': os.path.join('yolor','weights','yolor_csp_x.pt'),
            '49': os.path.join('yolor','weights','yolor_csp_x.onnx'),
            '50': os.path.join('yolor','weights','yolor_csp_x_star.pt'),
            '51': os.path.join('yolor','weights','yolor_csp_x_star.onnx'),
            '52': os.path.join('yolor','weights','yolor_csp_star.pt'),
            '53': os.path.join('yolor','weights','yolor_csp_star.onnx'),
            '54': os.path.join('yolor','weights','yolor_csp.pt'),
            '55': os.path.join('yolor','weights','yolor_csp.onnx'),
            '56': os.path.join('yolor','weights','yolor_p6.pt'),
            '57': os.path.join('yolor','weights','yolor_p6.onnx'),
            # YOLOX
            '58': os.path.join('yolox','weights','yolox_l.pth'),
            '59': os.path.join('yolox','weights','yolox_l.onnx'),
            '60': os.path.join('yolox','weights','yolox_nano.pth'),
            '61': os.path.join('yolox','weights','yolox_nano.onnx'),
            '62': os.path.join('yolox','weights','yolox_tiny.pth'),
            '63': os.path.join('yolox','weights','yolox_tiny.onnx'),
            '64': os.path.join('yolox','weights','yolox_darknet.pth'),
            '65': os.path.join('yolox','weights','yolox_darknet.onnx'),
            '66': os.path.join('yolox','weights','yolox_s.pth'),
            '67': os.path.join('yolox','weights','yolox_s.onnx'),
            '68': os.path.join('yolox','weights','yolox_m.pth'),
            '69': os.path.join('yolox','weights','yolox_m.onnx'),
            '70': os.path.join('yolox','weights','yolox_x.pth'),
            '71': os.path.join('yolox','weights','yolox_x.onnx'),
            # YOLOv8
            '72': os.path.join('yolov8','weights','yolov8n.pt'),
            '73': os.path.join('yolov8','weights','yolov8n.onnx'),
            '74': os.path.join('yolov8','weights','yolov8s.pt'),
            '75': os.path.join('yolov8','weights','yolov8s.onnx'),
            '76': os.path.join('yolov8','weights','yolov8m.pt'),
            '77': os.path.join('yolov8','weights','yolov8m.onnx'),
            '78': os.path.join('yolov8','weights','yolov8l.pt'),
            '79': os.path.join('yolov8','weights','yolov8l.onnx'),
            '80': os.path.join('yolov8','weights','yolov8x.pt'),
            '81': os.path.join('yolov8','weights','yolov8x.onnx'),
            '139': os.path.join('yolov8','weights','yolov8n.mlmodel'),
            '140': os.path.join('yolov8','weights','yolov8s.mlmodel'),
            '141': os.path.join('yolov8','weights','yolov8m.mlmodel'),
            '142': os.path.join('yolov8','weights','yolov8l.mlmodel'),
            '143': os.path.join('yolov8','weights','yolov8x.mlmodel'),
            
            # Text Detectors
            '82': 'craft',
            '83': 'dbnet18',
             
}

def get_weight_path(model_flag):
    coreml= False
    if model_flag in range(0, 20):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    elif model_flag in range(20, 34):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    elif model_flag in range(34, 48):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    elif model_flag in range(48, 58):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    elif model_flag in range(58, 72):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    elif model_flag in range(72, 82):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    elif model_flag in range(82, 85):
        onnx = False
        weight = weights[str(model_flag)]
        
    elif model_flag in range(120, 130):
        weight = weights[str(model_flag)]
        onnx=False
        coreml = True
    elif model_flag in range(130, 137):
        weight = weights[str(model_flag)]
        onnx=False
        coreml = True    
    elif model_flag in range(139, 145):
        weight = weights[str(model_flag)]
        onnx=False
        coreml = True
    return coreml, onnx, weight
        
