import os

weights = { 
            
            # '130': os.path.join('yolov7','weights','yolov7-tiny.mlmodel'),
            # '131': os.path.join('yolov7','weights','yolov7.mlmodel'),
            # '132': os.path.join('yolov7','weights','yolov7x.mlmodel'),
            '149': os.path.join('yolov7','weights','yolov7-w6-pose.pt'),
            # '134': os.path.join('yolov7','weights','yolov7-e6.mlmodel'),
            # '135': os.path.join('yolov7','weights','yolov7-d6.mlmodel'),
            # '136': os.path.join('yolov7','weights','yolov7-e6e.mlmodel'),
            
            '144': os.path.join('yolov8','weights','yolov8n-pose.pt'),
            '145': os.path.join('yolov8','weights','yolov8s-pose.pt'),
            '146': os.path.join('yolov8','weights','yolov8m-pose.pt'),
            '147': os.path.join('yolov8','weights','yolov8l-pose.pt'),
            '148': os.path.join('yolov8','weights','yolov8x-pose.pt') 
}

def get_weight_path(model_flag):
    weight = weights[str(model_flag)]
    return weight