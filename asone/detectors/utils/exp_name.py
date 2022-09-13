import os

exp_file_name = {'50': (os.path.join('asone','detectors','yolox','exps','default','yolox_l.py'),'yolox-l'),
                 '52': (os.path.join('asone','detectors','yolox','exps','default','yolox_nano.py'),'yolox-nano'),
                 '54': (os.path.join('asone','detectors','yolox','exps','default','yolox_tiny'),'yolox-tiny'),
                 '56': (os.path.join('asone','detectors','yolox','exps','default','yolov3.py'),'yolox-darknet'),
                 '58': (os.path.join('asone','detectors','yolox','exps','default','yolox_s.py'),'yolox-s'),
                 '60': (os.path.join('asone','detectors','yolox','exps','default','yolox_m.py'),'yolox-m'),
                 '62': (os.path.join('asone','detectors','yolox','exps','default','yolox_x.py'),'yolox-x')
                }


def get_exp__name(model_flag):
    
    if model_flag == 50:
        exp, model_name = exp_file_name['50'][0], exp_file_name['50'][1]
    elif model_flag == 52:
        exp, model_name = exp_file_name['52'][0], exp_file_name['52'][1]
    elif model_flag == 54:
        exp, model_name = exp_file_name['54'][0], exp_file_name['54'][1]
    elif model_flag == 56:
        exp, model_name = exp_file_name['56'][0], exp_file_name['56'][1]
    elif model_flag == 58:
        exp, model_name = exp_file_name['58'][0], exp_file_name['58'][1]
    elif model_flag == 60:
        exp, model_name = exp_file_name['60'][0], exp_file_name['60'][1]      
    elif model_flag == 62:
        exp, model_name = exp_file_name['62'][0], exp_file_name['62'][1]  

    return exp, model_name  