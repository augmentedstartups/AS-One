import os

exp_dir = os.path.dirname(os.path.dirname(__file__))

exp_file_name = {'58': (os.path.join(exp_dir, 'yolox','exps','yolox_l.py'),'yolox-l'),
                 '60': (os.path.join(exp_dir, 'yolox','exps','yolox_nano.py'),'yolox-nano'),
                 '62': (os.path.join(exp_dir, 'yolox','exps','yolox_tiny'),'yolox-tiny'),
                 '64': (os.path.join(exp_dir, 'yolox','exps','yolov3.py'),'yolox-darknet'),
                 '66': (os.path.join(exp_dir, 'yolox','exps','yolox_s.py'),'yolox-s'),
                 '68': (os.path.join(exp_dir, 'yolox','exps','yolox_m.py'),'yolox-m'),
                 '70': (os.path.join(exp_dir, 'yolox','exps','yolox_x.py'),'yolox-x')
                }


def get_exp__name(model_flag):
    
    if model_flag == 58:
        exp, model_name = exp_file_name['58'][0], exp_file_name['58'][1]
    elif model_flag == 60:
        exp, model_name = exp_file_name['60'][0], exp_file_name['60'][1]
    elif model_flag == 62:
        exp, model_name = exp_file_name['62'][0], exp_file_name['62'][1]
    elif model_flag == 64:
        exp, model_name = exp_file_name['64'][0], exp_file_name['64'][1]
    elif model_flag == 66:
        exp, model_name = exp_file_name['66'][0], exp_file_name['66'][1]
    elif model_flag == 68:
        exp, model_name = exp_file_name['68'][0], exp_file_name['68'][1]      
    elif model_flag == 70:
        exp, model_name = exp_file_name['70'][0], exp_file_name['70'][1]  

    return exp, model_name  