import os
configuration = {'0': os.path.join('asone','detectors','yolor','cfg','yolor_csp_x.cfg'),
                 '1': os.path.join('asone','detectors','yolor','cfg','yolor_csp.cfg'),
                 '2': os.path.join('asone','detectors','yolor','cfg','yolor_p6.cfg')}

def get_cfg_path(model_flag):
    if model_flag in [48,50]:
        cfg = configuration['0']
    if model_flag in [52,54]:
        cfg = configuration['1']
    if model_flag == 56:
        cfg = configuration['2']
    return cfg
    
    