import os
configuration = {'0': os.path.join('asone','detectors','yolor','cfg','yolor_csp_x.cfg'),
                 '1': os.path.join('asone','detectors','yolor','cfg','yolor_csp.cfg'),
                 '2': os.path.join('asone','detectors','yolor','cfg','yolor_p6.cfg')}

def get_cfg_path(model_flag):
    if model_flag in [40,42]:
        cfg = configuration['0']
    if model_flag in [44,46]:
        cfg = configuration['1']
    if model_flag == 48:
        cfg = configuration['2']
    return cfg
    
    