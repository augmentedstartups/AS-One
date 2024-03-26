import os 

weights = {          
            # Segmentor
            '171':os.path.join('sam','weights','sam_vit_h_4b8939.pth'),
}

def get_weight_path(model_flag):
    if model_flag == 171:
        weight = weights[str(model_flag)]
    
    return weight