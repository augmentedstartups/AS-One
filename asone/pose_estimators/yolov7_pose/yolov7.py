import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from .utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from .utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from .utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt


class Yolov7PoseEstimator:
    def __init__(self, weights="yolov7-w6-pose.pt", device='0'): 
        self.weights=weights
        # device = device
        self.device = select_device('0')
        half = self.device.type != 'cpu'
        self.model = attempt_load(self.weights, map_location=self.device)
        _ = self.model.eval()

    @torch.no_grad()
    def estimate(self, frame):

        frame_height, frame_width = frame.shape[:2]
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (frame_width), stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        image = image.to(self.device)
        image = image.float()
        start_time = time.time()
        
        with torch.no_grad():
            output, _ = self.model(image)
            
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], 
                                         nkpt=self.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        
        # ............................................
        # converting output to the format as yolov8
        reformated_output = []
        steps = 3
        kpts = output
        for idx in range(output.shape[0]):
            single_person_kpts = kpts[idx, 7:].T
            num_kpts = len(single_person_kpts) // steps
            xyc = []
            for kid in range(num_kpts):
                x_coord, y_coord = single_person_kpts[steps * kid], single_person_kpts[steps * kid + 1]
                xyc.append([x_coord, y_coord, single_person_kpts[steps * kid + 2]])
            reformated_output.append(xyc)
        out = np.array(reformated_output)
        output = torch.from_numpy(out)
        
        return output
        