from ultralytics import YOLO
import torch


class Yolov8PoseEstimator:
    def __init__(self, weights, use_cuda=True):
        self.model = YOLO(weights)
        self.device = 0 if use_cuda and torch.cuda.is_available() else  'cpu'
        
    def estimate(self, source):
        results = self.model(source, device=self.device)
        return results[0].keypoints
        