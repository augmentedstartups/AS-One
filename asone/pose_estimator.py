import cv2
import numpy as np
import os
from .utils import draw_kpts, plot_skeleton_kpts
import cv2
import time
import warnings

from asone.pose_estimators.yolov7_pose import Yolov7PoseEstimator
from asone.pose_estimators.yolov8_pose import Yolov8PoseEstimator
from asone.utils import get_weight_path, download_weights
from asone.schemas.output_schemas import ModelOutput

class PoseEstimator:
    def __init__(self, estimator_flag, weights: str=None, use_cuda=True):

        if weights:
            weights = weights
        else:
            weights = get_weight_path(estimator_flag)
            if not os.path.exists(weights):
                download_weights(weights)
        self.estimator = self.get_estimator(estimator_flag, weights, use_cuda)
        self.model_output = ModelOutput()

    def get_estimator(self, estimator_flag: int, weights: str, use_cuda: bool):
        
        if estimator_flag in range(149, 155):
            estimator = Yolov7PoseEstimator(weights=weights, use_cuda=use_cuda)

        elif estimator_flag in range(144, 149):
            estimator = Yolov8PoseEstimator(weights=weights,
                                       use_cuda=use_cuda)
        return estimator
    
    def estimate_image(self, frame):
    
        keypoints = self.estimator.estimate(frame)
        return keypoints.cpu().numpy().xy
    
    def estimate_video(self, video_path, save=True, conf_thresh=0.5, display=True):
       # Emit the warning for DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("estimate_video function is deprecated. Kindly use video_estimator instead", DeprecationWarning)
            
        if video_path == 0:
            cap = cv2.VideoCapture(0)
            video_path = 'webcam.mp4'
        else:
            cap = cv2.VideoCapture(video_path)
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        FPS = cap.get(cv2.CAP_PROP_FPS)

        if save:
            video_writer = cv2.VideoWriter(
                os.path.basename(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (int(width), int(height)),
            )
        
        frame_no = 1
        tic = time.time()
        frame_id = 1
        prevTime = 0
        fframe_num = 0
        while True:
            start_time = time.time()

            ret, img = cap.read()
            if not ret:
                break
            frame = img.copy()
            kpts = self.estimator.estimate(img)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if kpts is not None:
                img = draw_kpts(kpts, image=img) 
                
            cv2.line(img, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(img, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            frame_id += 1
            frame_no+=1
            if display:
                cv2.imshow('Window', img)

            if save:
                video_writer.write(img)
    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            yield (kpts), (img if display else frame, frame_id-1, fps)
    
    def video_estimator(self, video_path, save=True, conf_thresh=0.5, display=True):
       
        if video_path == 0:
            cap = cv2.VideoCapture(0)
            video_path = 'webcam.mp4'
        else:
            cap = cv2.VideoCapture(video_path)
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        FPS = cap.get(cv2.CAP_PROP_FPS)

        if save:
            video_writer = cv2.VideoWriter(
                os.path.basename(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (int(width), int(height)),
            )
        
        frame_no = 1
        tic = time.time()
        frame_id = 1
        prevTime = 0
        fframe_num = 0
        while True:
            start_time = time.time()

            ret, img = cap.read()
            if not ret:
                break
            frame = img.copy()
            kpts = self.estimator.estimate(img)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if kpts is not None:
                img = draw_kpts(kpts, image=img) 
                
            cv2.line(img, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(img, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            frame_id += 1
            frame_no+=1
            if display:
                cv2.imshow('Window', img)

            if save:
                video_writer.write(img)
    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            yield self.format_output((kpts), (img if display else frame, frame_id-1, fps))
    
    
    def format_output(self, bbox_details, frame_details):

        # Set detections
        self.model_output.dets.bbox = bbox_details
        if frame_details:
            # Set image info
            self.model_output.info.image = frame_details[0]
            self.model_output.info.frame_no = frame_details[1]
            self.model_output.info.fps = frame_details[2]

        return self.model_output