from ultralytics import YOLO


class Yolov8PoseEstimator:
    def __init__(self, weights, use_cuda):
        self.model = YOLO(weights)
        
    def estimate(self, source):
        results = self.model(source)
        # output = results[0].keypoints
        # # print(output)
        # print(type(output))
        # print(output.shape)
        # exit()
        return results[0].keypoints
        