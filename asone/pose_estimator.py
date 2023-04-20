from asone.pose_estimators.yolov7_pose import Yolov7PoseEstimator

class PoseEstimator:
    def __init__(self):
        a = Yolov7PoseEstimator(device=True)
        a.estimate()