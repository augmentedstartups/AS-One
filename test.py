import asone
from asone import PoseEstimator

video_path = 'data/sample_videos/football1.mp4'
pose_estimator = PoseEstimator(estimator_flag=asone.YOLOV7_W6_POSE, use_cuda=True)
# pose_estimator = PoseEstimator(estimator_flag=asone.YOLOV8M_POSE, use_cuda=True)
pose_estimator.estimate_video(video_path, save=True, display=False)
