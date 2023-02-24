import asone
from asone import ASOne


# Instantiate Asone object
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV7_PYTORCH, use_cuda=True) #set use_cuda=False to use cpu

filter_classes = ['person'] # set to None to track all classes

# ##############################################
#           To track using video file
# ##############################################
# Get tracking function
track_fn = dt_obj.track_video('data/sample_videos/test.mp4', output_dir='data/', save_result=True, display=False, filter_classes=filter_classes)

# Loop over track_fn to retrieve outputs of each frame 
for bbox_details, frame_details in track_fn:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details