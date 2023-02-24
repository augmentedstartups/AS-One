import asone
from asone import ASOne

# Instantiate Asone object
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.CRAFT, use_cuda=True) #set use_cuda=False to use cpu

# ##############################################
#           To track using video file
# ##############################################
# Get tracking function
track_fn = dt_obj.track_video('data/sample_videos/license_video.mp4', output_dir='data', save_result=False, display=True)

# Loop over track_fn to retrieve outputs of each frame 
for bbox_details, frame_details in track_fn:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    