from pose_estimate import PoseEstimator

a = PoseEstimator()
a.run()


# parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
#     parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
#     parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
#     parser.add_argument('--view-img', action='store_true', help='display results')  #display results
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf