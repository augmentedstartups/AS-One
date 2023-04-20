import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from .utils.datasets import letterbox
from .utils.torch_utils import select_device
from models.experimental import attempt_load
from .utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from .utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt


class Yolov7PoseEstimator:
    def __init__(self, poseweights="yolov7-w6-pose.pt", 
                 source="data/sample_videos/video2.mp4", device='0'): 
        self.poseweights = poseweights
        self.source = source
        self.device = '0' if device else cpu
   
    #function for plot fps and time comparision graph
    def plot_fps_time_comparision(self, time_list,fps_list):
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('FPS')
        plt.title('FPS and Time Comparision Graph')
        plt.plot(time_list, fps_list,'b',label="FPS & Time")
        plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
           
    @torch.no_grad()
    def estimate(self,view_img=False,
            save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):

        frame_count = 0  #count no of frames
        total_fps = 0  #count total fps
        time_list = []   #list to store time
        fps_list = []    #list to store fps
        
        device = select_device(self.device) #select device
        half = device.type != 'cpu'

        model = attempt_load(self.poseweights, map_location=device)  #Load model
        _ = model.eval()
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        source = self.source
        if source.isnumeric() :    
            cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
        else :
            cap = cv2.VideoCapture(source)    #pass video to videocapture object
    
        if (cap.isOpened() == False):   #check if videocapture not opened
            print('Error while trying to read video. Please check path again')
            raise SystemExit()

        else:
            frame_width = int(cap.get(3))  #get video frame width
            frame_height = int(cap.get(4)) #get video frame height

            
            vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
            resize_height, resize_width = vid_write_image.shape[:2]
            out_video_name = f"{source.split('/')[-1].split('.')[0]}"
            out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                (resize_width, resize_height))

            while(cap.isOpened): #loop until cap opened or video not complete
            
                print("Frame {} Processing".format(frame_count+1))

                ret, frame = cap.read()  #get frame and success from video capture
                
                if ret: #if success is true, means frame exist
                    orig_image = frame #store frame
                    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                    image_ = image.copy()
                    image = transforms.ToTensor()(image)
                    image = torch.tensor(np.array([image.numpy()]))
                
                    image = image.to(device)  #convert image data to device
                    image = image.float() #convert image to float precision (cpu)
                    start_time = time.time() #start time for fps calculation
                
                    with torch.no_grad():  #get predictions
                        output_data, _ = model(image)

                    output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                                0.25,   # Conf. Threshold.
                                                0.65, # IoU Threshold.
                                                nc=model.yaml['nc'], # Number of classes.
                                                nkpt=model.yaml['nkpt'], # Number of keypoints.
                                                kpt_label=True)
                    
                    output = output_to_keypoint(output_data)

                    im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                    im0 = im0.cpu().numpy().astype(np.uint8)
                    
                    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                    for i, pose in enumerate(output_data):  # detections per image
                    
                        if len(output_data):  #check if no pose
                            for c in pose[:, 5].unique(): # Print results
                                n = (pose[:, 5] == c).sum()  # detections per class
                                print("No of Objects in Current Frame : {}".format(n))
                            
                            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                                c = int(cls)  # integer class
                                kpts = pose[det_index, 6:]
                                print(kpts)
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                            line_thickness=line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                            orig_shape=im0.shape[:2])

                    
                    end_time = time.time()  #Calculatio for FPS
                    fps = 1 / (end_time - start_time)
                    total_fps += fps
                    frame_count += 1
                    
                    fps_list.append(total_fps) #append FPS in list
                    time_list.append(end_time - start_time) #append time in list
                    
                    # Stream results
                    if view_img:
                        cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                        cv2.waitKey(1)  # 1 millisecond

                    out.write(im0)  #writing the video frame

                else:
                    break

            cap.release()
            # cv2.destroyAllWindows()
            avg_fps = total_fps / frame_count
            print(f"Average FPS: {avg_fps:.3f}")
            
            #plot the comparision graph
            self.plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)



    

