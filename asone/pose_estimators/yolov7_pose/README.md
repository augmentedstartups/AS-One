# yolov7-pose-estimation

### New Features
- Added Support for Comparision of (FPS & Time) Graph
- How to run Code in Google Colab
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported
### Coming Soon
- Development of streamlit dashboard for Pose-Estimation

### Steps to run Code
- If you are using google colab then you will first need to mount the drive with mentioned command first, <b>(Windows or Linux users)</b> both can skip this step.
```
from google.colab import drive
drive.mount("/content/drive")
```
- Clone the repository.
```
git clone https://github.com/RizwanMunawar/yolov7-pose-estimation.git
```

- Goto the cloned folder.
```
cd yolov7-pose-estimation
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv psestenv
source psestenv/bin/activate

### For Window Users
python3 -m venv psestenv
cd psestenv
cd Scripts
activate
cd ..
cd ..
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.

```
pip install -r requirements.txt
```

- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}


- Run the code with mentioned command below.
```
python pose-estimate.py

#if you want to change source file
python pose-estimate.py --source "your custom video.mp4"

#For CPU
python pose-estimate.py --source "your custom video.mp4" --device cpu

#For GPU
python pose-estimate.py --source "your custom video.mp4" --device 0

#For View-Image
python pose-estimate.py --source "your custom video.mp4" --device 0 --view-img

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python pose-estimate.py --source "your IP Camera Stream URL" --device 0 --view-img

#For WebCam
python pose-estimate.py --source 0 --view-img

#For External Camera
python pose-estimate.py --source 1 --view-img
```

- Output file will be created in the working directory with name <b>["your-file-name-without-extension"+"_keypoint.mp4"]</b>

#### RESULTS

<table>
  <tr>
    <td>Football Match Pose-Estimation</td>
     <td>Cricket Match Pose-Estimation</td>
     <td>FPS and Time Comparision</td>
     <td>Live Stream Pose-Estimation</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/185089411-3f9ae391-ec23-4ca2-aba0-abf3c9991050.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185228806-4ba62e7a-12ef-4965-a44a-6b5ba9a3bf28.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185324844-20ce3d48-f5f5-4a17-8b62-9b51ab02a716.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185587159-6643529c-7840-48d6-ae1d-2d7c27d417ab.png" width=640 height =180></td>
  </tr>
 </table>

#### References
- https://github.com/WongKinYiu/yolov7
- https://github.com/augmentedstartups/yolov7
- https://github.com/augmentedstartups
- https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
- https://github.com/ultralytics/yolov5

#### My Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
