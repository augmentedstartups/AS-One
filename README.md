# ASOne
## Docker Installation Instructions For Windows
#### Table of Contents  
- [System Requirements](#system-requirements)  
- [Enable WSL Feature](#enable-wsl-feature)
- [Docker Installation for Windows Systems](#docker-installation-for-windows-systems)
- [Setting up detectron2](#setting-up-detectron2)
- [Setting up Docker](#setting-up-docker)

### System Requirements
Windows machine must meet the following requirements to successfully install the docker:

**With WSL 2 backend** <br/>
Type **winver** in RUN to check the version of the installed windows.

- Windows 11 64-bit: Home or Pro version 21H2 or higher,\
  or Enterprise or Education version 21H2 or     higher
- Windows 10 64-bit: Home or Pro 21H1 (build 19043) or higher,\
  or Enterprise or Education 20H2 (build 19042) or higher
- Enable the WSL 2 feature on Windows. For detailed instructions,\
  refer to the [wsl installation](https://docs.microsoft.com/en-us/windows/wsl/install)

 **Hardware Requirements to run WSL 2 on Windows 10 or Windows 11** <br/>  

- 64-bit processor with Second Level Address Translation (SLAT)

- 4GB system RAM

- BIOS-level hardware virtualization support must be enabled in the \
  BIOS settings. For more information, see [Virtualization](https://docs.docker.com/desktop/troubleshoot/topics/)

## Enable WSL 2 Feature
**Follow the steps given below to install WSL(Windows subsystem Linux):**
Open PowerShell as administrator and run the commands given below:
1.  Enabling the Virtual Machine Platform
```
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
2. Enabling the Windows Subsystem for Linux
```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```
  Restart your system. <br/>
3. Download and install the [Linux kernel update package](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)
  
4. To check the verion of wsl run these commands in PowerShell
```
wsl -l -v
```
5. To upgrade or downgrade the version=2 or version=1
```
wsl --set-default-version 2
```

## Docker Installation for Windows Systems
(Supported for Windows 10 and 11 only) <br/>
**Note**: It is recommended to use WSL2 as backend. <br/>

### Steps to Install Docker from Command line on Windows:
-------------------------------------------------------------------------------------------------------------
**Note** **Kindly read the requirements section carefully!**

After downloading [Docker Desktop Installer.exe](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe), run the following command in command line window to <br/> 
Install Docker Desktop:
```
"Docker Desktop Installer.exe" install
```
If you’re using PowerShell you should run it as:
```
Start-Process 'path\to\Docker Desktop Installer.exe' -Wait install
```
And the install command accepts the following flags:

-  `--quiet`: Suppresses information output when running the installer
-  `--accept-license`: Accepts the [Docker Subscription Service Agreement](https://www.docker.com/legal/docker-subscription-service-agreement/) now, rather than requiring it to be accepted when the application is first run.
- `--backend=[backend name]`: selects the backend to use for Docker Desktop, hyper-v or wsl-2 (default)

If your admin account is different to your user account, you must add the user to the docker-users group:
```
net localgroup docker-users <user> /add
```
***And You Are All Done!***

## Setting up detectron2
1. Clone the Repo
```
git clone https://github.com/facebookresearch/detectron2.git
```
2. Goto the detectron2 directory
```
cd detectron2
```
3. Download some sample images in this folder

## Setting up Docker

1. Build docker contanier <br/>
Write a command like, "docker build -t yolov5:latest ."   
```
docker build -t [IMAGE_NAME]:[TAG] .
```

- IMAGE_NAME = Asign a name to image
- TAG = Asign a tag to image

2. Run Docker Contaner

```
docker run --gpus all --env="DISPLAY" --net=host -v [PATH_TO_LOCAL_DIR]:/workspace/  -it [IMAGE_NAME]:[TAG]
```
- PATH_TO_LOCAL_DIR = Path to detectron2 directory or use `${PWD}` if already in that directory

3. In Docker terminal run demo.py file

```
python demo/demo.py --input [PATH_TO_TEST_IMAGE]  --output [PATH_TO_OUTPUT_IMAGE] \
  --opts MODEL.DEVICE [DEVICE] \ 
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

- PATH_TO_TEST_IMAGE = Path of test image
- PATH_TO_OUTPUT_IMAGE = Path of Results
- DEVICE = device to use i.e. cpu or gpu
