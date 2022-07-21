# ASOne

## Docker Installation for Windows Systems
(Supported for Windows 10 and 11 only)
### Steps to Install Docker Interactively on Windows:
**Note**: It is recommended to use WSL2 as backend. <br/>

1. Kindly read the requirements section carefully, Given below!
2. Go to [Docker Official Website](https://docs.docker.com/desktop/install/windows-install/)
3. Download the installer file (Docker Desktop Installer.exe) from the website
4. Open the installer and Use WSL 2 instead of Hyper-V option on the Configuration Page is Selected
5. If your admin account is different to your user account, you must add the user to the <br/>
   docker-users group. Run Computer Management as an administrator and navigate to Local Users and <br/> 
   Groups > Groups > docker-users. Right-click to add the user to the group. Log out and log back in <br/> 
   for the changes to take effect.
6. And You Are All Done!
### Steps to Install Docker from Command line on Windows:

After downloading Docker Desktop Installer.exe, run the following command in a terminal to <br/> 
install Docker Desktop:
```
"Docker Desktop Installer.exe" install
```
If you’re using PowerShell you should run it as:
```
Start-Process '.\win\build\Docker Desktop Installer.exe' -Wait install
```
And the install command accepts the following flags:

-  --quiet: Suppresses information output when running the installer
-  --accept-license: Accepts the [Docker Subscription Service Agreement](https://www.docker.com/legal/    docker-subscription-service-agreement/) now, rather than requiring it to be accepted when the application is first run.

-  --allowed-org=<org name>: requires the user to sign in and be part of the specified Docker Hub organization when running the application

- --backend=<backend name>: selects the backend to use for Docker Desktop, hyper-v or wsl-2 (default)

If your admin account is different to your user account, you must add the user to the docker-users group:
```
net localgroup docker-users <user> /add
```







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

- Download and install the [Linux kernel update package](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)








## Setup detectron2
1. Clone the Repo
```
git clone https://github.com/facebookresearch/detectron2.git
```
2. Goto the detectron2 directory
```
cd detectron2
```
3. Download some sample images in this folder

## Docker Setup

1. Build docker contanier
```
docker build -t [IMAGE_NAME]:[TAG]
```

- IMAGE_NAME = Asign a name to image
- TAG = Asign a tag to image

2. Run Docker Contaner

```
docker run --gpus all --env="DISPLAY" --net=host -v [PATH_TO_LOCAL_DIR]:/workbase/  -it [IMAGE_NAME]:[TAG]
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
