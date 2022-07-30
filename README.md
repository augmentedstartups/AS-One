# ASOne
## Docker Installation Instructions For Windows
#### Table of Contents  
- [System Requirements](#system-requirements)  
- [Enable WSL 2 Feature](#enable-wsl-2-feature)
- [Docker Installation for Windows Systems](#docker-installation-for-windows-systems)
- [Setting up detectron2](#setting-up-detectron2)
- [Setting Environment Variable and Configuring Devices](#setting-environment-variable-and-configuring-devices)
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
**Note**

After downloading [Docker Desktop Installer.exe](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe), run the following command in `cmd` to <br/> 
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

## Setting Environment Variable and Configuring Devices
1. Install chocolatey (Command line application installer). Open command prompt as administrator and run the following command.
```
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
```
2. Install VcXsrv (Windows X Server) Tool
```
choco install vcxsrv
```
3. 
- Open XLaunch and select Multiple windows
- Select the option Start no client
- In Extra Settings, select the option 
  1. Clipboard
  2. Primary Selection
  3. Native opengl
- Save configuration file for later use
4. Open PowerShell and run the command to check ipv4 if you get multiple addresses then copy the first one.
```
ipconfig | ? { $_ -match 'Ipv4' }
```
5. Run the following command in cmd and type like this  
` setx DISPLAY 192.168.168.128:0.0`
```
setx DISPLAY <ipv4>:0.0
```

6. 
 - Open (cam2ip.exe)[cam2ip-1.6-64bit-cv/cam2ip.exe] and select extract all
 - Open cam2ip.exe and see "Listening on: 56000" 
 - IP stream will be on `http://localhost:56000/mjpeg`
## Setting up Docker

1. Build docker contanier <br/>
   
```
docker build -t [IMAGE_NAME]:[TAG]
```

- `IMAGE_NAME` = Assign a name to image
- `TAG` = Assign a tag to image

A valid command will be
```
docker build -t yolov5:latest
```
2. Run Docker Contaner

```
docker run --gpus all --env DISPLAY=%DISPLAY% --net=host -v [PATH_TO_LOCAL_DIR]:/workspace/  -it [IMAGE_NAME]:[TAG]
```

- `PATH_TO_LOCAL_DIR` = Path to detectron2 directory or use `${PWD}` if already in that directory

Example: `docker run --gpus all --env DISPLAY=%DISPLAY% --net=host -v C:\asone\detectron2\:/workspace/ -it asone:first` <br/>


or run the following command to run docker.
```
docker compose run windows
```

3. In Docker terminal run demo.py file

```
python demo/demo.py --input [PATH_TO_TEST_IMAGE]  --output [PATH_TO_OUTPUT_IMAGE] \
  --opts MODEL.DEVICE [DEVICE] \ 
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

- `PATH_TO_TEST_IMAGE` = Path of test image
- `PATH_TO_OUTPUT_IMAGE` = Path of Results
- `DEVICE` = device to use i.e. cpu or gpu
