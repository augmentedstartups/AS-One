# ASOne

#### Table of Contents  
- [Docker Intallation](#docker-installation)
  - [Ubuntu](#ubuntu)
    - [Install Using Shell Script](#install-using-shell-script)
    - [Manual Installation](#manuall-install)
- [Docker Demo](#docker-demo)
  - [Docker Build](#docker-build)
  - [Docker Demo](#docker-demo-1)
    - [Setup Detectron](#setup-detectron)
    - [Run Demo](#run-demo)

# Docker Installation
## Ubuntu

#### Install using Shell Script

```
chmod a+x docker-installation.sh
./docker-installation.sh 
```

#### Manuall Install
1. Run following command to remove all old versions on docker

```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

2. Set up Repository

  - Update the apt package index and install packages to allow apt to use a repository over HTTPS:

```
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```
  - Add Docker’s official GPG key:

```
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

  - Use the following command to set up the repository:

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
3. Install Docker Engine

  - Update the apt package index, and install the latest version of Docker Engine, containerd, and Docker Compose:

```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

## Windows
#### Table of Contents  
- [System Requirements](#system-requirements)  
- [WSL Installation](#wsl-installation)
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

- Download and install the [Linux kernel update package](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)

### WSL Installation
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
  3. Download the [Standalone WSL 2.0 Linux Kernel Update](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi) and install it. <br/>
  4. Download the [Linux distribution from Microsoft Store](https://apps.microsoft.com/store/detail/ubuntu-20044-lts/9MTTCL66CPXJ) and install it. After installing it, type UNIX Username and Password.<br/>
5. Run the following commands in PowerShell
```
wsl --install
```
The above command only works if WSL is not installed at all, if you run wsl --install and see the 
WSL help text then run this command to check online availabe distributions:
```
wsl --list --online
```
Or run this command
```
wsl -l --all
```
6. To install additional Linux distributions after the initial install, you may also use the following
   command and you can find the distribution name by running the command above.
```
wsl --install -d ubuntu-20.04
```
7. To check the verion of wsl
```
wsl -l -v
```
8. To upgrade or downgrade the version=2 or version=1
```
wsl --set-version ubuntu-20.04 2
```

### Docker Installation
(Supported for Windows 10 and 11 only) <br/>
**Note**: It is recommended to use WSL2 as backend. <br/>

#### Steps to Install Docker from Command line on Windows:
-------------------------------------------------------------------------------------------------------------

After downloading [Docker Desktop Installer.exe](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe), run the following command in a terminal to <br/> 
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



# Docker Demo

## Docker Build

#### Build docker contanier

```
docker build -t [IMAGE_NAME]:[TAG]
```

  - `IMAGE_NAME` = Asign a name to image
  - `TAG` = Asign a tag to image

## Docker Demo

#### Setting Up detectron2

1. Clone the Repo
```
git clone https://github.com/facebookresearch/detectron2.git
```
2. Goto the detectron2 directory
```
cd detectron2
```
3. Download some sample images in this folder


#### Run Demo

1. Run Docker Contaner

```
docker run --gpus all --env="DISPLAY" --net=host -v [PATH_TO_LOCAL_DIR]:/workspace/  -it [IMAGE_NAME]:[TAG]
```
  - `PATH_TO_LOCAL_DIR` = Path to detectron2 directory or use `${PWD}` if already in that directory

2. In Docker terminal run demo.py file

```
python demo/demo.py --input [PATH_TO_TEST_IMAGE]  --output [PATH_TO_OUTPUT_IMAGE] \
  --opts MODEL.DEVICE [DEVICE] \ 
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

- `PATH_TO_TEST_IMAGE` = Path of test image
- `PATH_TO_OUTPUT_IMAGE` = Path of Results
- `DEVICE` = device to use i.e. cpu or gpu
