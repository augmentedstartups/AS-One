#!/bin/bash

echo "[INFO]: Removing previous verions of dockers..."
echo "[INFO]: Removing previous verions of dockers..." > logs.txt
if sudo apt-get remove docker docker-engine docker.io containerd runc -y >> logs.txt; then
    echo "[INFO]: Previous docker removed successfully!"
    echo "[INFO]: Previous docker removed successfully!" >> logs.txt
fi

echo "[INFO]: Updating apt-package index..."
echo "[INFO]: Updating apt-package index..." >> logs.txt


if sudo apt-get update -y >> logs.txt; then
    echo "[INFO]: apt-package index updated successfuly!"
    echo "[INFO]: apt-package index updated successfuly!" >> logs.txt
else
    echo "[ERROR]: Error while updating apt-package index. Check logs.txt file for more info."
    echo "[ERROR]: Error while updating apt-package index." >> logs.txt
    # exit 1
fi

echo "[INFO]: Installing required apt packages..."
echo "[INFO]: Installing required apt packages..." >> logs.txt

if sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y ; then
    echo "[INFO]: Required apt packages installed successfully!"
    echo "[INFO]: Required apt packages installed successfully!" >> logs.txt
else
    echo "[ERROR]: Error installing required apt packages. Check logs.txt file for more info."
    echo "[ERROR]: Error installing required apt packages." >> logs.txt
    exit 1
fi

echo "[INFO]: Adding docker GPG key..."
echo "[INFO]: Adding docker GPG key..." >> logs.txt

sudo mkdir -p /etc/apt/keyrings

if curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg >> logs.txt;then
    echo "[INFO]: Docker GPG key added successfully!"
    echo "[INFO]: Docker GPG key added successfully!" >> logs.txt
else
    echo "[ERROR]: Error adding docker GPG key. Check logs.txt file for more info."
    echo "[ERROR]: Error adding docker GPG key." >> logs.txt
    exit 1
fi


echo "[INFO]: Setting docker repository..."
echo "[INFO]: Setting docker repository..." >> logs.txt
if echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null >> logs.txt; then
    echo "[INFO]: Docker repository setup done."
    echo "[INFO]: Docker repository setup done." >> logs.txt
else
    echo "[ERROR]: Error setting up docker repository. Check logs.txt file for more info."
    echo "[ERROR]: Error setting up docker repository." >> logs.txt
    exit 1
fi

echo "[INFO]: Installing Docker Engine..."
echo "[INFO]: Installing Docker Engine..." >> logs.txt

if sudo apt-get update -y >> logs.txt; then
    if sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y >> logs.txt; then
        if sudo docker --version; then 
            echo "[INFO]: Docker Engine instaleld successfully!"
            echo "[INFO]: Docker Engine instaleld successfully!" >> logs.txt
        fi
    else
        echo "[ERROR]: Error installing docker engine. Check logs.txt file for more info."
        echo "[ERROR]: Error installing docker engine." >> logs.txt
        exit 1
    fi
else
    echo "[ERROR]: Error updating apt packages. Check logs.txt file for more info."
    echo "[ERROR]: Error updating apt packages." >> logs.txt
    # exit 1
fi

echo "[INFO]: Adding docker to sudo group..."
echo "[INFO]: Adding docker to sudo group..." >> logs.txt
sudo xhost +local:docker

sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker

echo "[INFO]: Docker Installation and setup completed successfully!"
echo "[INFO]: Docker Installation and setup completed successfully!" >> logs.txt