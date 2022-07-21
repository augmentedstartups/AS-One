#!/bin/bash

echo "Removing previous verions of dockers"
sudo apt-get remove docker docker-engine docker.io containerd runc -y

echo "Updating apt-package index..."
sudo apt-get update -y

echo "Installing apt required apt packages..."

sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y

echo "Adding Docker GPG Key..."

sudo mkdir -p /etc/apt/keyrings -y

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "Setting docker repository..."

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "Installing Docker Engine..."
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

sudo docker --version

echo "Docker Engine Installed Successfully!"
