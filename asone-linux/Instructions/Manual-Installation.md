# ASOne


# Docker Manual Installation

## Ubuntu


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

4. Install `nvidia-docker` to allow docker interact with GPU.

```
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

5. Give docker access to devices.

```
sudo xhost +local:docker

sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker

```