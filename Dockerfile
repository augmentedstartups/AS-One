FROM pytorch/pytorch:latest

# Set Time Zone to prevent issues for installing some apt packages
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install apt packages
RUN apt-get update -y 
RUN apt-get install git gcc \
        g++ python3-opencv \
        vim -y

RUN mkdir /app
WORKDIR /app 

ADD asone asone

ADD sample_videos sample_videos
ADD main.py main.py
# ADD demo.py demo.py

ADD setup.py setup.py
ADD requirements.txt requirements.txt


RUN pip3 install Cython numpy
RUN pip3 install cython-bbox
ADD pypi_README.md pypi_README.md

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install .


WORKDIR /workspace
# Entry Point
CMD /bin/bash
