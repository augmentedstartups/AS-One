FROM pytorch/pytorch:latest

# Set Time Zone to prevent issues for installing some apt packages
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install apt packages
RUN apt-get update -y 
RUN apt-get install git gcc g++ python3-opencv  -y 

# install python libraries library
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip3 install opencv-python

# Entry Point
CMD /bin/bash
