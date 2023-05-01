FROM python:3.8

RUN apt-get update 
RUN pip install pip --upgrade

COPY . ./voxelmorph/
WORKDIR /voxelmorph

RUN pip install voxelmorph
RUN pip install numpy==1.20
RUN pip install tensorflow
RUN pip install datalad


ENTRYPOINT [ "/bin/bash"]