# DockerFile taken from Verou repository: https://github.com/edf-hpc/verrou/blob/master/docker/Dockerfile
# Build stage
ARG IMAGE_BASE=ubuntu:20.04

ARG VALGRIND_VERSION=3.20.0
ARG VALGRIND_BRANCH=VALGRIND_3_20_0
ARG VERROU_BRANCH=synchroLib-v2.4.0
ARG NUM_THREADS=6

#Build Stage
FROM ${IMAGE_BASE} AS build

ARG VALGRIND_VERSION
ARG VALGRIND_BRANCH
ARG VERROU_BRANCH
ARG NUM_THREADS

RUN apt-get update \
  &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --reinstall ca-certificates \
  && update-ca-certificates \
  &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git build-essential automake python3 libc6-dbg nano \
  && rm -fr /var/lib/apt/lists/*

RUN mkdir -p /build
WORKDIR /build

RUN git clone --branch=${VALGRIND_BRANCH} --single-branch --depth=1 git://sourceware.org/git/valgrind.git valgrind-${VALGRIND_VERSION}+verrou-dev

WORKDIR /build/valgrind-${VALGRIND_VERSION}+verrou-dev

RUN git clone --branch=${VERROU_BRANCH} --single-branch --depth=1 https://github.com/yohanchatelain/verrou.git verrou \
  && patch -p1 <verrou/valgrind.diff

RUN  ./autogen.sh \
  && ./configure --enable-only64bit --enable-verrou-fma --prefix=/opt/valgrind-${VALGRIND_VERSION}+verrou-dev \
  && make -j ${NUM_THREADS}\
  && make install

RUN cd verrou/Interlibmath && make
RUN cd verrou/synchroLib && make

# Final stage
FROM ${IMAGE_BASE} AS common
ARG VALGRIND_VERSION

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 libc6-dbg python3-numpy  python3-matplotlib texlive-latex-extra texlive-fonts-recommended dvipng cm-super\
  && rm -fr /var/lib/apt/lists/*

COPY --from=build /opt/valgrind-${VALGRIND_VERSION}+verrou-dev /opt/valgrind-${VALGRIND_VERSION}+verrou-dev

RUN cp /opt/valgrind-${VALGRIND_VERSION}+verrou-dev/env.sh /etc/profile.d/99-valgrind+verrou.sh \
  && chmod 644 /etc/profile.d/99-valgrind+verrou.sh


FROM python:3.8 as default
ARG VALGRIND_VERSION

RUN apt-get update 

RUN pip install pip --upgrade

COPY --from=common /opt/valgrind-${VALGRIND_VERSION}+verrou-dev /opt/valgrind-${VALGRIND_VERSION}+verrou-dev

# Update pip and install Python packages
RUN python3 -m pip install --upgrade pip
RUN pip install voxelmorph
RUN pip install numpy==1.20
RUN pip install tensorflow
RUN pip install click
RUN pip install protobuf==3.20
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# RUN pip install pycrunch-trace
# RUN pip install datalad

# Copies valgrind from previous stage
COPY --from=common /opt/valgrind-${VALGRIND_VERSION}+verrou-dev /opt/valgrind-${VALGRIND_VERSION}+verrou-dev
COPY --from=common /lib/x86_64-linux-gnu/libquadmath.so.0 /lib/x86_64-linux-gnu/libquadmath.so.0

COPY . ./voxelmorph/
COPY ./voxelmorph/py/utils.py /usr/local/lib/python3.8/site-packages/voxelmorph/py
WORKDIR /voxelmorph
COPY --from=build /build/valgrind-3.20.0+verrou-dev/verrou/Interlibmath/interlibmath.so ./interlibmath.so
COPY --from=build /build/valgrind-3.20.0+verrou-dev/verrou/synchroLib /voxelmorph/synchroLib
RUN chmod -R o+w /voxelmorph


# Sources the verrou commands
ENV PATH=/opt/valgrind-3.20.0+verrou-dev/bin:${PATH}
ENV PYTHONPATH=/opt/valgrind-3.20.0+verrou-dev/lib/python3.8/site-packages:/opt/valgrind-3.20.0+verrou-dev:/lib/python3.8/site-packages:
ENV MANPATH=/opt/valgrind-3.20.0+verrou-dev/share/man:${MANPATH}
ENV CPATH=/opt/valgrind-3.20.0+verrou-dev/include:${CPATH}
ENV VERROU_COMPILED_WITH_FMA=yes
ENV VERROU_COMPILED_WITH_QUAD=yes


# FROM gcr.io/bazel-public/bazel:latest as TCMalloc
# #Install TCMalloc
# WORKDIR /tcmalloc/source/
# RUN git clone https://github.com/google/tcmalloc.git .
# RUN bazel test //tcmalloc/...

RUN  apt-get -y install google-perftools

ENTRYPOINT ["/bin/bash"]
