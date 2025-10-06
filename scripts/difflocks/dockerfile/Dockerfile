# FROM nvcr.io/nvidia/pytorch:22.08-py3

# for torch 2.0 it needs nvidia driver>=530
#chek what is contained in each version here: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html
# FROM nvcr.io/nvidia/pytorch:23.03-py3
FROM nvcr.io/nvidia/pytorch:23.10-py3


# Arguments
ARG user
ARG uid
ARG home
ARG workspace
ARG shell

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video,utility,compute

ENV DEBIAN_FRONTEND noninteractive

# Before any libraries are installed, make sure everyone knows about cuda-aware ompi
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
ENV PATH="${PATH}:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ucx/lib"

#Make SSH keys work inside the container
RUN  echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config
RUN  echo "    IdentityFile ~/.ssh/id_ed25519" >> /etc/ssh/ssh_config

##### PACKAGES
RUN apt-get -y update && apt-get install -y libassimp-dev libjpeg-dev libpng-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libgl1-mesa-dev libegl1-mesa-dev sudo


# Dependencies for glvnd and X11. -> https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1
# libboost-all-dev is already installed when installing libpcl-dev
RUN apt-get -y update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libglx0 \
    libxext6 \
    libx11-6 \
    libglfw3 \
    libglfw3-dev \
    libpcl-dev \
    tmux \
    zip\
    locate\
    libboost-program-options-dev\
    libboost-filesystem-dev\
    libboost-graph-dev\
    libboost-system-dev\
    libboost-test-dev\
    libeigen3-dev\
    libsuitesparse-dev\
    libfreeimage-dev\
    libmetis-dev\
    libgoogle-glog-dev\
    libgflags-dev\
    libglew-dev\
    qtbase5-dev\
    libqt5opengl5-dev\
    libcgal-dev\
    libatlas-base-dev\
    libsuitesparse-dev\
    libomp-dev\
    libomp5\
    imagemagick


# Basic Utilities (dirmngr and gpg-agent is needed so that apt-get adv works)
RUN apt-get -y update && apt-get install -y --no-install-recommends sudo ssh tmux git meld cmake cmake-curses-gui htop vim iputils-ping apt-utils apt-transport-https software-properties-common wget build-essential gdb valgrind silversearcher-ag curl dirmngr gpg-agent checkinstall locate baobab ninja-build libncurses5-dev gimp pkg-config clang

#general deps
RUN python3 -m pip install tensorboard==2.11.0 natsort piq scikit-learn scikit-image torchnet wandb hjson
# we need to downgrade it because tensorboard has an issue with newer versions https://exerror.com/attributeerror-module-setuptools-_distutils-has-no-attribute-version/
RUN python3 -m pip install setuptools==59.5.0
RUN pip3 install --upgrade huggingface_hub


#deps for easypbr
RUN apt-get install -y libglfw3-dev libboost-all-dev libeigen3-dev libpcl-dev libopencv-dev

#blender
RUN sudo apt-get update && sudo apt-get install -y llvm-dev blender

#Things that are nice to have in order to run some applications
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends libtbb-dev libglm-dev libkqueue-dev libpcap-dev network-manager net-tools meshlab ffmpeg unzip git-lfs mesa-utils mesa-utils-extra libnvidia-egl-wayland1 libegl1 libgles2-mesa v4l-utils libv4l-dev

#qt 5 which is needed for some packages
RUN sudo apt-get install -y qt5-qmake qtbase5-dev-tools

#more python packages
RUN python3 -m pip install maturin smplx[all]

#diffusers, transformers etc
RUN python3 -m pip install --upgrade diffusers[torch] transformers

#for helix core (p4v)
RUN sudo apt-get install -y libxkbfile-dev
RUN sudo apt install -y libxcb-xinerama0-dev libxcb-cursor-dev


RUN sudo apt install -y python3.10-venv

#bunch of python libs that are nice to have
RUN python3 -m pip install opencv-contrib-python gdown timm watchdog
#to solve this:
# https://github.com/facebookresearch/nougat/issues/40
RUN python3 -m pip install opencv-python==4.8.0.74
#Things specific to smirk
RUN python3 -m pip install mediapipe==0.10.10 chumpy==0.70


#poetry
# https://github.com/orgs/python-poetry/discussions/1879#discussioncomment-7284113
# ENV  POETRY_VIRTUALENVS_CREATE=false #if we set create virtual_env to false we cannot pull from the model registry anymore for some reason
RUN curl -sSL https://install.python-poetry.org | python3 -
# RUN export PATH="$HOME/.local/bin:$PATH"
ENV PATH="${home}/.local/bin:$PATH"
# RUN echo $PATH
# #https://medium.com/opsops/using-poetry-within-docker-container-without-venv-f8f07c9deba3
# RUN /bin/true\
#     && ${home}/.local/bin/poetry config virtualenvs.create false \
#     && ${home}/.local/bin/poetry install --no-interaction \
#     && rm -rf /root/.cache/pypoetry


#for difflocks we just use conda
# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy
# Set PATH to include Conda
ENV PATH="/opt/conda/bin:$PATH"

RUN conda --version

 RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name difflocks_env python=3.8 
ENV PATH /opt/conda/envs/difflocks_env/bin:$PATH
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "difflocks_env", "/bin/bash", "-c"]
RUN conda init bash
RUN conda info
RUN conda config --add channels conda-forge
RUN conda install -y pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
RUN conda install -y  pip packaging ninja datasets transformers jsonmerge torchdiffeq torchsde accelerate scikit-image trimesh igl tensorboard 
RUN pip install dctorch clean-fid clip-anytorch watchdog mediapipe==0.10.10 einops pynput
RUN conda install -y  conda-forge::opencv
RUN conda install -y  -c conda-forge diffusers
RUN echo fs.inotify.max_user_watches=655360 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p 
RUN pip install natten==0.17.1+torch240cu124 -f https://shi-labs.com/natten/wheels/
RUN MAX_JOBS=16 pip install flash-attn --no-build-isolation
    # The code to run when container is started:
# ENTRYPOINT ["conda", "activate",  "difflocks_env"]

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Activate the environment by default when running the container
CMD ["bash", "-c", "source activate difflocks_env && exec bash"]





##### CLEANUP
#Cleanup
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update

# Make SSH available
EXPOSE 22
EXPOSE 42421
# TensorBoard https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 6006
# IPython https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 8888
#for unreal remote execution https://www.youtube.com/watch?v=VcD1BOo2VGs
EXPOSE 6766
EXPOSE 9998
EXPOSE 8998

# Mount the user's home directory
VOLUME "${home}"

#Intel vtune and  MKL
VOLUME "/opt/intel"

# Clone user into docker image and set up X11 sharing
RUN \
  echo "${user}:x:${uid}:${uid}:${user},,,:${home}:${shell}" >> /etc/passwd && \
  echo "${user}:x:${uid}:" >> /etc/group && \
  echo "${user} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${user}" && \
  chmod 0440 "/etc/sudoers.d/${user}"


#Set the user in the dialaout group so that the gps works
RUN usermod -a -G dialout ${user} && usermod -a -G video ${user} && usermod -a -G audio ${user} && usermod -a -G audio ${user}


#own some stuff because they were created by the root user in the docker container
# RUN chown -R ${user} /opt
# USER ${user}
# USER root

# Switch to user
USER "${user}"
# This is required for sharing Xauthority
ENV QT_X11_NO_MITSHM=1
ENV CATKIN_TOPLEVEL_WS="${workspace}/devel"
# Switch to the workspace
WORKDIR ${workspace}
# WORKDIR /workspace

# switch to interactive
ENV DEBIAN_FRONTEND teletype
