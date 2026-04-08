# Select the base image (`devel` includes headers and is suitable for development and compilation)
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Update package metadata and install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    python3.8-dev \
    python3-setuptools \
    python3-distro \
    gnupg2 \
    curl \
    wget \
    sudo \
    tmux \
    ca-certificates

# Link `python` and `pip` to Python 3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 2. install cuDNN 8.6.0
# cuDNN 8.6.0 is built for CUDA 11.8 but compatible with CUDA 11.x
RUN apt-get install -y --no-install-recommends \
    libcudnn8=8.6.0.163-1+cuda11.8 \
    libcudnn8-dev=8.6.0.163-1+cuda11.8 \
    && apt-mark hold libcudnn8 libcudnn8-dev

# 3. Install TensorRT 8.5.3
# TensorRT 8.5.3 is built for CUDA 11.8 but compatible with CUDA 11.x
RUN apt-get install -y --no-install-recommends \
    libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8 \
    libnvinfer-plugin-dev=8.5.3-1+cuda11.8 \
    libnvparsers8=8.5.3-1+cuda11.8 \
    libnvparsers-dev=8.5.3-1+cuda11.8 \
    libnvonnxparsers8=8.5.3-1+cuda11.8 \
    libnvonnxparsers-dev=8.5.3-1+cuda11.8 \
    python3-libnvinfer=8.5.3-1+cuda11.8 \
    && apt-mark hold libnvinfer8 libnvinfer-dev python3-libnvinfer

# 4. Install ROS 2
# keyboard-configuration and tzdata are required to avoid interactive prompts during installation
RUN apt-get install -y debconf && \
    echo 'keyboard-configuration keyboard-configuration/layoutcode string us' | debconf-set-selections && \
    echo "keyboard-configuration keyboard-configuration/modelcode string pc105" | debconf-set-selections && \
    echo "tzdata tzdata/Areas string Asia" | debconf-set-selections && \
    echo "tzdata tzdata/Zones/Asia string Hong_Kong" | debconf-set-selections

RUN apt-get install -y tzdata keyboard-configuration wget

# Install ROS 2 Foxy with Ubuntu 20.04
# ROS 2 Foxy is the Tier 1 ROS 2 distribution for Ubuntu 20.04 and Python 3.8
# ROS 2 Humble is the Tier 3 ROS 2 distribution for Ubuntu 20.04 and Python 3.8
# If you want to install ROS 2 Humble, it is recommended to use Ubuntu 22.04 and Python 3.10 instead for better compatibility and support
# For non-supported ROS 2 distributions, please install with the ROS 2 source rather than the other mirror sources
RUN wget http://fishros.com/install -O fishros && \
    chmod 777 ./fishros && \
    printf "1\n1\n2\n1\n5\n1\n1" | ./fishros

# 5. Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Start in bash by default
CMD ["/bin/bash"]
