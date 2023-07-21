FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        git \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        htop \
        sudo \
        unzip \
        vim \
        screen

RUN sudo apt install software-properties-common -y \
    && sudo add-apt-repository ppa:deadsnakes/ppa -y \
    && sudo apt install python3.10 -y \
    && sudo apt install python-is-python3 -y \
    && sudo apt install python3-pip -y
RUN sudo apt install python3.10-venv -y


# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
#     && unzip awscliv2.zip \
#     && sudo ./aws/install

# RUN git config --global user.email "hli113@stevens.edu" \
#     && git config --global user.name "Haohang Li" \
#     && git config --global --add safe.directory /workspaces/2023SummerLabeling

# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100 poetry
RUN pip install --default-timeout=100 poetry
