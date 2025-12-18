FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    ibverbs-utils \
    rdma-core \
    libibverbs-dev \
    librdmacm-dev \
    libibumad-dev \
    perftest \
    net-tools \
    iproute2 \
    ethtool \
    pciutils \
    numactl \
    libnuma-dev \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMPI 5.0.x
RUN cd /tmp && \
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz && \
    tar xzf openmpi-5.0.5.tar.gz && \
    cd openmpi-5.0.5 && \
    ./configure --prefix=/usr/local/openmpi \
    --with-cuda=/usr/local/cuda \
    --with-verbs \
    --enable-mpirun-prefix-by-default && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/openmpi-5.0.5*

ENV PATH=/usr/local/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

# Install NCCL 2.22 (using NVIDIA's package)
RUN cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y libnccl2 libnccl-dev && \
    rm -f cuda-keyring_1.1-1_all.deb

# Clone and build nccl-tests
RUN cd /opt && \
    git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 MPI_HOME=/usr/local/openmpi CUDA_HOME=/usr/local/cuda

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Create mock RDMA devices and network interfaces
# These scripts simulate the presence of RDMA hardware
COPY setup_mock_rdma.sh /opt/setup_mock_rdma.sh
RUN chmod +x /opt/setup_mock_rdma.sh

# Create workspace
WORKDIR /workspace

# Set up intentionally broken NCCL configuration
# This is the starting point that needs to be fixed
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=lo
ENV NCCL_NET=Socket
ENV NCCL_P2P_DISABLE=0
ENV NCCL_SHM_DISABLE=0

# Copy initial files
COPY baseline_benchmark.sh /workspace/baseline_benchmark.sh
COPY pytorch_ddp_test.py /workspace/pytorch_ddp_test.py
COPY optimization_report_template.md /workspace/optimization_report_template.md
COPY mock_ibv_devinfo.sh /usr/local/bin/ibv_devinfo
COPY mock_nvidia_smi.sh /usr/local/bin/nvidia-smi-topo

RUN chmod +x /workspace/baseline_benchmark.sh && \
    chmod +x /usr/local/bin/ibv_devinfo && \
    chmod +x /usr/local/bin/nvidia-smi-topo

# Initialize the mock RDMA environment on container start
RUN /opt/setup_mock_rdma.sh

CMD ["/bin/bash"]
