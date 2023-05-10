## CUDA & SGX integration

This directory contains an example of building pipeline that integrates SGX and CUDA into one project.

> **Warning** The code in this directory only demonstrates the integration with the build system. Check [kernel_launcher](/kernel_launcher/) for actual implementation of the user communication protocol between CPU and GPU. Check directory [checksum](/checksum/) for implementation of the self-verifying checksum function.

### Installation instructions

1) Install SGX SDK (this may take around 30-60 minutes depending on the hardware). Follow installation instructions here (default configuration, Ubuntu 20.04):
https://github.com/intel/linux-sgx

2) Install NVCC
```
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

### Compilation instructions

Before compiling make sure that the environment variables for SGX SDK are set:
```
source ${sgx-sdk-install-path}/environment
```

Compile in simulation mode using:
```
make SGX_MODE=SIM
```

Run using:
```
./app
```

### End-to-end build inside container

```
podman run -it --rm -v ./:/SGX -w /SGX --security-opt label=disable docker.io/nvidia/cuda:11.8.0-devel-ubuntu20.04 /bin/bash -c "\
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y build-essential ocaml ocamlbuild automake autoconf libtool wget python-is-python3 libssl-dev git cmake perl && \
    apt-get install -y libssl-dev libcurl4-openssl-dev protobuf-compiler libprotobuf-dev debhelper cmake reprepro unzip pkgconf libboost-dev libboost-system-dev libboost-thread-dev lsb-release libsystemd0 && \
    git clone -b sgx_2.19 https://github.com/intel/linux-sgx.git ; \
    cd linux-sgx && make preparation && \
    cp external/toolset/ubuntu20.04/* /usr/local/bin && \
    make sdk_install_pkg && \
    cd /SGX && \
    (echo yes | ./linux/installer/bin/sgx_linux_x64_sdk_2.19.100.3.bin) && \
    source sgxsdk/environment && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH && \
    export LIBRARY_PATH=/usr/local/cuda/lib64/:$LIBRARY_PATH && \
    make SGX_MODE=SIM && \
    exec bash"
```

After build `./app` is supposed to be functional only if GPU-passthrough in container is enabled. Configure the podman/docker call to match the local installation. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html