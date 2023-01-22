## CUDA & SGX integration

This directory contains an example of building pipeline that integrates SGX and CUDA into one project.

> **Warning** The code in this directory only demonstrates the integration with the build system. Check [kernel_launcher](/kernel_launcher/) for actual implementation of the user communication protocol between CPU and GPU. Check directory [checksum](/checksum/) for implementation of the self-verifying checksum function.

### Installation instructions

1) Install SGX SDK. Follow installation instructions here (default configuration, Ubuntu 20.04):
https://github.com/intel/linux-sgx

2) Install NVCC
sudo apt update
sudo apt install nvidia-cuda-toolkit

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