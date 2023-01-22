Installation instructions

1) Install SGX SDK. Follow installation instructions here (default configuration, Ubuntu 20.04):
https://github.com/intel/linux-sgx

2) Install NVCC
sudo apt update
sudo apt install nvidia-cuda-toolkit

Compilation instructions:

Before compiling make sure that the environment variables for SGX SDK are set:
source ${sgx-sdk-install-path}/environment

Compile in simulation mode using:
make SGX_MODE=SIM

Run using:
./app