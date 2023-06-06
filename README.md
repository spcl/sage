# SAGE: Software-based Attestation for GPU Execution

## This repository contains

### The proof-of-concept implementation of SAGE

#### Checksum benchmark [checksum/main/readme.md](checksum/main/readme.md)

It is designed to perform evaluation of checksum runtime with different configurations of checksum implementation (number of instructions, presence of "adversary NOP") and obtain the achieved % of peak performance. The timings obtained by running experiments from that directory correspond to the table "Evaluation of checksum implementations" from the paper.

#### Integration with SGX [SGX/README.md](SGX/README.md)

This is proof of concept implementation of compilation pipeline to illustrate the build process of SAGE as SGX application.

#### Communication protocol between CPU and GPU [kernel_launcher/readme.md](kernel_launcher/readme.md)

This example is designed to evaluate the overheads of kernel launches and memory transfers added by SAGE. The results in this directory correspond to the figures "Overheads of data transfers and kernel launches" from the paper.

### TRNG

Race-condition-based true random number generation on GPU: [trng/readme.md](trng/readme.md)

This directory contains TRNG implementation and the setup for running statistical tests to support the results in the section "Random Number Generation on GPUs" of the paper.

### Proofs

Tamarin proofs for the modified version of SAKE: [proofs/README.md](proofs/README.md)

This directory contains the code to perform formal verification of modified SAKE protocol, as explained in the section "Formal Verification of Modified SAKE" of the paper.

### Ampere reverse engineering

tools to assist the reverse engineering of the Ampere architecture: [checksum/instr_decode/readme.md](checksum/instr_decode/readme.md)

## Prerequisites

### Software

* `podman` (or `docker`). While it is not a requirement for SAGE implementation, we provide the most detailed instructions to reproduce our experiments in containerized environment.
* CUDA 11. (CUDA 12 mostly works but is known to be not fully compatible with certain experiment setups and our reverse-engineered environment, expect occasional hanging of the implementation or mismatch between obtained and expected runtimes).
* Install SGX SDK
* Python 3.9 with `matplotlib`, `seaborn`, `pandas`, and `numpy` packages is neccessary for obtaining the plots.

### Hardware

* NVIDIA A100 GPU
* (Optional) SGX support in CPU. If not available, can be runned in simulation mode.

## Paper

https://arxiv.org/abs/2209.03125

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2209.03125,
  doi = {10.48550/ARXIV.2209.03125},
  url = {https://arxiv.org/abs/2209.03125},
  author = {Ivanov, Andrei and Rothenberger, Benjamin and Dethise, Arnaud and Canini, Marco and Hoefler, Torsten and Perrig, Adrian},
  keywords = {Cryptography and Security (cs.CR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SAGE: Software-based Attestation for GPU Execution},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
