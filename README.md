# SAGE: Software-based Attestation for GPU Execution

## This repository contains

- the proof-of-concept implementation of SAGE
  - checksum benchmark [checksum/main/readme.md](checksum/main/readme.md)
  - integration with SGX [SGX/README.md](SGX/README.md)
  - communication protocol between CPU and GPU [kernel_launcher/readme.md](kernel_launcher/readme.md)
- race-condition-based true random number generation on GPU: [trng/readme.md](trng/readme.md)
- tamarin proofs for the modified version of SAKE: [proofs/README.md](proofs/README.md)
- tools to assist the reverse engineering of the Ampere architecture: [checksum/instr_decode/readme.md](checksum/instr_decode/readme.md)

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
