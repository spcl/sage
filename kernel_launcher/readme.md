### Data transfer and communication protocols.

This directory contains a demonstration of the custom data transfer and communication protocols that protect code and data during transit between the SGX enclave and GPU.

Here we show one iteration of inference using the example of a multi-layer perceptron consisting of two fully connected (linear) layers and one rectified linear layer (ReLU) between them.

A list of targets can be found in [Makefile](Makefile).

* Run `make base` to build a base example with default protocol and no protection.
* Run `make prot` to build a secure implementation.
