### Data transfer and communication protocols.

This directory contains a demonstration of the custom data transfer and communication protocols that protect code and data during transit between the SGX enclave and GPU.

Here we show one iteration of inference using the example of a multi-layer perceptron consisting of two fully connected (linear) layers and one rectified linear layer (ReLU) between them.

A list of targets can be found in [Makefile](Makefile).

* Run `make base` to build a base example with default protocol and no protection.
* Run `make prot` to build a secure implementation.
* Run `make perf.txt perf_kernel_and_copy.txt` to collect data for evaluation of overheads of copies and kernel launches. It may take around 15 minutes to complete. Use `python plot_overhead.py` and `python plot_perf.py` to build plots using collected data (plotting requires common libraries to be available in current python installation: `conda install python==3.9 matplotlib seaborn pandas numpy`).