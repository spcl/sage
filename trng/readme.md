# Quick example

Prerequisites: recent NVIDIA GPU (A100).

Write 10000 bytes of random binary data into file `random.bin`.

```
nvcc race_conditions_gpu_trng.cu -o trng
./trng random.bin 10000
```

# Randomness tests

It took me approximately 4 hours to get 50 MB of random data used in tests. The throughput is 4 KB/s on V100, so it can generate 256 bits in 8ms.

Generate 50 MB of random data (see command above) into `random.bin` (GPU required). After this step, GPU is not needed.
```
./trng random.bin 52428800
```

### TestU01

Our TRNG passes almost all tests that we tried from TestU01:
* pseudoDIEHARD (all)
* FIPS_140_2 (all)
* SmallCrush (all, except "Gap")
* Alphabit (all)
* Rabbit (all)

The output of the tests is saved in `race_conditions_gpu_trng_TestU01_result.txt`.

The only test that fails (Gap) seems to require a larger chunk of random data than what I experimented with (>50 MB).

##### Manual setup

* Download and compile TestU01 https://github.com/umontreal-simul/TestU01-2009/. 
* Compile TestU01 runner `gcc testu01.c -o t01` 
* Run it `./t01 random.bin | tee output.txt`
* Postprocess output `sed -n '/Summary/,/End/p' output.txt`. Output should match [race_conditions_gpu_trng_TestU01_result.txt](race_conditions_gpu_trng_TestU01_result.txt).

##### Run in container

```
podman run -it --rm -v ./:/trng -w /trng --security-opt label=disable ubuntu:22.04 /bin/bash -c "\
    apt-get update && apt-get install -y gcc libtestu01-0-dev && \
    gcc testu01.c -ltestu01 -o t01 && \
    (./t01 random.bin | tee output.txt) && \
    (sed -n '/Summary/,/End/p' output.txt | tee output_postprocessed.txt)"
```

### ENT

I run ENT (https://www.fourmilab.ch/random/) test to check that bits are unbiased and have full entropy. Here is its output:

```
Entropy = 7.999996 bits per byte.

Optimum compression would reduce the size
of this 50000000 byte file by 0 percent.

Chi square distribution for 50000000 samples is 267.74, and randomly
would exceed this value 27.95 percent of the times.

Arithmetic mean value of data bytes is 127.4910 (127.5 = random).
Monte Carlo value for Pi is 3.142299486 (error 0.02 percent).
Serial correlation coefficient is 0.000072 (totally uncorrelated = 0.0)
```

##### Run in container

```
podman run -it --rm -v ./:/trng -w /trng --security-opt label=disable ubuntu:22.04 /bin/bash -c "\
    apt-get update && apt-get install -y ent && \
    ent random.bin"
```