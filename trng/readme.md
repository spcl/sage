### Usage

Write 10000 bytes of random binary data into file random_output.bin.

```
nvcc race_conditions_gpu_trng.cu -o trng
./trng random_output.bin 10000
```

### Randomness tests

I made the new implementation of TRNG based on race conditions (`race_conditions_gpu_trng.cu`). 
So far it is the best TRNG implementation we have for GPU.
It passes almost all tests that I tried from TestU01 including diehard tests. I attached the output of the tests (`race_conditions_gpu_trng_TestU01_result.txt`).

The only test that fails seems to require a larger chunk of random data than what I experimented with. It took me a couple of hours to get 50 MB of randomness. The throughput is 4 KB/s on V100, so it can generate 256 bits in 8ms.

I run ENT (https://www.fourmilab.ch/random/) test to check that bits are unbiased and have full entropy. I generate each bit independently and don't mix them in any way, so I hope
that test shows the generator entropy correctly. Here is its output:

Entropy = 7.999996 bits per byte.

Optimum compression would reduce the size
of this 50000000 byte file by 0 percent.

Chi square distribution for 50000000 samples is 267.74, and randomly
would exceed this value 27.95 percent of the times.

Arithmetic mean value of data bytes is 127.4910 (127.5 = random).
Monte Carlo value for Pi is 3.142299486 (error 0.02 percent).
Serial correlation coefficient is 0.000072 (totally uncorrelated = 0.0)
