
CAP ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')


.PHONY: prot
prot: mlp_protected output_ref.bin linear.bin relu.bin
	./mlp_protected 3 10 1 1 1 0

.PHONY: base
base: mlp
	./mlp 3 10 1 1 1 0
	
.PHONY: test
test: mlp mlp_protected output_ref.bin linear.bin relu.bin
	./mlp 0 1 1 1 1 0
	./mlp_protected 0 1 1 1 1 0

# app example overall performance: x axis batch scale, y axis runtime (stacked bars: base vs test)
perf.txt: mlp mlp_protected output_ref.bin linear.bin relu.bin benchmark_perf.sh
	bash benchmark_perf.sh | tee perf.txt

# evaluation of overhead for copy and kernel launch
# x axis input size, y axis runtime

perf_kernel_and_copy.txt: mlp mlp_protected output_ref.bin linear.bin relu.bin perf_kernel_and_copy.sh
	bash perf_kernel_and_copy.sh | tee perf_kernel_and_copy.txt

mlp: mlp.cu mlp.h
	nvcc mlp.cu -arch=sm_$(CAP) -rdc=true -maxrregcount 32 -o mlp

mycubin.cubin: mlp.cu mlp.h
	nvcc mlp.cu -cubin -dlink -arch=sm_$(CAP) -maxrregcount 32 -o mycubin.cubin

output_ref.bin: mlp
	./mlp 0 1 1 1 1 0

linear.bin: mycubin.cubin
	../checksum/extract_section.sh mycubin.cubin linear.bin .text._Z6linearPv

relu.bin: mycubin.cubin
	../checksum/extract_section.sh mycubin.cubin relu.bin .text._Z4reluPv

mlp_protected: mlp_protected.cu mlp.h
	nvcc mlp_protected.cu -arch=sm_$(CAP) -rdc=true -maxrregcount 32 -o mlp_protected

.PHONY: clean
clean:
	rm -f mlp mlp_protected mycubin.cubin linear.bin relu.bin output_ref.bin