
CAP ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')

	
base:
	nvcc mlp.cu -arch=sm_$(CAP) -rdc=true -maxrregcount 32 && ./a.out

prot:
	nvcc mlp.cu -cubin -dlink -arch=sm_$(CAP) -maxrregcount 32 -o mycubin.cubin
	../checksum/extract_section.sh mycubin.cubin linear.bin .text._Z6linearPv
	../checksum/extract_section.sh mycubin.cubin relu.bin .text._Z4reluPv
	nvcc mlp_protected.cu -arch=sm_$(CAP) -rdc=true -maxrregcount 32 && ./a.out