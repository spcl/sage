.PHONY: all

cap = 75

all: checksum checksum_function.bin

checksum: checksum.cu
	nvcc -o ./checksum -rdc=true -lineinfo -arch=sm_${cap} -lcuda checksum.cu

checksum.cubin: checksum
	mv $$(cuobjdump ./checksum -xelf all | awk '{ print $$5; }') checksum.cubin

checksum_function.bin: checksum.cubin
	./extract_section.sh checksum.cubin checksum_function.bin .text._Z17checksum_functionPjS_S_

.PHONY: run
run: all
	./checksum

.PHONY: clean
clean:
	rm -f ./checksum checksum_function.bin checksum.cubin

