CAP = 75
GENARGS = --with_adversarial_nop

FUNCNAME = .text._Z12checksum_ptx5State

.EXTRA_PREREQS = Makefile

all: cuda_src_ptx.sass_ctrl checksum_launcher

cuda_src_cpp.cpp cuda_src_ptx.cu cuda_src.h: instr_gen_addshift.py 
	python instr_gen_addshift.py $(GENARGS)

cuda_src_ptx.cubin: cuda_src_ptx.cu cuda_src.h
	nvcc cuda_src_ptx.cu -cubin -dlink -arch=sm_$(CAP) -maxrregcount 32 -o cuda_src_ptx.cubin

cuda_src_ptx_patched.cubin: cuda_src_ptx.cubin checksum_function_generated.bin
	bash patch_cubin.sh cuda_src_ptx.cubin checksum_function_generated.bin $(FUNCNAME) cuda_src_ptx_patched.cubin

checksum_function_extracted.bin: cuda_src_ptx.cubin
	../extract_section.sh cuda_src_ptx.cubin checksum_function_extracted.bin $(FUNCNAME)

checksum_launcher: checksum_launcher.cpp cuda_src_cpp.cpp cuda_src.h checksum_function_extracted.bin checksum_function_generated.bin cuda_src_ptx.cubin cuda_src_ptx_patched.cubin
	nvcc checksum_launcher.cpp cuda_src_cpp.cpp -Xcompiler -fopenmp -lcuda -O2 -o checksum_launcher

cuda_src_ptx.sass: cuda_src_ptx.cubin
	nvdisasm -hex -g cuda_src_ptx.cubin > cuda_src_ptx.sass

cuda_src_ptx.sass_ctrl: cuda_src_ptx.sass
	cat cuda_src_ptx.sass | python ../sass_ctrl.py > cuda_src_ptx.sass_ctrl

checksum_function_generated.bin: cuda_src_sass.sass
	python ../sass_compiler.py cuda_src_sass.sass checksum_function_generated.bin

.PHONY: run_reference run_extracted run_patched run_generated test

run_reference: checksum_launcher
	./checksum_launcher -c cuda_src_ptx.cubin -k checksum_kernel -b checksum_function_extracted.bin -m

run_extracted: checksum_launcher
	./checksum_launcher -c cuda_src_ptx.cubin -k checksum_kernel_from_data -b checksum_function_extracted.bin -m

run_patched: checksum_launcher
	./checksum_launcher -c cuda_src_ptx_patched.cubin -k checksum_kernel -b checksum_function_generated.bin

run_generated: checksum_launcher
	./checksum_launcher -c cuda_src_ptx.cubin -k checksum_kernel_from_data -b checksum_function_generated.bin

test_reference: checksum_launcher
	cuda-memcheck ./checksum_launcher -c cuda_src_ptx.cubin -k checksum_kernel -b checksum_function_extracted.bin -g 3 -t 4 -v -m

test_extracted: checksum_launcher
	cuda-memcheck ./checksum_launcher -c cuda_src_ptx.cubin -k checksum_kernel_from_data -b checksum_function_extracted.bin -g 3 -t 4 -v -a -m

test_patched: checksum_launcher
	cuda-memcheck ./checksum_launcher -c cuda_src_ptx_patched.cubin -k checksum_kernel -b checksum_function_generated.bin -g 3 -t 4 -v -m

test_generated: checksum_launcher
	cuda-memcheck ./checksum_launcher -c cuda_src_ptx.cubin -k checksum_kernel_from_data -b checksum_function_generated.bin -g 3 -t 4 -v

test: test_reference test_extracted test_patched test_generated

.PHONY: clean
clean:
	rm -f cuda_src_cpp.cpp cuda_src_ptx.cu cuda_src.h cuda_src_ptx.cubin cuda_src_ptx_patched.cubin checksum_function_extracted.bin checksum_launcher cuda_src_ptx.sass cuda_src_ptx.sass_ctrl checksum_function_generated.bin
