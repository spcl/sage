# compile and link dynamically using:

g++ -c main.cpp;
nvcc -c test.cu;
g++ main.o test.o -o test -L/lib/x86_64-linux-gnu -lcudart -lcudadevrt

# in case libcudart.so are located in a different location, one can use
ldconfig -p | grep cudart
# or
find / -name "libcudart.so" 2> /dev/null

# More information (e.g., for static linking) can be found here:
https://stackoverflow.com/questions/4786764/linking-in-library-that-contains-reference-to-cuda-kernel