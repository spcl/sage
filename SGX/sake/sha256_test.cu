/*
nvcc -o test sha256_test.cu sha256.cu -arch=sm_61 -lcuda

sm_50 for GTX750
*/ 


#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>

#include "sha256.cuh"

typedef struct TestVector_t { 
    std::string data;
    std::string hash;
} TestVector_t;

typedef std::vector<TestVector_t> TestData_t;

char * hash_to_string(unsigned char * buff) {
	char * string = (char *)malloc(70);
	int k, i;
	for (i = 0, k = 0; i < 32; i++, k+= 2)
	{
		sprintf(string + k, "%.2x", buff[i]);
		//printf("%02x", buff[i]);
	}
	string[64] = 0;
	return string;
}

void sha256_test() {
    // Source of test vectors: https://www.di-mgt.com.au/sha_testvectors.html
    // https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/SHA256.pdf

    const TestVector_t v1 = {
        "", // empty string
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    };

    const TestVector_t v2 = {
        "abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    };

    const TestVector_t v3 = {
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    };

    const TestVector_t v4 = {
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
    };

    TestData_t test_vectors = { v1, v2, v3, v4 };

    for (int i = 0; i < test_vectors.size(); i++) {
        printf("tv %d\t", i);

        unsigned char out[32] = { 0 };
        unsigned char* data = reinterpret_cast<unsigned char*>(const_cast<char*>(test_vectors[i].data.c_str()));
        cuda_sha256_hash_batch(data, (unsigned int)test_vectors[i].data.size(), &out[0], 1);

        if (hash_to_string(out) == test_vectors[i].hash) {
            printf("Hash matches!\n");
        }
        else {
            std::cout << hash_to_string(out) << std::endl;
        }
        
    }

}

int main() {
    sha256_test();

    return 0;
}