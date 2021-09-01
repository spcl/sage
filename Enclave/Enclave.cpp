#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>

#include "sgx_tcrypto.h"
#include "sgx_trts.h"
#include "sgx_utils.h"

#define SGX_AES_KEY_SIZE 16
#define NONCE_SIZE 16

// index of the protocol
int i = 0;

uint8_t prng_iv[SGX_AES_KEY_SIZE];
sgx_aes_ctr_128bit_key_t prng_key;

int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

sgx_status_t init_encl()
{
  printf("[E] Initialising Enclave ...");
    
  sgx_status_t ret_status;
 
  // generate a random IV
  ret_status = sgx_read_rand(prng_iv, SGX_AES_KEY_SIZE);
  if (ret_status != SGX_SUCCESS)
    return ret_status;

  // generate a random AES key
  ret_status = sgx_read_rand((unsigned char*)&prng_key, SGX_AES_KEY_SIZE);
  if (ret_status != SGX_SUCCESS)
    return ret_status;

  printf(" done!\n");

  printf("[E] prng iv: %u\n", prng_iv);
  printf("[E] prng key: %u\n", prng_key);
  return SGX_SUCCESS;
}

sgx_status_t generate_nonce(unsigned int num_blocks, uint8_t* out_buf, size_t len)
{
  sgx_status_t ret_status;

  printf("[E] Generating %u nonces...", num_blocks);
  
  size_t buffer_len = num_blocks*NONCE_SIZE;

  uint8_t* in_buf = (uint8_t*)calloc(num_blocks, NONCE_SIZE);
  //uint8_t* out_buf = (uint8_t*)calloc(num_blocks, NONCE_SIZE);

  // use AES in CTR mode as a PRNG
  ret_status = sgx_aes_ctr_encrypt(&prng_key, in_buf, buffer_len, prng_iv, 8, out_buf);
  if (ret_status != SGX_SUCCESS)
    return ret_status;

  printf(" done!\n");

  free(in_buf);
  //free(out_buf);
  return SGX_SUCCESS;
}
