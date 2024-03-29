#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>

#include "sgx_tcrypto.h"
#include "sgx_trts.h"
#include "sgx_utils.h"

#define NONCE_SIZE 16

// index of the protocol
int i = 0;

uint8_t prng_iv[SGX_AESCTR_KEY_SIZE];
sgx_aes_ctr_128bit_key_t prng_key;
sgx_cmac_128bit_key_t mac_key;
sgx_ecc_state_handle_t ecc_handle;
sgx_ec256_private_t ecc_private;
sgx_ec256_public_t ecc_public;

int printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

sgx_status_t print_hex(uint8_t *buf, size_t len)
{
    for (size_t it = 0; it < len; it++)
    {
        if (it > 0)
            printf(":");
        printf("%02X", buf[it]);
    }
    printf("\n");
    return SGX_SUCCESS;
}

// sgx_status_t hash(char *buf, size_t size) {
//     sgx_status_t sgx_sha256_msg(
//         const uint8_t *p_src,
//         uint32_t src_len,
//         sgx_sha256_hash_t *p_hash);
// }

sgx_status_t init_encl()
{
    printf("[E] Initialising Enclave ...");

    sgx_status_t ret_status;

    // generate a random IV for PRNG
    ret_status = sgx_read_rand(prng_iv, SGX_AESCTR_KEY_SIZE);
    if (ret_status != SGX_SUCCESS)
        return ret_status;

    // generate a random AES key for PRNG
    ret_status = sgx_read_rand((unsigned char *)&prng_key, SGX_AESCTR_KEY_SIZE);
    if (ret_status != SGX_SUCCESS)
        return ret_status;

    // generate a random AES key for MACing in SAKE
    ret_status = sgx_read_rand((unsigned char *)&mac_key, SGX_CMAC_KEY_SIZE);
    if (ret_status != SGX_SUCCESS)
        return ret_status;

    // generate ECC key pair for SAKE
    ret_status = sgx_ecc256_open_context(&ecc_handle);
    if (ret_status != SGX_SUCCESS)
        return ret_status;

    ret_status = sgx_ecc256_create_key_pair(&ecc_private, &ecc_public, ecc_handle);
    if (ret_status != SGX_SUCCESS)
        return ret_status;

    printf(" done!\n");

    printf("[E] prng iv:\t");
    print_hex(&prng_iv[0], SGX_AESCTR_KEY_SIZE);
    printf("[E] prng key:\t");
    print_hex(&prng_key[0], SGX_AESCTR_KEY_SIZE);
    printf("[E] mac key:\t");
    print_hex(&mac_key[0], SGX_CMAC_KEY_SIZE);
    printf("[E] ECC pub key:\t");
    print_hex(&ecc_private.r[0], SGX_ECP256_KEY_SIZE);
    printf("[E] ECC priv key:\t");
    print_hex(&ecc_public.gx[0], SGX_ECP256_KEY_SIZE);
    print_hex(&ecc_public.gy[0], SGX_ECP256_KEY_SIZE);

    return SGX_SUCCESS;
}

sgx_status_t generate_nonce(unsigned int num_blocks, uint8_t *out_buf, size_t out_len)
{
    sgx_status_t ret_status;

    uint32_t buffer_len = num_blocks * NONCE_SIZE;
    if (buffer_len > out_len) { // the ciphertext will be larger than the available buffer
        printf("[E] Buffer length mismatch in `generate_nonce!\n`");
        return SGX_ERROR_UNEXPECTED;
    }

    uint8_t *in_buf = (uint8_t *)calloc(num_blocks, NONCE_SIZE);
    // use AES in CTR mode as a PRNG
    ret_status = sgx_aes_ctr_encrypt(&prng_key, in_buf, buffer_len, prng_iv, 8, out_buf);
    if (ret_status != SGX_SUCCESS)
        return ret_status;

    free(in_buf);
    return SGX_SUCCESS;
}

sgx_status_t cleanup_encl() {
    sgx_status_t ret_status;

    printf("[E] Clean up enclave ...");
    ret_status = sgx_ecc256_close_context(ecc_handle);
    if (ret_status != SGX_SUCCESS)
        return ret_status;

    return SGX_SUCCESS;
}