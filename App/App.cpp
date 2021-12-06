#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

#include "timing.h"
// #include "../compile-test/test.hpp"
#include "../common/cuda_mem.cuh"
#include "../checksum/runner.hpp"
#include "../sake/sake.hpp"

#define NONCE_SIZE 16
#define NONCE_INTERVAL 1000000

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }
    return 0;
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate
     * the input string to prevent buffer overflow.
     */
    printf("%s", str);
}

void print_hex(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        if (i > 0)
            printf(":");
        printf("%02X", buf[i]);
    }
    printf("\n");
}

void copy_nonce(Message *msg, const char *buf, size_t buf_size) {
    msg->lock.lock();
    strncpy(msg->ptr, buf, buf_size);
    msg->ptr[buf_size] = '\0';
    msg->size = buf_size;
    msg->id++;
    msg->lock.unlock();
}

void print_msg(Message* msg) {
    msg->lock.lock();
    print_hex((uint8_t*)msg->ptr, msg->size);
    msg->lock.unlock();
}

/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);
    /* Initialize the enclave */
    if(initialize_enclave() < 0){
        printf("[A] Enclave initialisation failed.\n");
        return -1;
    }
    
    sgx_status_t ret_status;
    sgx_status_t sgx_status;

    ret_status = init_encl(global_eid, &sgx_status);
    if (ret_status != SGX_SUCCESS) {
        printf("[A] Enclave initialisation failed.");
        return -1;
    }

    printf("[A] Enclave creation and initialisation successful!\n");

    // nonces
    int num_blocks = 1;
    uint8_t* out_buf = (uint8_t*)calloc(num_blocks, NONCE_SIZE);
    // ret_status = generate_nonce(global_eid, &sgx_status, num_blocks, out_buf, num_blocks*NONCE_SIZE);
    
    // for (int i = 0; i < num_blocks; i++) {
    //     print_hex(&out_buf[i*NONCE_SIZE], NONCE_SIZE);
    // }

    // timing
    clockid_t clk_id = CLOCK_MONOTONIC;
    struct timespec curr, prev;
    clock_gettime(clk_id, &curr);
    prev = curr;

    Message* msgs;
    sake_runner(&msgs);

    while(true) {
        clock_gettime(clk_id, &curr);
        if (difftimespec_us(curr, prev) > NONCE_INTERVAL) {
            ret_status = generate_nonce(global_eid, &sgx_status, num_blocks, out_buf,  num_blocks*NONCE_SIZE);
            if (ret_status != SGX_SUCCESS) {
                printf("[A] Generating nonce failed.");
                return -1;
            }

            // transfer nonce
            copy_nonce(&msgs[0], (const char*)out_buf, NONCE_SIZE);
            print_hex(out_buf, NONCE_SIZE);
            prev = curr;
        }
    }

    printf("[A] Launching checksum execution\n");
    // checksum_runner();

    /* CUDA test */
    /*
    printf("[A] Launching CUDA test\n");
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    addWithCuda(c, a, b, arraySize);
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    */

    /* Utilize edger8r attributes */
    edger8r_array_attributes();
    edger8r_pointer_attributes();
    edger8r_type_attributes();
    edger8r_function_attributes();

    /* Utilize trusted libraries */
    ecall_libc_functions();
    ecall_libcxx_functions();
    ecall_thread_functions();

    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);

    printf("[A] Enclave destroyed\n");
    return 0;
}
