/*
 * SOURCE: https://github.com/mochimodev/cuda-hashing-algos
 * Authors: Matt Zweil & The Mochimo Core Contributor Team
 * 
 * sha256.cuh CUDA Implementation of SHA256 Hashing    
 *
 * Date: 12 June 2019
 * Revision: 1
 * 
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

void cuda_sha256_hash_batch(unsigned char* in, unsigned int inlen, unsigned char* out, unsigned int n_batch);
__device__ void sha256_hash(unsigned char* in, unsigned int inlen, unsigned char* out);