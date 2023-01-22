#pragma once

#include <time.h>

#define rdtsc(low,high) \
     __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high))

inline void getcycles (long long int * cycles)
{
  unsigned long low;
  long high;
  rdtsc(low,high);
  *cycles = high; 
  *cycles <<= 32; 
  *cycles |= low; 
}

int64_t difftimespec_ns(const struct timespec curr, const struct timespec prev)
{
    return ((int64_t)curr.tv_sec - (int64_t)prev.tv_sec) * (int64_t)1000000000
         + ((int64_t)curr.tv_nsec - (int64_t)prev.tv_nsec);
}

int64_t difftimespec_us(const struct timespec curr, const struct timespec prev)
{
    return ((int64_t)curr.tv_sec - (int64_t)prev.tv_sec) * (int64_t)1000000
         + ((int64_t)curr.tv_nsec - (int64_t)prev.tv_nsec) / 1000;
}