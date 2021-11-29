#pragma once

inline __host__ __device__ void cudaGlobalFence() {
    #if defined(__CUDA_ARCH__)
        __threadfence_system();
    #else
        __sync_synchronize();
    #endif
}

class MemoryLock {
public:
    MemoryLock() : locked(0) {
    }

    __host__ __device__ bool tryLock() volatile {
        // Since CAS returns old value, the operation is successful
        // if an old value (second arg of CAS) equal to the return value
        bool success = false;
        #if defined(__CUDA_ARCH__)
            success = (0 == atomicCAS_system((unsigned*)&locked, 0, 1));
        #else
            success = (0 == __sync_val_compare_and_swap((unsigned*)&locked, 0, 1));
        #endif
        return success;
    }

    __host__ __device__ void lock() volatile {
        while (!tryLock()) {}
    }

    __host__ __device__ void unlock() volatile {
        cudaGlobalFence();
        locked = 0;
    }
private:
    unsigned locked;
};

struct Message {
    __host__ __device__ Message()
        : id(0), ptr(nullptr), size(0), threadRank(-1) {}

    __host__ __device__ Message(int id, char* ptr, size_t size, int threadRank)
        : id(id), ptr(ptr), size(size), threadRank(threadRank) {}

    __host__ __device__ volatile Message& operator=(const Message& rhs) volatile {
        id = rhs.id;
        ptr = rhs.ptr;
        size = rhs.size;
        threadRank = rhs.threadRank;
        return *this;
    }

    int id;
    char* ptr;
    size_t size;
    int threadRank;
    MemoryLock lock;
};
