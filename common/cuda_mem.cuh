#pragma once

#if defined(__CUDA_ARCH__)
    #define __hd__ __host__ __device__
#else
    #define __hd__
#endif

inline __hd__ void cudaGlobalFence() {
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

    __hd__ bool tryLock() volatile {
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

    __hd__ void lock() volatile {
        while (!tryLock()) {}
    }

    __hd__ void unlock() volatile {
        cudaGlobalFence();
        locked = 0;
    }
private:
    unsigned locked;
};

struct Message {
    __hd__ Message()
        : id(0), ptr(nullptr), size(0), threadRank(-1) {}

    __hd__ Message(int id, char* ptr, size_t size, int threadRank)
        : id(id), ptr(ptr), size(size), threadRank(threadRank) {}

    __hd__ volatile Message& operator=(const Message& rhs) volatile {
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
