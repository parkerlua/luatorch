#include "memory_pool.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// a cached memory block
typedef struct PoolBlock {
    void*   ptr;
    size_t  size;
    int     in_use;
    struct PoolBlock* next;
} PoolBlock;

// the pool
static PoolBlock* pool_head = NULL;
static pthread_mutex_t pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static size_t total_allocated = 0;
static size_t total_cached = 0;
static int pool_initialized = 0;

extern "C" void pool_init() {
    pthread_mutex_lock(&pool_mutex);
    pool_initialized = 1;
    pthread_mutex_unlock(&pool_mutex);
}

// round up to nearest power of 2 for better reuse, minimum 512 bytes
// perf fix: also ensure 128 byte alignment for optimal coalesced memory access on 4090
// cudaMalloc already returns 256-byte aligned pointers so power-of-2 rounding handles this
static size_t round_up(size_t size) {
    if (size < 512) return 512;
    size_t p = 1;
    while (p < size) p <<= 1;
    return p;
}

extern "C" void* pool_alloc(size_t size) {
    if (size == 0) return NULL;

    size_t rounded = round_up(size);

    pthread_mutex_lock(&pool_mutex);

    // look for a free block of matching size
    PoolBlock* block = pool_head;
    PoolBlock* best = NULL;
    while (block) {
        if (!block->in_use && block->size == rounded) {
            best = block;
            break;
        }
        // also accept slightly larger blocks to reduce fragmentation
        if (!block->in_use && block->size >= rounded && block->size <= rounded * 2) {
            if (!best || block->size < best->size) {
                best = block;
            }
        }
        block = block->next;
    }

    if (best) {
        best->in_use = 1;
        total_cached -= best->size;
        pthread_mutex_unlock(&pool_mutex);
        return best->ptr;
    }

    // no matching block, allocate new
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, rounded);
    if (err != cudaSuccess) {
        // try clearing cache and retrying
        pthread_mutex_unlock(&pool_mutex);
        pool_clear();
        pthread_mutex_lock(&pool_mutex);

        err = cudaMalloc(&ptr, rounded);
        if (err != cudaSuccess) {
            fprintf(stderr, "luatorch error: pool_alloc failed for %zu bytes: %s\n",
                rounded, cudaGetErrorString(err));
            pthread_mutex_unlock(&pool_mutex);
            return NULL;
        }
    }

    // create new block entry
    // fix: added NULL check on malloc, old code crashed if system memory exhausted
    PoolBlock* new_block = (PoolBlock*)malloc(sizeof(PoolBlock));
    if (!new_block) {
        fprintf(stderr, "luatorch error: failed to allocate pool block entry\n");
        cudaFree(ptr);
        pthread_mutex_unlock(&pool_mutex);
        return NULL;
    }
    new_block->ptr    = ptr;
    new_block->size   = rounded;
    new_block->in_use = 1;
    new_block->next   = pool_head;
    pool_head = new_block;

    total_allocated += rounded;
    pthread_mutex_unlock(&pool_mutex);
    return ptr;
}

extern "C" void pool_free(void* ptr) {
    if (!ptr) return;

    pthread_mutex_lock(&pool_mutex);

    PoolBlock* block = pool_head;
    while (block) {
        if (block->ptr == ptr && block->in_use) {
            block->in_use = 0;
            total_cached += block->size;
            pthread_mutex_unlock(&pool_mutex);
            return;
        }
        block = block->next;
    }

    // not in pool, just free directly
    pthread_mutex_unlock(&pool_mutex);
    cudaFree(ptr);
}

extern "C" void pool_clear() {
    pthread_mutex_lock(&pool_mutex);

    PoolBlock* block = pool_head;
    while (block) {
        PoolBlock* next = block->next;
        if (!block->in_use) {
            cudaFree(block->ptr);
            total_allocated -= block->size;
            total_cached -= block->size;
            // remove from list by marking ptr null
            block->ptr = NULL;
        }
        block = next;
    }

    // compact the list, remove null entries
    PoolBlock** prev = &pool_head;
    block = pool_head;
    while (block) {
        if (block->ptr == NULL) {
            *prev = block->next;
            PoolBlock* to_free = block;
            block = block->next;
            free(to_free);
        } else {
            prev = &block->next;
            block = block->next;
        }
    }

    pthread_mutex_unlock(&pool_mutex);
}

// fix: read counters under mutex to avoid race with alloc/free threads
extern "C" size_t pool_allocated_bytes() {
    pthread_mutex_lock(&pool_mutex);
    size_t val = total_allocated;
    pthread_mutex_unlock(&pool_mutex);
    return val;
}

extern "C" size_t pool_cached_bytes() {
    pthread_mutex_lock(&pool_mutex);
    size_t val = total_cached;
    pthread_mutex_unlock(&pool_mutex);
    return val;
}
