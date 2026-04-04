#ifndef LUATORCH_MEMORY_POOL_H
#define LUATORCH_MEMORY_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// initialize the memory pool, call once at startup
void pool_init();

// allocate gpu memory, reuses freed blocks when possible
void* pool_alloc(size_t size);

// return memory to the pool instead of calling cudaFree
void pool_free(void* ptr);

// free all pooled memory, call between training runs
void pool_clear();

// get pool statistics
size_t pool_allocated_bytes();
size_t pool_cached_bytes();

#ifdef __cplusplus
}
#endif

#endif
