#include "../backend/Memory.hpp"

#include <stdio.h>
#include <new>

// Global allocation functions that abort when out of memory. The standard ones throw std::bad_alloc,
// which links the C++ exception runtime into static builds.

[[noreturn]] NO_INLINE static void OutOfMemory(size_t size)
{
    fprintf(stderr, "Out of memory (allocating %zu bytes)\n", size);
    abort();
}

static void* Allocate(size_t size)
{
    void* ptr = malloc(size ? size : 1);
    if (!ptr) [[unlikely]]
        OutOfMemory(size);
    return ptr;
}

static void* AllocateAligned(size_t size, std::align_val_t alignment)
{
    void* ptr = AlignedMalloc(size ? size : 1, static_cast<size_t>(alignment));
    if (!ptr) [[unlikely]]
        OutOfMemory(size);
    return ptr;
}

void* operator new(size_t size) { return Allocate(size); }
void* operator new[](size_t size) { return Allocate(size); }
void* operator new(size_t size, std::align_val_t alignment) { return AllocateAligned(size, alignment); }
void* operator new[](size_t size, std::align_val_t alignment) { return AllocateAligned(size, alignment); }

void operator delete(void* ptr) noexcept { free(ptr); }
void operator delete[](void* ptr) noexcept { free(ptr); }
void operator delete(void* ptr, size_t) noexcept { free(ptr); }
void operator delete[](void* ptr, size_t) noexcept { free(ptr); }
void operator delete(void* ptr, std::align_val_t) noexcept { AlignedFree(ptr); }
void operator delete[](void* ptr, std::align_val_t) noexcept { AlignedFree(ptr); }
void operator delete(void* ptr, size_t, std::align_val_t) noexcept { AlignedFree(ptr); }
void operator delete[](void* ptr, size_t, std::align_val_t) noexcept { AlignedFree(ptr); }
