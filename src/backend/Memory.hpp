#pragma once

#include "Common.hpp"

#include <stdlib.h>
#include <string.h>

#if defined(PLATFORM_WINDOWS)
    #include <malloc.h>
#endif

#ifdef USE_SSE
    #include <immintrin.h>
#endif

inline void* AlignedMalloc(size_t size, size_t alignment)
{
    void* ptr = nullptr;
#if defined(PLATFORM_WINDOWS)
    ptr = _aligned_malloc(size, alignment);
#elif defined(PLATFORM_LINUX)
    alignment = Max(alignment, sizeof(void*));
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) ptr = nullptr;
#endif
    return ptr;
}

inline void AlignedFree(void* ptr)
{
#if defined(PLATFORM_WINDOWS)
    _aligned_free(ptr);
#elif defined(PLATFORM_LINUX)
    free(ptr);
#endif
}

// placement new without the <new> header: PLACEMENT_NEW(ptr) Type(args...)
struct PlacementNewTag { };
INLINE void* operator new(size_t, void* ptr, PlacementNewTag) noexcept { return ptr; }
INLINE void operator delete(void*, void*, PlacementNewTag) noexcept { }
#define PLACEMENT_NEW(ptr) new (ptr, PlacementNewTag{})

bool EnableLargePagesSupport();

[[nodiscard]] void* Malloc(size_t size);
void Free(void* ptr);

template<typename T>
INLINE void AlignedMemcpy64(T* dst, const T* src)
{
    constexpr size_t size = sizeof(T);
    static_assert(size % 64 == 0, "Size must be multiple of 64");

#if defined(USE_AVX512)
    const __m512i* src512 = reinterpret_cast<const __m512i*>(src);
    __m512i* dst512 = reinterpret_cast<__m512i*>(dst);
    for (size_t i = 0; i < size / 64u; i++)
    {
        _mm512_store_si512(dst512 + i, _mm512_load_si512(src512 + i));
    }
#elif defined(USE_AVX2)
    const __m256i* src256 = reinterpret_cast<const __m256i*>(src);
    __m256i* dst256 = reinterpret_cast<__m256i*>(dst);
    for (size_t i = 0; i < size / 32u; i++)
    {
        _mm256_store_si256(dst256 + i, _mm256_load_si256(src256 + i));
    }
#elif defined(USE_SSE2)
    const __m128i* src128 = reinterpret_cast<const __m128i*>(src);
    __m128i* dst128 = reinterpret_cast<__m128i*>(dst);
    for (size_t i = 0; i < size / 16u; i++)
    {
        _mm_store_si128(dst128 + i, _mm_load_si128(src128 + i));
    }
#else
    memcpy(dst, src, size);
#endif
}

INLINE void Prefetch(const void* ptr)
{
    ASSERT(ptr != nullptr);
#ifdef USE_SSE
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(USE_ARM_NEON)
    __builtin_prefetch(reinterpret_cast<const char*>(ptr), 0, 0);
#else
    (void)ptr;
#endif // USE_SSE
}
