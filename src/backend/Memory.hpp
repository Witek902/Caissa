#pragma once

#include "Common.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#ifdef USE_SSE
    #include <immintrin.h>
#endif

inline void* AlignedMalloc(size_t size, size_t alignment)
{
    void* ptr = nullptr;
#if defined(PLATFORM_WINDOWS)
    ptr = _aligned_malloc(size, alignment);
#elif defined(PLATFORM_LINUX)
    alignment = std::max(alignment, sizeof(void*));
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

bool EnableLargePagesSupport();

[[nodiscard]] void* Malloc(size_t size);
void Free(void* ptr);

// https://stackoverflow.com/a/8545389
template <typename T, std::size_t N = 16>
class AlignmentAllocator
{
public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

public:
    AlignmentAllocator() throw () { }

    template <typename T2>
    AlignmentAllocator(const AlignmentAllocator<T2, N>&) throw () { }

    ~AlignmentAllocator() throw () { }

    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) const { return &r; }
    pointer allocate(size_type n) { return (pointer)AlignedMalloc(n * sizeof(value_type), N); }

    void deallocate(pointer p, size_type) { AlignedFree(p); }
    void construct(pointer p, const value_type& wert) { new (p) value_type(wert); }
    void destroy(pointer p) { p->~value_type(); }
    size_type max_size() const throw () { return size_type(-1) / sizeof(value_type); }

    template <typename T2>
    struct rebind { typedef AlignmentAllocator<T2, N> other; };

    bool operator != (const AlignmentAllocator<T, N>& other) const { return !(*this == other); }
    bool operator == (const AlignmentAllocator<T, N>&) const { return true; }
};

template <class T>
struct Allocator
{
    typedef T value_type;

    Allocator() = default;
    template <class U> constexpr Allocator(const Allocator <U>&) noexcept { }

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_array_new_length();
        }

        if (auto p = static_cast<T*>(Malloc(n * sizeof(T))))
        {
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t) noexcept
    {
        Free(p);
    }
};

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
    std::memcpy(dst, src, size);
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
