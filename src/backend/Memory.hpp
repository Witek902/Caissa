#pragma once

#include "Common.hpp"

#include <limits>
#include <new>
#include <iostream>


inline [[nodiscard]] void* AlignedMalloc(size_t size, size_t alignment)
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

    pointer adress(reference r) { return &r; }
    const_pointer adress(const_reference r) const { return &r; }
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
