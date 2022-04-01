#pragma once

#include "Common.hpp"

#include <limits>
#include <new>
#include <iostream>

bool EnableLargePagesSupport();

[[nodiscard]] void* Malloc(size_t size);
void Free(void* ptr);


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
