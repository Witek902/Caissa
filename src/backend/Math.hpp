#pragma once

#include "Common.hpp"

#include <bit>

template<typename T>
INLINE constexpr bool IsPowerOfTwo(const T n)
{
    return (n & (n - 1)) == 0;
}

template<typename T>
INLINE constexpr T Sqr(const T& x)
{
    return x * x;
}

// return high bits of a 64 bit multiplication
INLINE uint64_t MulHi64(uint64_t a, uint64_t b)
{
#if defined(__GNUC__) && defined(ARCHITECTURE_X64)
    return ((unsigned __int128)a * (unsigned __int128)b) >> 64;
#elif defined(_MSC_VER) && defined(ARCHITECTURE_X64)
    return (uint64_t)__umulh(a, b);
#else
    uint64_t aLow = (uint32_t)a, aHi = a >> 32;
    uint64_t bLow = (uint32_t)b, bHi = b >> 32;
    uint64_t c1 = (aLow * bLow) >> 32;
    uint64_t c2 = aHi * bLow + c1;
    uint64_t c3 = aLow * bHi + (uint32_t)c2;
    return aHi * bHi + (c2 >> 32) + (c3 >> 32);
#endif
}

template<typename T, T multiple>
INLINE constexpr const T RoundUp(const T x)
{
    return ((x + (multiple - 1)) / multiple) * multiple;
}

template<typename T>
INLINE void AtomicMax(std::atomic<T>& outMax, T const& value) noexcept
{
    T prev = outMax;
    while (prev < value && !outMax.compare_exchange_weak(prev, value)) { }
}

inline float Log(float x)
{
    // based on:
    // https://stackoverflow.com/questions/39821367/very-fast-logarithm-natural-log-function-in-c

    // range reduction
    const int32_t e = (std::bit_cast<int32_t>(x) - 0x3f2aaaab) & 0xff800000;
    const float m = std::bit_cast<float>(std::bit_cast<int32_t>(x) - e);
    const float i = 1.19209290e-7f * (float)e;

    const float f = m - 1.0f;
    const float s = f * f;

    // Compute log1p(f) for f in [-1/3, 1/3]
    float r = -0.130187988f * f + 0.140889585f;
    float t = -0.121489584f * f + 0.139809534f;
    r = r * s + t;
    r = r * f - 0.166845024f;
    r = r * f + 0.200121149f;
    r = r * f - 0.249996364f;
    r = r * f + 0.333331943f;
    r = r * f - 0.500000000f;
    r = r * s + f;
    r = i * 0.693147182f + r; // log(2)
    return r;
}

inline float FastLog2(float x)
{
    // based on:
    // https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c/9411984#9411984

    union
    {
        int32_t i32;
        float f;
    } u;

    u.f = x;
    float result = (float)(((u.i32 >> 23) & 255) - 128);
    u.i32 &= ~(255 << 23);
    u.i32 += 127 << 23;
    result += ((-0.33333333f) * u.f + 2.0f) * u.f - 0.66666666f;
    return result;
}
