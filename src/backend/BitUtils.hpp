#pragma once

#include "Common.hpp"

#if defined(PLATFORM_WINDOWS)
    #include <intrin.h>
#endif

#ifdef USE_SSE
    #include <immintrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)

    INLINE uint32_t PopCount(uint8_t x)
    {
#ifdef USE_POPCNT
        return __popcnt16(x);
#else // !USE_POPCNT
        const uint8_t m1 = 0x55;
        const uint8_t m2 = 0x33;
        const uint8_t m4 = 0x0f;
        x = (x & m1) + ((x >> 1) & m1);
        x = (x & m2) + ((x >> 2) & m2);
        x = (x & m4) + ((x >> 4) & m4);
        return x;
#endif // USE_POPCNT
    }

    INLINE uint32_t PopCount(uint16_t x)
    {
#ifdef USE_POPCNT
        return __popcnt16(x);
#else // !USE_POPCNT
        const uint16_t m1 = 0x5555;
        const uint16_t m2 = 0x3333;
        const uint16_t m4 = 0x0f0f;
        const uint16_t m8 = 0x00ff;
        x = (x & m1) + ((x >> 1) & m1);
        x = (x & m2) + ((x >> 2) & m2);
        x = (x & m4) + ((x >> 4) & m4);
        x = (x & m8) + ((x >> 8) & m8);
        return x;
#endif // USE_POPCNT
    }

    INLINE uint32_t PopCount(uint32_t x)
    {
#ifdef USE_POPCNT
        return __popcnt(x);
#else // !USE_POPCNT
        x -= ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        return (((x + (x >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif // USE_POPCNT
    }

    INLINE uint32_t PopCount(uint64_t x)
    {
#if defined(USE_POPCNT) && defined(_WIN64)
        return (uint32_t)__popcnt64(x);
#else // !USE_POPCNT
        // https://en.wikipedia.org/wiki/Hamming_weight
        const uint64_t m1 = 0x5555555555555555;
        const uint64_t m2 = 0x3333333333333333;
        const uint64_t m4 = 0x0f0f0f0f0f0f0f0f;
        const uint64_t h01 = 0x0101010101010101;
        x -= (x >> 1) & m1;
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        return static_cast<uint32_t>((x * h01) >> 56);
#endif // USE_POPCNT
    }

    INLINE uint32_t FirstBitSet(uint16_t x)
    {
        ASSERT(x != 0);
#ifdef USE_POPCNT
        unsigned long index;
        _BitScanForward(&index, x);
        return (uint32_t)index;
#else // !USE_POPCNT
        uint16_t v = x;   // 32-bit word input to count zero bits on right
        uint32_t c = 16;  // c will be the number of zero bits on the right
        v &= -static_cast<int16_t>(v);
        if (v) c--;
        if (v & static_cast<uint16_t>(0x00FF)) c -= 8;
        if (v & static_cast<uint16_t>(0x0F0F)) c -= 4;
        if (v & static_cast<uint16_t>(0x3333)) c -= 2;
        if (v & static_cast<uint16_t>(0x5555)) c -= 1;
        return c;
#endif // USE_POPCNT
    }

    INLINE uint32_t FirstBitSet(uint32_t x)
    {
        ASSERT(x != 0);
#ifdef USE_POPCNT
        unsigned long index;
        _BitScanForward(&index, x);
        return (uint32_t)index;
#else // !USE_POPCNT
        uint32_t v = x;   // 32-bit word input to count zero bits on right
        uint32_t c = 32;  // c will be the number of zero bits on the right
        v &= -static_cast<int32_t>(v);
        if (v) c--;
        if (v & 0x0000FFFF) c -= 16;
        if (v & 0x00FF00FF) c -= 8;
        if (v & 0x0F0F0F0F) c -= 4;
        if (v & 0x33333333) c -= 2;
        if (v & 0x55555555) c -= 1;
        return c;
#endif // USE_POPCNT
    }

    INLINE uint32_t FirstBitSet(uint64_t x)
    {
        ASSERT(x != 0);
#if defined(USE_POPCNT) && defined(_WIN64)
        unsigned long index;
        _BitScanForward64(&index, x);
        return (uint32_t)index;
#else // !USE_POPCNT
        uint64_t v = x;   // 32-bit word input to count zero bits on right
        uint32_t c = 64;  // c will be the number of zero bits on the right
        v &= -static_cast<int64_t>(v);
        if (v) c--;
        if (v & 0x00000000FFFFFFFFul) c -= 32;
        if (v & 0x0000FFFF0000FFFFul) c -= 16;
        if (v & 0x00FF00FF00FF00FFul) c -= 8;
        if (v & 0x0F0F0F0F0F0F0F0Ful) c -= 4;
        if (v & 0x3333333333333333ul) c -= 2;
        if (v & 0x5555555555555555ul) c -= 1;
        return c;
#endif // USE_POPCNT
    }

    INLINE uint32_t LastBitSet(uint32_t x)
    {
        ASSERT(x != 0);
#ifdef USE_POPCNT
        unsigned long index;
        _BitScanReverse(&index, x);
        return (uint32_t)index;
#else // !USE_POPCNT
        uint32_t c = 0;
        if (x <= 0x0000FFFF) c += 16, x <<= 16;
        if (x <= 0x00FFFFFF) c += 8, x <<= 8;
        if (x <= 0x0FFFFFFF) c += 4, x <<= 4;
        if (x <= 0x3FFFFFFF) c += 2, x <<= 2;
        if (x <= 0x7FFFFFFF) c++;
        return 31u - c;
#endif // USE_POPCNT
    }

    INLINE uint32_t LastBitSet(uint64_t x)
    {
        ASSERT(x != 0);
#if defined(USE_POPCNT) && defined(_WIN64)
        unsigned long index;
        _BitScanReverse64(&index, x);
        return (uint32_t)index;
#else // !USE_POPCNT
        // algorithm by Kim Walisch and Mark Dickinson 
        const uint8_t index64[64] =
        {
             0, 47,  1, 56, 48, 27,  2, 60,
            57, 49, 41, 37, 28, 16,  3, 61,
            54, 58, 35, 52, 50, 42, 21, 44,
            38, 32, 29, 23, 17, 11,  4, 62,
            46, 55, 26, 59, 40, 36, 15, 53,
            34, 51, 20, 43, 31, 22, 10, 45,
            25, 39, 14, 33, 19, 30,  9, 24,
            13, 18,  8, 12,  7,  6,  5, 63,
        };
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x |= x >> 32;
        return index64[(x * 0x03f79d71b4cb0a89ull) >> 58];
#endif // USE_POPCNT
    }

#else

    INLINE uint32_t PopCount(uint64_t x)
    {
        return (uint32_t)__builtin_popcountll(x);
    }

    INLINE uint32_t FirstBitSet(uint16_t x)
    {
        ASSERT(x != 0);
        return (uint32_t)__builtin_ctz(x);
    }

    INLINE uint32_t FirstBitSet(uint32_t x)
    {
        ASSERT(x != 0);
        return (uint32_t)__builtin_ctz(x);
    }

    INLINE uint32_t FirstBitSet(uint64_t x)
    {
        ASSERT(x != 0);
        return (uint32_t)__builtin_ctzll(x);
    }

    INLINE uint32_t LastBitSet(uint64_t x)
    {
        ASSERT(x != 0);
        return 63u ^ (uint32_t)__builtin_clzll(x);
    }

#endif

inline uint64_t ParallelBitsDeposit(uint64_t src, uint64_t mask)
{
#if defined(USE_BMI2) && (defined(_WIN64) || defined(__x86_64__))
    return _pdep_u64(src, mask);
#else
    uint64_t result = 0;
    for (uint64_t m = 1; mask; m += m)
    {
        if (src & m)
        {
            result |= mask & -static_cast<int64_t>(mask);
        }
        mask &= mask - 1;
    }
    return result;
#endif
}

inline uint32_t ParallelBitsDeposit(uint32_t src, uint32_t mask)
{
#ifdef USE_BMI2
    return _pdep_u32(src, mask);
#else
    uint32_t result = 0;
    for (uint32_t m = 1; mask; m += m)
    {
        if (src & m)
        {
            result |= mask & -static_cast<int32_t>(mask);
        }
        mask &= mask - 1;
    }
    return result;
#endif
}

inline uint64_t ParallelBitsExtract(uint64_t src, uint64_t mask)
{
#if defined(USE_BMI2) && (defined(_WIN64) || defined(__x86_64__))
    return _pext_u64(src, mask);
#else
    uint64_t result = 0;
    for (uint64_t bb = 1; mask; bb += bb)
    {
        if (src & mask & -static_cast<int64_t>(mask))
        {
            result |= bb;
        }
        mask &= mask - 1;
    }
    return result;
#endif
}

inline uint64_t ParallelBitsExtract(uint32_t src, uint32_t mask)
{
#ifdef USE_BMI2
    return _pext_u32(src, mask);
#else
    uint32_t result = 0;
    for (uint32_t bb = 1; mask; bb += bb)
    {
        if (src & mask & -static_cast<int32_t>(mask))
        {
            result |= bb;
        }
        mask &= mask - 1;
    }
    return result;
#endif
}

inline uint64_t SwapBytes(uint64_t x)
{
#if defined(PLATFORM_WINDOWS)
    return _byteswap_uint64(x);
#else
    return __builtin_bswap64(x);
#endif
}

inline uint8_t ReverseBits(uint8_t x)
{
    constexpr uint8_t lookup[16] = { 0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf };
    return (lookup[x & 0xf] << 4) | lookup[x >> 4];
}
