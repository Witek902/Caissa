#pragma once

#include <inttypes.h>
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
    #include <csignal>
#endif


#ifndef CONFIGURATION_FINAL
    #if defined(_MSC_VER)
        #define ASSERT(x) if (!(x)) __debugbreak();
    #elif defined(__GNUC__) || defined(__clang__)
        #define ASSERT(x) if (!(x)) std::raise(SIGINT);
    #endif
#else
    #define ASSERT(x)
#endif

#define CACHELINE_SIZE 64

#if defined(_MSC_VER)

    #define USE_TABLE_BASES

    // "C++ nonstandard extension: nameless struct"
    #pragma warning(disable : 4201)

    // "unreferenced local function"
    #pragma warning(disable : 4505)

    // "structure was padded due to alignment specifier"
    #pragma warning(disable : 4324)

    #define INLINE __forceinline
    #define INLINE_LAMBDA [[msvc::forceinline]]
    #define NO_INLINE __declspec(noinline)

    INLINE uint32_t PopCount(const uint64_t x)
    {
        return (uint32_t)__popcnt64(x);
    }

    INLINE uint32_t FirstBitSet(const uint64_t x)
    {
        unsigned long index;
        const auto ret = _BitScanForward64(&index, x);
        return (uint32_t)index;
    }

    INLINE uint32_t LastBitSet(const uint64_t x)
    {
        unsigned long index;
        const auto ret = _BitScanReverse64(&index, x);
        return (uint32_t)index;
    }

#elif defined(__GNUC__) || defined(__clang__)

    #define INLINE __attribute__((always_inline)) inline
    #define INLINE_LAMBDA
    #define NO_INLINE __attribute__((noinline))

    INLINE uint32_t PopCount(const uint64_t x)
    {
        return (uint32_t)__builtin_popcountll(x);
    }

    INLINE uint32_t FirstBitSet(const uint64_t x)
    {
        return (uint32_t)__builtin_ctzll(x);
    }

    INLINE uint32_t LastBitSet(const uint64_t x)
    {
        return 63u ^ (uint32_t)__builtin_clzll(x);
    }

#endif

#if defined(__GNUC__)
    #define UNNAMED_STRUCT __extension__
#else
    #define UNNAMED_STRUCT
#endif

template<typename T>
INLINE constexpr bool IsPowerOfTwo(const T n)
{
    return (n & (n - 1)) == 0;
}

using ScoreType = int16_t;

static constexpr ScoreType InfValue           = 32767;
static constexpr ScoreType InvalidValue       = INT16_MAX;
static constexpr ScoreType CheckmateValue     = 32000;
static constexpr ScoreType TablebaseWinValue  = 31000;

static constexpr ScoreType MaxSearchDepth = 256;
