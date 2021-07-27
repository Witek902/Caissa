#pragma once

#include <inttypes.h>

#ifndef CONFIGURATION_FINAL
    #define ASSERT(x) if (!(x)) __debugbreak();
#else
    #define ASSERT(x)
#endif

#define INLINE __forceinline
#define INLINE_LAMBDA [[msvc::forceinline]]
#define NO_INLINE __declspec(noinline)

template<typename T>
INLINE constexpr bool IsPowerOfTwo(const T n)
{
    return (n & (n - 1)) == 0;
}

using ScoreType = int32_t;

static constexpr int32_t CheckmateValue     = 100000;
static constexpr int32_t TablebaseWinValue  = 90000;
static constexpr int32_t InfValue           = 10000000;
static constexpr int32_t InvalidValue       = 9999999;

static constexpr int32_t MaxSearchDepth = 256;
