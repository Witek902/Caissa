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

using ScoreType = int16_t;

static constexpr ScoreType InfValue           = 32767;
static constexpr ScoreType InvalidValue       = INT16_MAX;
static constexpr ScoreType CheckmateValue     = 32000;
static constexpr ScoreType TablebaseWinValue  = 31000;

static constexpr ScoreType MaxSearchDepth = 256;
