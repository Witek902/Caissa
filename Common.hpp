#pragma once

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