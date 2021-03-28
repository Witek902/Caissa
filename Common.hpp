#pragma once

#ifndef CONFIGURATION_FINAL
    #define ASSERT(x) if (!(x)) __debugbreak();
#else
    #define ASSERT(x)
#endif

#define INLINE __forceinline
#define NO_INLINE __declspec(noinline)