#include "Common.hpp"
#include "Memory.hpp"
#include "Numa.hpp"
#include "PositionHash.hpp"
#include "Endgame.hpp"
#include "SearchUtils.hpp"

#include <stdio.h>

#ifdef USE_SSE
    #include <immintrin.h>
#endif

#if defined(PLATFORM_WINDOWS) && !defined(CONFIGURATION_FINAL)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif // NOMINMAX
    #include <Windows.h>
#endif

#ifndef CONFIGURATION_FINAL
NO_INLINE bool AssertionFailed(const char* expression, const char* file, int line)
{
    fprintf(stderr, "Assertion failed: %s (%s:%d)\n", expression, file, line);
#if defined(PLATFORM_WINDOWS)
    return IsDebuggerPresent() != FALSE;
#else
    return true;
#endif // PLATFORM_WINDOWS
}
#endif // CONFIGURATION_FINAL

void InitEngine()
{
    // force rounding denormals to zero
#ifdef USE_SSE
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif // USE_SSE

    numa::Init();
    EnableLargePagesSupport();
    Square::Init();
    InitBitboards();
    InitZobristHash();
    InitEndgame();
    SearchUtils::Init();
}
