#include "Common.hpp"
#include "Memory.hpp"
#include "Numa.hpp"
#include "PositionHash.hpp"
#include "Endgame.hpp"
#include "SearchUtils.hpp"

#include <iostream>

#ifdef USE_SSE
    #include <immintrin.h>
#endif

#ifndef CONFIGURATION_FINAL
NO_INLINE void AssertionFailed(const char* expression, const char* file, int line)
{
    std::cout << "Assertion failed: " << expression << " (" << file << ":" << line << ")" << std::endl;
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
