#include "Common.hpp"

#include "Memory.hpp"
#include "PositionHash.hpp"
#include "Endgame.hpp"
#include "Evaluate.hpp"
#include "SearchUtils.hpp"

#if defined(PLATFORM_WINDOWS)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif // NOMINMAX
    #include <Windows.h>
#endif

void InitEngine()
{
    // force rounding denormals to zero
#ifdef USE_SSE
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif // USE_SSE

    EnableLargePagesSupport();
    Square::Init();
    InitBitboards();
    InitZobristHash();
    InitEndgame();
    SearchUtils::Init();
}
