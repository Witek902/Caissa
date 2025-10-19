#include "Common.hpp"

#include "Memory.hpp"
#include "PositionHash.hpp"
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
    SearchUtils::Init();
}

std::string GetExecutablePath()
{
    std::string ret;
#if defined(PLATFORM_WINDOWS)
    char path[MAX_PATH];
    HMODULE hModule = GetModuleHandle(NULL);
    if (hModule != NULL)
    {
        // Use GetModuleFileName() with module handle to get the path
        GetModuleFileNameA(hModule, path, (sizeof(path)));
        ret = std::string(path);
    }
#elif defined(PLATFORM_LINUX)
    if (char* execPath = realpath("/proc/self/exe", nullptr))
    {
        ret = execPath;
        free(execPath);
    }
#endif
    return ret;
}
