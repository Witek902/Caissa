#include "Common.hpp"

#include "Memory.hpp"
#include "PositionHash.hpp"
#include "Endgame.hpp"
#include "Evaluate.hpp"
#include "SearchUtils.hpp"

#if defined(PLATFORM_WINDOWS)
	#define WIN32_LEAN_AND_MEAN
	#define NOMINMAX
	#include <Windows.h>
#endif

void InitEngine()
{
    EnableLargePagesSupport();
    InitBitboards();
    InitZobristHash();
    InitEndgame();
    InitEvaluation();
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
