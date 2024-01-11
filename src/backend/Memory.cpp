#include "Memory.hpp"

#include <iostream>

#if defined(PLATFORM_WINDOWS)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#include <Windows.h>


bool EnableLargePagesSupport()
{
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;

    // open process token
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
    {
        return false;
    }

    // get the luid
    if (!LookupPrivilegeValueW(NULL, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid))
    {
        return false;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    // enable or disable privilege
    BOOL status = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);

    // It is possible for AdjustTokenPrivileges to return TRUE and still not succeed.
    // So always check for the last error value.
    DWORD error = GetLastError();
    if (!status || (error != ERROR_SUCCESS))
    {
        return false;
    }

    CloseHandle(hToken);

    std::cout << "info string Large page support enabled. Minimum page size: " << (GetLargePageMinimum() / 1024u) << " KB" << std::endl;

    return true;
}

NO_INLINE void* Malloc(size_t size)
{
    void* ptr = nullptr;

    // try large pages first
    const size_t largePageMinNumpages = 4;
    const size_t minLargePageSize = largePageMinNumpages * ::GetLargePageMinimum();
    if (size >= minLargePageSize)
    {
        const size_t roundedSize = ((size + minLargePageSize - 1) / minLargePageSize) * minLargePageSize;
        ptr = ::VirtualAlloc(NULL, roundedSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    }

    // fallback to regular pages
    if (!ptr)
    {
        ptr = ::VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    }

    return ptr;
}

void Free(void* ptr)
{
    ::VirtualFree(ptr, 0, MEM_RELEASE);
}


#elif defined(__GNUC__) || defined(__clang__)


bool EnableLargePagesSupport()
{
    return false;
}

void* Malloc(size_t size)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr size_t alignment = 2 * 1024 * 1024;
#else
    constexpr size_t alignment = CACHELINE_SIZE;
#endif // defined(__linux__)

    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);

#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(ptr, size, MADV_HUGEPAGE);
#endif // defined(__linux__)

    return ret != 0 ? nullptr : ptr;
}

void Free(void* ptr)
{
    free(ptr);
}


#endif
