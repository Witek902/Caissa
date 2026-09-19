#include "Memory.hpp"


#if defined(PLATFORM_WINDOWS)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#include <Windows.h>
#include <psapi.h>

bool EnableLargePagesSupport()
{
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;

    // open process token
    if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
    {
        return false;
    }

    // get the luid
    if (!::LookupPrivilegeValueW(NULL, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid))
    {
        ::CloseHandle(hToken);
        return false;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    // enable or disable privilege
    BOOL status = ::AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);

    // It is possible for AdjustTokenPrivileges to return TRUE and still not succeed.
    // So always check for the last error value.
    DWORD error = ::GetLastError();
    if (!status || (error != ERROR_SUCCESS))
    {
        ::CloseHandle(hToken);
        return false;
    }

    ::CloseHandle(hToken);

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

int64_t GetLargePageBytes(const void* ptr, size_t size)
{
    PSAPI_WORKING_SET_EX_INFORMATION info = {};
    info.VirtualAddress = const_cast<void*>(ptr);
    if (!::QueryWorkingSetEx(::GetCurrentProcess(), &info, sizeof(info)) || !info.VirtualAttributes.Valid)
    {
        return -1;
    }

    // large-page allocations are all or nothing
    return info.VirtualAttributes.LargePage ? (int64_t)size : 0;
}


#elif defined(__GNUC__) || defined(__clang__)

#if defined(__linux__)
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#endif // defined(__linux__)

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
    if (ret == 0)
    {
        madvise(ptr, size, MADV_HUGEPAGE);
    }
#endif // defined(__linux__)

    return ret != 0 ? nullptr : ptr;
}

void Free(void* ptr)
{
    free(ptr);
}

int64_t GetLargePageBytes(const void* ptr, size_t size)
{
#if defined(__linux__)
    std::ifstream smaps("/proc/self/smaps");
    if (!smaps)
    {
        return -1;
    }

    // Sums AnonHugePages of the mappings overlapping the range. The madvise in Malloc gives the
    // range its own mapping, so neighbouring allocations are not counted.
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = begin + size;
    bool overlaps = false;
    uint64_t largePageBytes = 0;
    std::string line;
    while (std::getline(smaps, line))
    {
        unsigned long long mappingBegin, mappingEnd, kilobytes;
        if (sscanf(line.c_str(), "%llx-%llx", &mappingBegin, &mappingEnd) == 2)
        {
            overlaps = mappingBegin < end && mappingEnd > begin;
        }
        else if (overlaps && sscanf(line.c_str(), "AnonHugePages: %llu kB", &kilobytes) == 1)
        {
            largePageBytes += kilobytes * 1024;
        }
    }
    return (int64_t)std::min<uint64_t>(largePageBytes, size);
#else
    UNUSED(ptr);
    UNUSED(size);
    return -1;
#endif // defined(__linux__)
}


#endif
