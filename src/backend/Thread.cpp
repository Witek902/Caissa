#include "Thread.hpp"

#if defined(PLATFORM_WINDOWS)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif // NOMINMAX
    #include <Windows.h>
    #include <process.h>
#else
    #include <time.h>
    #include <unistd.h>
#endif

namespace {

struct ThreadStart
{
    Thread::Function function;
    void* userData;
};

#if defined(PLATFORM_WINDOWS)

static_assert(sizeof(SRWLOCK) == sizeof(void*), "SRWLOCK must be one pointer (SRWLOCK_INIT is all zeros)");
static_assert(sizeof(CONDITION_VARIABLE) == sizeof(void*), "CONDITION_VARIABLE must be one pointer");

SRWLOCK* AsSrwLock(void*& storage) { return reinterpret_cast<SRWLOCK*>(&storage); }
CONDITION_VARIABLE* AsConditionVariable(void*& storage) { return reinterpret_cast<CONDITION_VARIABLE*>(&storage); }

unsigned __stdcall ThreadMain(void* parameter)
{
    const ThreadStart start = *static_cast<ThreadStart*>(parameter);
    delete static_cast<ThreadStart*>(parameter);
    start.function(start.userData);
    return 0;
}

#else

void* ThreadMain(void* parameter)
{
    const ThreadStart start = *static_cast<ThreadStart*>(parameter);
    delete static_cast<ThreadStart*>(parameter);
    start.function(start.userData);
    return nullptr;
}

#endif // PLATFORM_WINDOWS

} // namespace

Thread::~Thread()
{
    ASSERT(!mRunning);
}

Thread::Thread(Thread&& other)
    : mHandle(other.mHandle)
    , mRunning(other.mRunning)
{
    other.mRunning = false;
}

Thread& Thread::operator = (Thread&& other)
{
    ASSERT(!mRunning);
    if (this != &other)
    {
        mHandle = other.mHandle;
        mRunning = other.mRunning;
        other.mRunning = false;
    }
    return *this;
}

bool Thread::Start(Function function, void* userData)
{
    ASSERT(!mRunning);

    ThreadStart* start = new ThreadStart{ function, userData };
#if defined(PLATFORM_WINDOWS)
    mHandle = reinterpret_cast<void*>(_beginthreadex(nullptr, 0, &ThreadMain, start, 0, nullptr));
    mRunning = mHandle != nullptr;
#else
    mRunning = pthread_create(&mHandle, nullptr, &ThreadMain, start) == 0;
#endif
    if (!mRunning)
    {
        delete start;
    }
    return mRunning;
}

void Thread::Join()
{
    if (!mRunning)
    {
        return;
    }

#if defined(PLATFORM_WINDOWS)
    WaitForSingleObject(mHandle, INFINITE);
    CloseHandle(mHandle);
    mHandle = nullptr;
#else
    pthread_join(mHandle, nullptr);
#endif
    mRunning = false;
}

uint32_t Thread::GetNumLogicalProcessors()
{
#if defined(PLATFORM_WINDOWS)
    SYSTEM_INFO info;
    GetNativeSystemInfo(&info);
    return info.dwNumberOfProcessors;
#else
    const long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? static_cast<uint32_t>(count) : 0u;
#endif
}

#if defined(PLATFORM_WINDOWS)

Mutex::Mutex() { InitializeSRWLock(AsSrwLock(mLock)); }
Mutex::~Mutex() = default;
void Mutex::Lock() { AcquireSRWLockExclusive(AsSrwLock(mLock)); }
void Mutex::Unlock() { ReleaseSRWLockExclusive(AsSrwLock(mLock)); }

ConditionVariable::ConditionVariable() { InitializeConditionVariable(AsConditionVariable(mVariable)); }
ConditionVariable::~ConditionVariable() = default;

void ConditionVariable::Wait(Mutex& mutex)
{
    SleepConditionVariableSRW(AsConditionVariable(mVariable), AsSrwLock(mutex.mLock), INFINITE, 0);
}

bool ConditionVariable::WaitFor(Mutex& mutex, uint32_t timeoutMs)
{
    return SleepConditionVariableSRW(AsConditionVariable(mVariable), AsSrwLock(mutex.mLock), timeoutMs, 0) != FALSE;
}

void ConditionVariable::NotifyOne() { WakeConditionVariable(AsConditionVariable(mVariable)); }
void ConditionVariable::NotifyAll() { WakeAllConditionVariable(AsConditionVariable(mVariable)); }

#else

Mutex::Mutex() { pthread_mutex_init(&mLock, nullptr); }
Mutex::~Mutex() { pthread_mutex_destroy(&mLock); }
void Mutex::Lock() { pthread_mutex_lock(&mLock); }
void Mutex::Unlock() { pthread_mutex_unlock(&mLock); }

ConditionVariable::ConditionVariable()
{
#if defined(__APPLE__)
    pthread_cond_init(&mVariable, nullptr);
#else
    // timed waits use the monotonic clock, so changing the system time does not affect them
    pthread_condattr_t attr;
    pthread_condattr_init(&attr);
    pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    pthread_cond_init(&mVariable, &attr);
    pthread_condattr_destroy(&attr);
#endif
}

ConditionVariable::~ConditionVariable() { pthread_cond_destroy(&mVariable); }

void ConditionVariable::Wait(Mutex& mutex)
{
    pthread_cond_wait(&mVariable, &mutex.mLock);
}

bool ConditionVariable::WaitFor(Mutex& mutex, uint32_t timeoutMs)
{
#if defined(__APPLE__)
    timespec timeout;
    timeout.tv_sec = timeoutMs / 1000u;
    timeout.tv_nsec = static_cast<long>(timeoutMs % 1000u) * 1000000L;
    return pthread_cond_timedwait_relative_np(&mVariable, &mutex.mLock, &timeout) == 0;
#else
    timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += timeoutMs / 1000u;
    deadline.tv_nsec += static_cast<long>(timeoutMs % 1000u) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L)
    {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }
    return pthread_cond_timedwait(&mVariable, &mutex.mLock, &deadline) == 0;
#endif
}

void ConditionVariable::NotifyOne() { pthread_cond_signal(&mVariable); }
void ConditionVariable::NotifyAll() { pthread_cond_broadcast(&mVariable); }

#endif // PLATFORM_WINDOWS
