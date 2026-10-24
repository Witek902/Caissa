#pragma once

// Threads and synchronization over Win32 (SRWLOCK, CONDITION_VARIABLE) and pthreads.
// The Win32 objects are stored as pointer-sized fields, so this header does not include <Windows.h>.

#include "Common.hpp"

#if !defined(PLATFORM_WINDOWS)
    #include <pthread.h>
#endif

// OS thread running a function with a user-data pointer. Must be joined before destruction.
class Thread
{
public:
    using Function = void (*)(void* userData);

    Thread() = default;
    ~Thread();

    Thread(Thread&& other);
    Thread& operator = (Thread&& other);

    // returns false if the thread cannot be created
    bool Start(Function function, void* userData);

    // runs a copy of the callable (e.g. a lambda) on a new thread
    template<typename Func>
    bool Start(Func&& func)
    {
        using FuncType = typename RemoveReference<Func>::Type;
        FuncType* copy = new FuncType(Forward<Func>(func));
        const Function trampoline = [](void* userData)
        {
            FuncType* f = static_cast<FuncType*>(userData);
            (*f)();
            delete f;
        };
        if (Start(trampoline, copy)) return true;
        delete copy;
        return false;
    }

    // waits for the thread to finish, no-op if it is not running
    void Join();

    bool IsRunning() const { return mRunning; }

    static uint32_t GetNumLogicalProcessors();

private:
    Thread(const Thread&) = delete;
    Thread& operator = (const Thread&) = delete;

#if defined(PLATFORM_WINDOWS)
    void* mHandle = nullptr;    // HANDLE
#else
    pthread_t mHandle;
#endif
    bool mRunning = false;
};

// exclusive, non-recursive lock
class Mutex
{
public:
    Mutex();
    ~Mutex();

    void Lock();
    void Unlock();

private:
    Mutex(const Mutex&) = delete;
    Mutex& operator = (const Mutex&) = delete;

    friend class ConditionVariable;

#if defined(PLATFORM_WINDOWS)
    void* mLock = nullptr;      // SRWLOCK
#else
    pthread_mutex_t mLock;
#endif
};

class ConditionVariable
{
public:
    ConditionVariable();
    ~ConditionVariable();

    // the mutex must be locked by the caller, it is released while waiting and locked again on return
    // wakeups may be spurious: wait in a loop on the condition
    void Wait(Mutex& mutex);

    // returns false on timeout
    bool WaitFor(Mutex& mutex, uint32_t timeoutMs);

    void NotifyOne();
    void NotifyAll();

private:
    ConditionVariable(const ConditionVariable&) = delete;
    ConditionVariable& operator = (const ConditionVariable&) = delete;

#if defined(PLATFORM_WINDOWS)
    void* mVariable = nullptr;  // CONDITION_VARIABLE
#else
    pthread_cond_t mVariable;
#endif
};

// locks a mutex for the lifetime of the scope
class ScopedLock
{
public:
    explicit ScopedLock(Mutex& mutex) : mMutex(mutex) { mMutex.Lock(); }
    ~ScopedLock() { mMutex.Unlock(); }

private:
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator = (const ScopedLock&) = delete;

    Mutex& mMutex;
};
