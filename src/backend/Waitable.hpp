#pragma once

#include "Common.hpp"

#include <atomic>

// Helper class allowing for waiting for an event.
class Waitable final
{
public:
    Waitable();
    ~Waitable();

    // Check if the task has been finished.
    bool IsFinished() { return mFinished.load(); }

    // Wait for a task to finish.
    // NOTE This can be called only on the main thread!
    void Wait();

    void Reset();

    void OnFinished();

private:

    Waitable(const Waitable&) = delete;
    Waitable(Waitable&&) = delete;

    // mutex and condition variable, defined in Waitable.cpp
    struct Sync;
    Sync* mSync;
    std::atomic<bool> mFinished;
};
