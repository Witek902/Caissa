#include "Waitable.hpp"

#include <condition_variable>
#include <mutex>

struct Waitable::Sync
{
    std::mutex mutex;
    std::condition_variable conditionVariable;
};

Waitable::Waitable()
    : mSync(new Sync)
    , mFinished(false)
{ }

Waitable::~Waitable()
{
    Wait();
    delete mSync;
}

void Waitable::Wait()
{
    std::unique_lock<std::mutex> lock(mSync->mutex);
    while (!mFinished)
    {
        mSync->conditionVariable.wait(lock);
    }
}

void Waitable::Reset()
{
    ASSERT(mFinished.load());

    mFinished = false;
}

void Waitable::OnFinished()
{
    std::unique_lock<std::mutex> lock(mSync->mutex);

    const bool oldState = mFinished.exchange(true);
    ASSERT(!oldState);
    (void)oldState;

    mSync->conditionVariable.notify_all();
}
