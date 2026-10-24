#include "Waitable.hpp"
#include "Thread.hpp"

struct Waitable::Sync
{
    Mutex mutex;
    ConditionVariable conditionVariable;
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
    ScopedLock lock(mSync->mutex);
    while (!mFinished)
    {
        mSync->conditionVariable.Wait(mSync->mutex);
    }
}

void Waitable::Reset()
{
    ASSERT(mFinished.load());

    mFinished = false;
}

void Waitable::OnFinished()
{
    ScopedLock lock(mSync->mutex);

    const bool oldState = mFinished.exchange(true);
    ASSERT(!oldState);
    (void)oldState;

    mSync->conditionVariable.NotifyAll();
}
