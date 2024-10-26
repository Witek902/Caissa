#include "Waitable.hpp"

Waitable::Waitable()
    : mFinished(false)
{ }

Waitable::~Waitable()
{
    Wait();
}

void Waitable::Wait()
{
    std::unique_lock<std::mutex> lock(mMutex);
    while (!mFinished)
    {
        mConditionVariable.wait(lock);
    }
}

void Waitable::Reset()
{
    ASSERT(mFinished.load());

    mFinished = false;
}

void Waitable::OnFinished()
{
    std::unique_lock<std::mutex> lock(mMutex);

    const bool oldState = mFinished.exchange(true);
    ASSERT(!oldState);
    (void)oldState;

    mConditionVariable.notify_all();
}
