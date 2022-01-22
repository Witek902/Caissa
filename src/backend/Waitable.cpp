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
    if (!mFinished)
    {
        std::unique_lock<std::mutex> lock(mMutex);
        while (!mFinished)
        {
            mConditionVariable.wait(lock);
        }
    }
}

void Waitable::Reset()
{
    ASSERT(mFinished.load());

    mFinished = false;
}

void Waitable::OnFinished()
{
    const bool oldState = mFinished.exchange(true);
    ASSERT(!oldState);
    (void)oldState;

    std::unique_lock<std::mutex> lock(mMutex);
    mConditionVariable.notify_all();
}
