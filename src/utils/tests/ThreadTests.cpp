#include "TestCommon.hpp"
#include "../../backend/Thread.hpp"
#include "../../backend/Time.hpp"
#include "../../backend/containers/DynArray.hpp"

void RunThreadTests()
{
    std::cout << "Running thread tests..." << std::endl;

    TEST_EXPECT(Thread::GetNumLogicalProcessors() >= 1);

    // start and join, with a function pointer and with a lambda
    {
        uint32_t value = 0;
        Thread thread;
        TEST_EXPECT(!thread.IsRunning());
        TEST_EXPECT(thread.Start([](void* userData) { *static_cast<uint32_t*>(userData) = 42; }, &value));
        TEST_EXPECT(thread.IsRunning());
        thread.Join();
        TEST_EXPECT(!thread.IsRunning());
        TEST_EXPECT(value == 42);

        TEST_EXPECT(thread.Start([&value]() { value = 7; }));
        thread.Join();
        TEST_EXPECT(value == 7);
        thread.Join();
    }

    // mutex protects a counter incremented by several threads
    {
        Mutex mutex;
        uint64_t counter = 0;
        DynArray<Thread> threads(8);
        for (Thread& thread : threads)
        {
            TEST_EXPECT(thread.Start([&mutex, &counter]()
            {
                for (uint32_t i = 0; i < 10000; ++i)
                {
                    ScopedLock lock(mutex);
                    ++counter;
                }
            }));
        }
        for (Thread& thread : threads) thread.Join();
        TEST_EXPECT(counter == 80000);
    }

    // condition variable handoff: odd turns belong to the worker, even turns to this thread
    {
        Mutex mutex;
        ConditionVariable cv;
        uint32_t turn = 0;
        Thread worker;
        TEST_EXPECT(worker.Start([&mutex, &cv, &turn]()
        {
            for (uint32_t i = 0; i < 100; ++i)
            {
                ScopedLock lock(mutex);
                while (turn % 2 == 0) cv.Wait(mutex);
                ++turn;
                cv.NotifyAll();
            }
        }));
        for (uint32_t i = 0; i < 100; ++i)
        {
            ScopedLock lock(mutex);
            ++turn;
            cv.NotifyAll();
            while (turn % 2 == 1) cv.Wait(mutex);
        }
        worker.Join();
        TEST_EXPECT(turn == 200);
    }

    // timed wait without a notification times out after about the requested time
    {
        Mutex mutex;
        ConditionVariable cv;
        ScopedLock lock(mutex);
        const TimePoint start = TimePoint::GetCurrent();
        bool signaled = true;
        for (uint32_t i = 0; i < 10 && signaled; ++i) signaled = cv.WaitFor(mutex, 50);
        TEST_EXPECT(!signaled);
        TEST_EXPECT((TimePoint::GetCurrent() - start).ToSeconds() >= 0.03f);
    }
}
