#include "ThreadPool.hpp"
#include "../backend/Waitable.hpp"
#include "minitrace/minitrace.h"

#include <assert.h>

namespace threadpool {

Task::Task()
{
    Reset();
}

void Task::Reset()
{
    mState = State::Invalid;
    mDependencyState = 0;
    mTasksLeft = 0;
    mParent = InvalidTaskID;
    mDependency = InvalidTaskID;
    mHead = InvalidTaskID;
    mTail = InvalidTaskID;
    mSibling = InvalidTaskID;
    mWaitable = nullptr;
    mDebugName = nullptr;
}

Task::Task(Task&& other)
{
    mState = other.mState.load();
    mTasksLeft = other.mTasksLeft.load();
    mParent = other.mParent;
    mDependency = other.mDependency;
    mHead = other.mHead;
    mTail = other.mTail;
    mSibling = other.mSibling;
    mWaitable = other.mWaitable;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

WorkerThread::WorkerThread(ThreadPool* pool, uint32_t id)
    : mThread{&ThreadPool::SchedulerCallback, pool, this}
    , mId(id)
    , mStarted(true)
{
}

WorkerThread::~WorkerThread()
{
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

static ThreadPool gThreadPool;

ThreadPool& ThreadPool::GetInstance()
{
    return gThreadPool;
}

ThreadPool::ThreadPool()
    : mFirstFreeTask(InvalidTaskID)
{
	mtr_init("trace.json");

    MTR_META_THREAD_NAME("Main Thread");

    // TODO make it configurable
    InitTasksTable(TasksCapacity);

    const uint32_t numThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency() - 1);
    SpawnWorkerThreads(numThreads);
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(mTasksQueueMutex);

        for (const WorkerThreadPtr& thread : mThreads)
        {
            thread->mStarted = false;
        }

        mTaskQueueCV.notify_all();
    }

    for (const WorkerThreadPtr& thread : mThreads)
    {
        thread->mThread.join();
    }

	mtr_flush();
	mtr_shutdown();
}

bool ThreadPool::InitTasksTable(uint32_t newSize)
{
    mTasks.resize(newSize);

    mFirstFreeTask = 0;
    for (uint32_t i = 0; i < newSize - 1; ++i)
    {
        mTasks[i].mNextFree = i + 1;
    }
    mTasks[newSize - 1].mNextFree = InvalidTaskID;

    return true;
}

void ThreadPool::SpawnWorkerThreads(uint32_t num)
{
    for (uint32_t i = 0; i < num; ++i)
    {
        mThreads.emplace_back(std::make_unique<WorkerThread>(this, i));
    }
}

void ThreadPool::SchedulerCallback(WorkerThread* thread)
{
    TaskContext context;
    context.pool = this;
    context.threadId = thread->mId;

    char threadName[16];
    sprintf(threadName, "Worker %u", thread->mId);
    MTR_META_THREAD_NAME(threadName);

    for (;;)
    {
        Task* task = nullptr;
        {
            std::unique_lock<std::mutex> lock(mTasksQueueMutex);

            std::deque<TaskID>* queue = nullptr;

            // wait for new task
            while (thread->mStarted)
            {
                // find queue with pending tasks
                queue = nullptr;
                for (uint32_t i = 0; i < NumPriorities; ++i)
                {
                    if (!mTasksQueues[i].empty())
                    {
                        queue = &mTasksQueues[i];
                        break;
                    }
                }

                if (queue != nullptr)
                {
                    break;
                }

                mTaskQueueCV.wait(lock);
            }

            if (!thread->mStarted)
            {
                break;
            }

            // pop a task from the queue
            context.taskId = queue->front();
            task = &mTasks[context.taskId];
            queue->pop_front();
        }

        if (task->mCallback)
        {
            // Queued -> Executing
            {
                const Task::State oldState = task->mState.exchange(Task::State::Executing);
                assert(Task::State::Queued == oldState); // Task is expected to be in 'Queued' state
                (void)oldState;
            }

            MTR_BEGIN("Task", task->mDebugName);

            // execute
            task->mCallback(context);

            MTR_END("Task", task->mDebugName);

            // Executing -> Finished
            {
                const Task::State oldState = task->mState.exchange(Task::State::Finished);
                assert(Task::State::Executing == oldState); // Task is expected to be in 'Executing' state
                (void)oldState;
            }
        }
        else
        {
            // Queued -> Finished

            const Task::State oldState = task->mState.exchange(Task::State::Finished);
            assert(Task::State::Queued == oldState); // Task is expected to be in 'Queued' state
            (void)oldState;
        }

        FinishTask(context.taskId);
    }
}

void ThreadPool::FinishTask(TaskID taskID)
{
    TaskID taskToFinish = taskID;

    // Note: loop instead of recursion to avoid stack overflow in case of long dependency chains
    while (taskToFinish != InvalidTaskID)
    {
        TaskID parentTask;
        Waitable* waitable = nullptr;

        {
            // TODO lockless
            std::unique_lock<std::mutex> lock(mTaskListMutex);

            Task& task = mTasks[taskToFinish];

            parentTask = task.mParent;
            waitable = task.mWaitable;

            const int32_t tasksLeft = --task.mTasksLeft;
            assert(tasksLeft >= 0); // Tasks counter underflow
            if (tasksLeft > 0)
            {
                return;
            }

            // notify about fullfilling the dependency
            {
                TaskID siblingID = task.mHead;
                while (siblingID != InvalidTaskID)
                {
                    OnTaskDependencyFullfilled_NoLock(siblingID);
                    siblingID = mTasks[siblingID].mSibling;
                }
            }

            FreeTask_NoLock(taskToFinish);
        }

        // notify waitable object
        if (waitable)
        {
            waitable->OnFinished();
        }

        // update parent (without recursion)
        taskToFinish = parentTask;
    }
}

void ThreadPool::EnqueueTaskInternal_NoLock(TaskID taskID)
{
    Task& task = mTasks[taskID];

    const Task::State oldState = task.mState.exchange(Task::State::Queued);
    assert(Task::State::Created == oldState); // Task is expected to be in 'Created' state
    assert((Task::Flag_IsDispatched | Task::Flag_DependencyFullfilled) == task.mDependencyState);
    (void)oldState;

    // push to queue
    {
        std::unique_lock<std::mutex> lock(mTasksQueueMutex);
        mTasksQueues[task.mPriority].push_back(taskID);
        mTaskQueueCV.notify_all();
    }
}

void ThreadPool::FreeTask_NoLock(TaskID taskID)
{
    assert(taskID < mTasks.size());

    Task& task = mTasks[taskID];

    const Task::State oldState = task.mState.exchange(Task::State::Invalid);
    assert(Task::State::Finished == oldState); // Task is expected to be in 'Finished' state
    (void)oldState;

    task.mNextFree = mFirstFreeTask;
    mFirstFreeTask = taskID;
}

TaskID ThreadPool::AllocateTask_NoLock()
{
    if (mFirstFreeTask == InvalidTaskID)
    {
        return InvalidTaskID;
    }

    Task& task = mTasks[mFirstFreeTask];

    const Task::State oldState = task.mState.exchange(Task::State::Queued);
    assert(Task::State::Invalid == oldState); // Task is expected to be in 'Invalid' state
    (void)oldState;

    TaskID newNextFree = task.mNextFree;
    TaskID taskID = mFirstFreeTask;
    mFirstFreeTask = newNextFree;
    return taskID;
}

TaskID ThreadPool::CreateTask(const TaskDesc& desc)
{
    assert(desc.priority < NumPriorities);

    // TODO lockless
    std::unique_lock<std::mutex> lock(mTaskListMutex);

    TaskID taskID = AllocateTask_NoLock();
    assert(taskID != InvalidTaskID);

    if (taskID == InvalidTaskID)
    {
        return InvalidTaskID;
    }

    Task& task = mTasks[taskID];
    task.Reset();
    task.mDependencyState = 0;
    task.mPriority = desc.priority;
    task.mTasksLeft = 1;
    task.mCallback = desc.function;
    task.mParent = desc.parent;
    task.mDependency = desc.dependency;
    task.mWaitable = desc.waitable;
    task.mHead = InvalidTaskID;
    task.mState = Task::State::Created;
    task.mDebugName = desc.debugName;

    if (desc.parent != InvalidTaskID)
    {
        mTasks[desc.parent].mTasksLeft++;
    }

    TaskID dependencyID = mTasks[taskID].mDependency;

    bool dependencyFullfilled = true;
    if (dependencyID != InvalidTaskID)
    {
        Task& dependency = mTasks[dependencyID];

        assert(Task::State::Invalid != dependency.mState); // Invalid state of dependency task

        //NFE_SCOPED_LOCK(mFinishedTasksMutex); // TODO: how to get rid of it?
        if (dependency.mTasksLeft > 0)
        {
            // update dependency list
            if (dependency.mTail != InvalidTaskID)
            {
                mTasks[dependency.mTail].mSibling = taskID;
            }
            else
            {
                dependency.mHead = taskID;
            }

            dependency.mTail = taskID;
            dependencyFullfilled = false;
        }
    }

    if (dependencyFullfilled)
    {
        task.mDependencyState = Task::Flag_DependencyFullfilled;
    }

    return taskID;
}

void ThreadPool::DispatchTask(TaskID taskID)
{
    assert(taskID != InvalidTaskID);

    // TODO lockless
    std::unique_lock<std::mutex> lock(mTaskListMutex);

    Task& task = mTasks[taskID];

    assert(Task::State::Created == task.mState); // Task is expected to be in 'Created' state

    const uint8_t oldDependencyState = task.mDependencyState.fetch_or(Task::Flag_IsDispatched);

    assert((oldDependencyState & Task::Flag_IsDispatched) == 0); // Task already dispatched

    // can enqueue only if not dispatched yet, but dependency was fullfilled
    if (Task::Flag_DependencyFullfilled == oldDependencyState)
    {
        EnqueueTaskInternal_NoLock(taskID);
    }
}

void ThreadPool::OnTaskDependencyFullfilled_NoLock(TaskID taskID)
{
    assert(taskID != InvalidTaskID);

    Task& task = mTasks[taskID];

    assert(Task::State::Created == task.mState); // Task is expected to be in 'Created' state

    const uint8_t oldDependencyState = task.mDependencyState.fetch_or(Task::Flag_DependencyFullfilled);

    assert((oldDependencyState & Task::Flag_DependencyFullfilled) == 0); // Task should not have dependency fullfilled

    // can enqueue only if was dispatched
    if (Task::Flag_IsDispatched == oldDependencyState)
    {
        EnqueueTaskInternal_NoLock(taskID);
    }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

TaskBuilder::TaskBuilder(const TaskID parentTask)
    : mParentTask(parentTask)
{
}

TaskBuilder::TaskBuilder(const TaskContext& taskContext)
    : mParentTask(taskContext.taskId)
{
}

TaskBuilder::TaskBuilder(Waitable& waitable)
    : mWaitable(&waitable)
{
}

TaskBuilder::~TaskBuilder()
{
    ThreadPool& tp = ThreadPool::GetInstance();

    if (mWaitable)
    {
        Fence(mWaitable);
    }

    // flush previous dependency
    if (mDependencyTask != InvalidTaskID)
    {
        tp.DispatchTask(mDependencyTask);
        mDependencyTask = InvalidTaskID;
    }

    for (uint32_t i = 0; i < mNumPendingTasks; ++i)
    {
        const TaskID pendingTask = mPendingTasks[i];
        tp.DispatchTask(pendingTask);
    }
}

void TaskBuilder::Fence(Waitable* waitable)
{
    ThreadPool& tp = ThreadPool::GetInstance();

    // flush previous dependency
    if (mDependencyTask != InvalidTaskID)
    {
        tp.DispatchTask(mDependencyTask);
        mDependencyTask = InvalidTaskID;
    }

    TaskDesc depDesc;
    depDesc.debugName = "TaskBuilder::Fence";
    depDesc.waitable = waitable;

    TaskID dependency = tp.CreateTask(depDesc);

    // flush pending tasks and link them to dependency task
    for (uint32_t i = 0; i < mNumPendingTasks; ++i)
    {
        TaskDesc desc;
        desc.debugName = "TaskBuilder::Fence/Sub";
        desc.parent = dependency;
        desc.dependency = mPendingTasks[i];
        tp.CreateAndDispatchTask(desc);

        tp.DispatchTask(mPendingTasks[i]);
    }
    mNumPendingTasks = 0;

    mDependencyTask = dependency;
}

void TaskBuilder::Task(const char* debugName, const TaskFunction& func)
{
    ThreadPool& tp = ThreadPool::GetInstance();

    TaskDesc desc;
    desc.function = func;
    desc.debugName = debugName;
    desc.parent = mParentTask;
    desc.dependency = mDependencyTask;

    TaskID taskID = tp.CreateTask(desc);
    mPendingTasks[mNumPendingTasks++] = taskID;
}

void TaskBuilder::CustomTask(TaskID customTask)
{
    ThreadPool& tp = ThreadPool::GetInstance();

    TaskDesc desc;
    desc.debugName = "TaskBuilder::CustomTask";
    desc.parent = mParentTask;
    desc.dependency = customTask;

    TaskID taskID = tp.CreateTask(desc);
    mPendingTasks[mNumPendingTasks++] = taskID;
}

void TaskBuilder::ParallelFor(const char* debugName, uint32_t arraySize, const ParallelForTaskFunction& func)
{
    ThreadPool& tp = ThreadPool::GetInstance();

    if (arraySize == 0)
    {
        return;
    }

    TaskDesc desc;
    desc.debugName = debugName;
    desc.parent = mParentTask;
    desc.dependency = mDependencyTask;

    TaskID parallelForTask = tp.CreateTask(desc);
    mPendingTasks[mNumPendingTasks++] = parallelForTask;

    const uint32_t numThreads = std::thread::hardware_concurrency();
    uint32_t numTasksToSpawn = std::min(arraySize, numThreads);

    struct alignas(64) ThreadData
    {
        uint32_t elementOffset = 0; // base element
        uint32_t numElements = 0;
        std::atomic<int32_t> counter = 0;
        uint32_t threadDataIndex = 0;

        ThreadData() = default;
        ThreadData(const ThreadData & other)
            : elementOffset(other.elementOffset)
            , numElements(other.numElements)
            , counter(other.counter.load())
            , threadDataIndex(other.threadDataIndex)
        {}
    };

    // TODO get rid of dynamic allocation, e.g. by using some kind of pool
    using ThreadDataPtr = std::shared_ptr<std::vector<ThreadData>>;
    ThreadDataPtr threadDataPtr = std::make_shared<std::vector<ThreadData>>();
    threadDataPtr->resize(numThreads);

    // subdivide work
    {
        uint32_t totalElements = 0;
        for (uint32_t i = 0; i < numTasksToSpawn; ++i)
        {
            ThreadData& threadData = (*threadDataPtr)[i];
            threadData.numElements = (arraySize / numTasksToSpawn) + ((arraySize % numTasksToSpawn > i) ? 1 : 0);
            threadData.elementOffset = totalElements;
            totalElements += threadData.numElements;
        }
    }

    for (uint32_t i = 0; i < numTasksToSpawn; ++i)
    {
        TaskDesc subTaskDesc;
        subTaskDesc.debugName = debugName;
        subTaskDesc.parent = parallelForTask;
        subTaskDesc.dependency = mDependencyTask;
        subTaskDesc.function = [func, threadDataPtr, numTasksToSpawn](const TaskContext& context)
        {
            // consume elements assigned to each thread (starting from self)
            for (uint32_t threadDataOffset = 0; threadDataOffset < numTasksToSpawn; ++threadDataOffset)
            {
                uint32_t threadDataIndex = context.threadId + threadDataOffset;
                if (threadDataIndex >= numTasksToSpawn)
                {
                    threadDataIndex -= numTasksToSpawn;
                }

                ThreadData& threadData = (*threadDataPtr)[threadDataIndex];

                uint32_t index;
                while ((index = threadData.counter++) < threadData.numElements)
                {
                    func(context, static_cast<uint32_t>(threadData.elementOffset + index));
                }
            }
        };

        tp.CreateAndDispatchTask(subTaskDesc);
    }
}

} // namespace threadpool
