#pragma once

#include <inttypes.h>
#include <thread>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>
#include <vector>
#include <deque>

namespace threadpool {

class Waitable;
class ThreadPool;

/**
 * Thread pool task unique identifier.
 */
using TaskID = uint32_t;

static constexpr TaskID InvalidTaskID = UINT32_MAX;

/**
 * Task execution context.
 */
struct TaskContext
{
    ThreadPool* pool;
    uint32_t threadId;  // thread ID (counted from 0)
    TaskID taskId;      // this task ID
};

// Function object representing a task.
using TaskFunction = std::function<void(const TaskContext& context)>;

// Parallel-for callback
using ParallelForTaskFunction = std::function<void(const TaskContext& context, uint32_t arrayIndex)>;


// Structure describing task, used during Task creation.
struct TaskDesc
{
    TaskFunction function;

    // waitable object (optional)
    Waitable* waitable = nullptr;

    // parent task to append to (optional)
    TaskID parent = InvalidTaskID;

    // dependency task (optional)
    TaskID dependency = InvalidTaskID;

    // Specifies target queue
    // Task with higher priority are always popped from the queues first.
    // Valid range is 0...(ThreadPool::NumPriorities-1)
    uint8_t priority = 1;

    const char* debugName = nullptr;

    // TODO limiting number of parallel running tasks of certain type
    // (eg. long-running resource loading tasks)

    TaskDesc() = default;
    TaskDesc(const TaskFunction& func) : function(func) { }
};

/**
 * Helper class allowing for waiting for an event.
 */
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

    void OnFinished();

private:

    Waitable(const Waitable&) = delete;
    Waitable(Waitable&&) = delete;

    std::mutex mMutex;
    std::condition_variable mConditionVariable;
    std::atomic<bool> mFinished;
};

/**
 * @brief Internal task structure.
 * @remarks Only @p ThreadPool class can access it.
 */
class Task final
{
    friend class ThreadPool;

public:

    enum class State : uint8_t
    {
        Invalid,    // unused task table entries are in invalid state
        Created,    // created task, waiting for a dependency to be fulfilled
        Queued,     // a task with fulfilled dependency, waiting in queue for execution
        Executing,  // task is being executed right now
        Finished,
    };

    static const uint8_t Flag_IsDispatched = 1;
    static const uint8_t Flag_DependencyFullfilled = 2;

    TaskFunction mCallback;  //< task routine

    std::atomic<State> mState;
    std::atomic<uint8_t> mDependencyState;

    // Number of sub-tasks left to complete.
    // If reaches 0, then whole task is considered as finished.
    std::atomic<int32_t> mTasksLeft;

    union
    {
        TaskID mParent;
        TaskID mNextFree;   //< free tasks list
    };

    // optional waitable object (it gets notified in the task is finished)
    Waitable* mWaitable;

    const char* mDebugName;

    // Dependency pointers:
    TaskID mDependency;  //< dependency tasks ID
    TaskID mHead;        //< the first task that is dependent on this task
    TaskID mTail;        //< the last task that is dependent on this task
    TaskID mSibling;     //< the next task that is dependent on the same "mDependency" task

    uint8_t mPriority;

    // TODO: alignment

    Task();
    Task(Task&& other);
    void Reset();
};

// Thread pool's worker thread
class WorkerThread
{
    friend class ThreadPool;

    std::thread mThread;
    uint32_t mId;                     // thread number
    std::atomic<bool> mStarted;     // if set to false, exit the thread

public:
    WorkerThread(ThreadPool* pool, uint32_t id);
    ~WorkerThread();
};

using WorkerThreadPtr = std::unique_ptr<WorkerThread>;


/**
 * @class ThreadPool
 * @brief Class enabling parallel tasks execution.
 */
class ThreadPool final
{
    friend class WorkerThread;

public:

    static constexpr uint32_t TasksCapacity = 1024 * 128;
    static constexpr uint32_t NumPriorities = 3;
    static constexpr uint32_t MaxPriority = NumPriorities - 1;

    ThreadPool();
    ~ThreadPool();

    static ThreadPool& GetInstance();

    // Create a new task.
    // The task will not be queued immidiately - it has to be queued manually via DispatchTask call
    // NOTE This function is thread-safe.
    TaskID CreateTask(const TaskDesc& desc);

    // Dispatch a created task for being executed.
    // NOTE: Using the task ID after dispatching the task is undefined behaviour.
    void DispatchTask(const TaskID taskID);

    void CreateAndDispatchTask(const TaskDesc& desc)
    {
        DispatchTask(CreateTask(desc));
    }

private:

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;

    void SchedulerCallback(WorkerThread* thread);

    TaskID AllocateTask_NoLock();
    void FreeTask_NoLock(TaskID taskID);
    void FinishTask(TaskID taskID);
    void EnqueueTaskInternal_NoLock(TaskID taskID);
    void OnTaskDependencyFullfilled_NoLock(TaskID taskID);

    // create "num" additional worker threads
    void SpawnWorkerThreads(uint32_t num);

    bool InitTasksTable(uint32_t newSize);

    // Worker threads variables:
    std::vector<WorkerThreadPtr> mThreads;

    // queues for tasks with "Queued" state
    std::deque<TaskID> mTasksQueues[NumPriorities];
    std::mutex mTasksQueueMutex;                 //< lock for "mTasksQueue" access
    std::condition_variable mTaskQueueCV;         //< CV for notifying about a new task in the queue

    std::mutex mTaskListMutex;
    std::vector<Task> mTasks; // TODO growable fixed-size allocator
    TaskID mFirstFreeTask;
};

// helper class that allows easy task-graph building
class TaskBuilder
{
public:
    static constexpr uint32_t MaxTasks = 128;

    TaskBuilder(const TaskID parentTask = InvalidTaskID);
    explicit TaskBuilder(const TaskContext& taskContext);
    explicit TaskBuilder(Waitable& waitable);
    ~TaskBuilder();

    // push a new task
    // Note: multiple pushed tasks can run in parallel
    void Task(const char* debugName, const TaskFunction& func);

    // Push a custom task
    // Note: The task must be created, but not yet dispatched
    void CustomTask(TaskID customTask);

    // push parallel-for task
    void ParallelFor(const char* debugName, uint32_t arraySize, const ParallelForTaskFunction& func);

    // Push a sync point
    // All tasks pushed after the fence will start only when all the tasks pushed before the fence finish execution
    // Optionally signals waitable object
    void Fence(Waitable* waitable = nullptr);

private:
    // forbid dynamic allocation (only stack allocation is allowed)
    void* operator new(size_t size) = delete;
    void operator delete(void* ptr) = delete;
    void* operator new[](size_t size) = delete;
    void operator delete[](void* ptr) = delete;
    void* operator new(size_t size, void* ptr) = delete;
    void* operator new[](size_t size, void* ptr) = delete;

    Waitable* mWaitable = nullptr;
    const TaskID mParentTask = InvalidTaskID;
    TaskID mDependencyTask = InvalidTaskID;

    // tasks that has to be synchronized after instering a fence or synchronizing with waitable object
    TaskID mPendingTasks[MaxTasks];
    uint32_t mNumPendingTasks = 0;
};

} // namespace threadpool
