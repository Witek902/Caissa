#include "TestCommon.hpp"
#include "../ThreadPool.hpp"
#include "../../backend/Waitable.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace threadpool;

void RunUnitTests(const std::vector<std::string>& args)
{
    const bool deepPerft = std::find(args.begin(), args.end(), "--deepPerft") != args.end();

    RunContainerTests();
    RunThreadTests();
    RunBitboardTests();
    RunBitboardAdvancedTests();
    RunBitOperationTests();
    RunBitboardTransformTests();
    RunSlidingAttackTests();
    RunRayTests();
    RunPieceAttackDetailTests();
    RunSquareUtilityTests();
    RunPositionTests();
    RunPositionAdvancedTests();
    RunScoreTests();
    RunMoveListTests();
    RunTranspositionTableTests();
    RunTimeTests();
    RunTimeManagerTests();
    RunEvalTests();
    RunPackedPositionTests();
    RunGameTests();
    RunPgnParserTests();

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);
        taskBuilder.Task("Search", [](const TaskContext&) { RunSearchTests(); });
        RunPerftTests(taskBuilder, deepPerft);
    }
    waitable.Wait();
}
