#include "TestCommon.hpp"
#include "../ThreadPool.hpp"
#include "../../backend/Evaluate.hpp"
#include "../../backend/Waitable.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

using namespace threadpool;

void RunUnitTests(const std::vector<std::string>& args)
{
    const bool deepPerft = std::find(args.begin(), args.end(), "--deepPerft") != args.end();

    if (!g_mainNeuralNetwork)
    {
        std::cout << "No neural network loaded, pass one with --evalFile <path>" << std::endl;
        exit(1);
    }

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
    RunThreatInputTests();
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
