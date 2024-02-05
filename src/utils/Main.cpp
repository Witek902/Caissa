#include "Common.hpp"

#include "../backend/Tablebase.hpp"
#include "../backend/Evaluate.hpp"

#include <iostream>
#include <vector>
#include <string>

extern void RunUnitTests();
extern bool RunPerformanceTests(const std::vector<std::string>& paths);
extern void SelfPlay(const std::vector<std::string>& args);
extern void PrepareTrainingData(const std::vector<std::string>& args);
extern void PlainTextToTrainingData(const std::vector<std::string>& args);
extern void GenerateEndgamePositions();
extern bool TestNetwork();
extern bool TrainNetwork();
extern void ValidateEndgame();
extern void AnalyzeGames();

int main(int argc, const char* argv[])
{
#ifdef _MSC_VER
    // increase max open files limit (required for neural net training)
    _setmaxstdio(2048);
#endif // _MSC_VER

    std::vector<std::string> args;
    {
        for (int32_t i = 1; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }
    }

    InitEngine();
    TryLoadingDefaultEvalFile();

    // load optional syzygy
    for (size_t i = 0; i < args.size(); ++i)
    {
        if ((args[i] == "--syzygy") && (i + i < args.size()))
        {
            LoadSyzygyTablebase(args[i + 1].c_str());
            args.erase(args.begin() + i, args.begin() + i + 2);
        }
    }

    if (args.empty())
    {
        std::cerr << "Missing argument" << std::endl;
        return 1;
    }

    const std::string toolName = args[0];
    args.erase(args.begin());

    if (toolName == "unittest")
        RunUnitTests();
    else if (toolName == "perftest")
        RunPerformanceTests(args);
    else if (toolName == "selfplay")
        SelfPlay(args);
    else if (toolName == "prepareTrainingData")
        PrepareTrainingData(args);
    else if (toolName == "plainTextToTrainingData")
        PlainTextToTrainingData(args);
    else if (toolName == "testNetwork")
        TestNetwork();
    else if (toolName == "validateEndgame")
        ValidateEndgame();
    else if (toolName == "analyzeGames")
        AnalyzeGames();
    else if (toolName == "trainNetwork")
        TrainNetwork();
    else if (toolName == "generateEndgamePositions")
        GenerateEndgamePositions();
    else
    {
        std::cerr << "Unknown option: " << args[0] << std::endl;
        return 1;
    }

    return 0;
}
