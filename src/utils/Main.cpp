#include "Common.hpp"

#include "../backend/Tablebase.hpp"
#include "../backend/Evaluate.hpp"

#include <iostream>
#include <vector>
#include <string>

extern void RunUnitTests();
extern bool RunPerformanceTests(const std::vector<std::string>& paths);
extern void SelfPlay(const std::vector<std::string>& args);
extern bool TrainPieceSquareTables();
extern bool TrainEndgame();
extern void GenerateEndgamePieceSquareTables();
extern bool TestNetwork();
extern bool TrainNetwork();
extern void ValidateEndgame();
extern void AnalyzeGames();

int main(int argc, const char* argv[])
{
    std::vector<std::string> args;
    {
        for (int32_t i = 1; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }
    }

    InitEngine();
    TryLoadingDefaultEvalFile();
    TryLoadingDefaultEndgameEvalFile();

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

    if (args[0] == "unittest")
    {
        RunUnitTests();
    }
    else if (args[0] == "perftest")
    {
        RunPerformanceTests(args);
    }
    else if (args[0] == "selfplay")
    {
        SelfPlay(args);
    }
    else if (args[0] == "testNetwork")
    {
        TestNetwork();
    }
    else if (args[0] == "trainPieceSquareTables")
    {
        TrainPieceSquareTables();
    }
    else if (args[0] == "trainEndgame")
    {
        TrainEndgame();
    }
    else if (args[0] == "generateEndgamePST")
    {
        GenerateEndgamePieceSquareTables();
    }
    else if (args[0] == "validateEndgame")
    {
        ValidateEndgame();
    }
    else if (args[0] == "analyzeGames")
    {
        AnalyzeGames();
    }
    else if (args[0] == "trainNetwork")
    {
        TrainNetwork();
    }
    else
    {
		std::cerr << "Unknown option: " << args[0] << std::endl;
		return 1;
    }

    return 0;
}
