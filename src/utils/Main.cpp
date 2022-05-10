#include "Common.hpp"

#include "../backend/Tablebase.hpp"

#include <iostream>
#include <vector>
#include <string>

extern void RunUnitTests();
extern bool RunPerformanceTests(const char* path);
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
    if (argc < 2)
    {
        std::cerr << "Missing argument" << std::endl;
        return 1;
    }

    std::vector<std::string> args;
    {
        for (int32_t i = 2; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }
    }

    InitEngine();

    LoadTablebase("C:\\Program Files (x86)\\syzygy\\");

    if (argc > 1 && strcmp(argv[1], "unittest") == 0)
    {
        RunUnitTests();
    }
    else if (argc > 2 && strcmp(argv[1], "perftest") == 0)
    {
        RunPerformanceTests(argv[2]);
    }
    else if (0 == strcmp(argv[1], "selfplay"))
    {
        SelfPlay(args);
    }
    else if (0 == strcmp(argv[1], "testNetwork"))
    {
        TestNetwork();
    }
    else if (0 == strcmp(argv[1], "trainPieceSquareTables"))
    {
        TrainPieceSquareTables();
    }
    else if (0 == strcmp(argv[1], "trainEndgame"))
    {
        TrainEndgame();
    }
    else if (0 == strcmp(argv[1], "generateEndgamePST"))
    {
        GenerateEndgamePieceSquareTables();
    }
    else if (0 == strcmp(argv[1], "validateEndgame"))
    {
        ValidateEndgame();
    }
    else if (0 == strcmp(argv[1], "analyzeGames"))
    {
        AnalyzeGames();
    }
    else if (0 == strcmp(argv[1], "trainNetwork"))
    {
        TrainNetwork();
    }

    system("pause");

    return 0;
}
