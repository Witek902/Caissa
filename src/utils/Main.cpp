#include "Common.hpp"

#include "../backend/Tablebase.hpp"

#include "../backend/nnue-probe/nnue.h"

#include <iostream>

extern void SelfPlay();
extern bool Train();
extern bool TrainEndgame();
extern void GenerateEndgamePieceSquareTables();
extern bool TestNetwork();

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Missing argument" << std::endl;
        return 1;
    }

    InitEngine();

    // HACK
    nnue_init("D:/CHESS/NNUE/nn-04cf2b4ed1da.nnue");

    //LoadTablebase("C:\\Program Files (x86)\\syzygy\\");

    if (0 == strcmp(argv[1], "selfplay"))
    {
        SelfPlay();
    }
    else if (0 == strcmp(argv[1], "testNetwork"))
    {
        TestNetwork();
    }
    else if (0 == strcmp(argv[1], "trainEndgame"))
    {
        TrainEndgame();
    }
    else if (0 == strcmp(argv[1], "generateEndgamePST"))
    {
        GenerateEndgamePieceSquareTables();
    }

    return 0;
}
