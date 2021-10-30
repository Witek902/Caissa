#include "Common.hpp"

#include "../backend/Tablebase.hpp"

#include "../backend/nnue-probe/nnue.h"

extern void SelfPlay();
extern bool Train();
extern bool TrainEndgame();
extern void GenerateEndgamePieceSquareTables();

int main(int argc, const char* argv[])
{
    (void)argc;
    (void)argv;

    InitEngine();

    // HACK
    //nnue_init("D:/CHESS/NNUE/nn-04cf2b4ed1da.nnue");

    LoadTablebase("C:\\Program Files (x86)\\syzygy\\");

    // TODO
    //SelfPlay();
    TrainEndgame();
    //GenerateEndgamePieceSquareTables();

    return 0;
}
