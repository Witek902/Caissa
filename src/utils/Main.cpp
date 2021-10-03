#include "Common.hpp"

#include "../backend/Tablebase.hpp"

#include "../backend/nnue-probe/nnue.h"

int main(int argc, const char* argv[])
{
    (void)argc;
    (void)argv;

    InitEngine();

    // HACK
    nnue_init("D:/CHESS/NNUE/nn-04cf2b4ed1da.nnue");

    LoadTablebase("C:\\Program Files (x86)\\syzygy\\");

    // TODO
    //SelfPlay();

    TrainEndgame();

    return 0;
}
