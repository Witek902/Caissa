#include "Common.hpp"

#include "../backend/Tablebase.hpp"

int main(int argc, const char* argv[])
{
    (void)argc;
    (void)argv;

    InitEngine();

    LoadTablebase("C:\\Program Files (x86)\\syzygy\\");

    // TODO
    //SelfPlay();

    TrainEndgame();

    return 0;
}
