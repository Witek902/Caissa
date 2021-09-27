#include "UCI.hpp"

int main(int argc, const char* argv[])
{
    InitEngine();

    UniversalChessInterface uci(argc, argv);
    uci.Loop();

    return 0;
}
