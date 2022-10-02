#include "UCI.hpp"

int main(int argc, const char* argv[])
{
    InitEngine();

    UniversalChessInterface uci;
    uci.Loop(argc, argv);

    return 0;
}
