#include "UCI.hpp"

#include "../backend/Position.hpp"
#include "../backend/Endgame.hpp"

#include <iostream>

int main(int argc, const char* argv[])
{
    InitBitboards(); 
    InitZobristHash();
    InitEndgame();

    UniversalChessInterface uci(argc, argv);
    uci.Loop();

    return 0;
}
