#include "UCI.hpp"

#include "../backend/Position.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/TranspositionTable.hpp"

int main(int argc, const char* argv[])
{
    TranspositionTable::Init();
    InitBitboards(); 
    InitZobristHash();
    InitEndgame();

    UniversalChessInterface uci(argc, argv);
    uci.Loop();

    return 0;
}
