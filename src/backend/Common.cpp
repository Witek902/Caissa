#include "Common.hpp"

#include "TranspositionTable.hpp"
#include "Position.hpp"
#include "Endgame.hpp"

void InitEngine()
{
    TranspositionTable::Init();
    InitBitboards();
    InitZobristHash();
    InitEndgame();
}
