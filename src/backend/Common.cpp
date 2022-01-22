#include "Common.hpp"

#include "TranspositionTable.hpp"
#include "Position.hpp"
#include "Endgame.hpp"
#include "Evaluate.hpp"

void InitEngine()
{
    TranspositionTable::Init();
    InitBitboards();
    InitZobristHash();
    InitEndgame();
    InitEvaluation();
}
