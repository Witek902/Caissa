#include "Common.hpp"

#include "Memory.hpp"
#include "Position.hpp"
#include "Endgame.hpp"
#include "Evaluate.hpp"

void InitEngine()
{
    EnableLargePagesSupport();
    InitBitboards();
    InitZobristHash();
    InitEndgame();
    InitEvaluation();
}
