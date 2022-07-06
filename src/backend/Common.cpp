#include "Common.hpp"

#include "Memory.hpp"
#include "PositionHash.hpp"
#include "Endgame.hpp"
#include "Evaluate.hpp"
#include "SearchUtils.hpp"

void InitEngine()
{
    EnableLargePagesSupport();
    InitBitboards();
    InitZobristHash();
    InitEndgame();
    InitEvaluation();
    SearchUtils::Init();
}
