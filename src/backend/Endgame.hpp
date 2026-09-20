#pragma once

#include "Common.hpp"

// enable collecting endgame stats
// #define COLLECT_ENDGAME_STATISTICS

// max number of pieces (kings excluded) for which the endgame evaluation is used
static constexpr int32_t c_endgameEvalMaxPieces = 6;

// Initialize endgame data
void InitEndgame();

// Try evaluate endgame position
// Returns false if no specialized evaluation function is available
// and must fallback to generic evaluation function
bool EvaluateEndgame(const Position& pos, int32_t& outScore);

#ifdef COLLECT_ENDGAME_STATISTICS
void PrintEndgameStatistics();
#endif // COLLECT_ENDGAME_STATISTICS
