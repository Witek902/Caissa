#pragma once

#include "Common.hpp"

// enable collecting endgame stats
// #define COLLECT_ENDGAME_STATISTICS

// Initialize endgame data
void InitEndgame();

// Try evaluate endgame position
// Returns false if no specialized evaluation function is available
// and must fallback to generic evaluation function
bool EvaluateEndgame(const Position& pos, int32_t& outScore);

#ifdef COLLECT_ENDGAME_STATISTICS
void PrintEndgameStatistics();
#endif // COLLECT_ENDGAME_STATISTICS
