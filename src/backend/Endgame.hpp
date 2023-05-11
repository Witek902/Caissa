#pragma once

#include "Common.hpp"

// enable collecting endgame stats
// #define COLLECT_ENDGAME_STATISTICS

// Initialize endgame data
void InitEndgame();

static constexpr int32_t c_endgameScaleMax = 1024;

// Try evaluate endgame position
// Returns false if no specialized evaluation function is available
// and must fallback to generic evaluation function
bool EvaluateEndgame(const Position& pos, int32_t& outScore, int32_t& outScale);

#ifdef COLLECT_ENDGAME_STATISTICS
void PrintEndgameStatistics();
#endif // COLLECT_ENDGAME_STATISTICS
