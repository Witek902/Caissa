#pragma once

#include "Common.hpp"

class Position;

// Initialize endgame data
void InitEndgame();

// Try evaluate endgame position
// Returns false if no specialized evaluation function is available
// and must fallback to generic evaluation function
bool EvaluateEndgame(const Position& pos, int32_t& outScore);
