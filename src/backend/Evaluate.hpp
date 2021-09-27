#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include <math.h>
#include <algorithm>

// convert evaluation score (in pawns) into win probability
inline float PawnToWinProbability(float pawnsDifference)
{
    return 1.0f / (1.0f + powf(10.0, -pawnsDifference / 4.0f));
}

// convert win probability into evaluation score (in pawns)
inline float WinProbabilityToPawns(float w)
{
    w = std::clamp(w, 0.0f, 1.0f);
    return 4.0f * log10f(w / (1.0f - w));
}

int32_t ScoreQuietMove(const Position& position, const Move& move);
ScoreType Evaluate(const Position& position);
bool CheckInsufficientMaterial(const Position& position);
