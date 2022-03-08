#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include "nnue-probe/nnue.h"

#include <math.h>
#include <algorithm>

void InitEvaluation();

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

ScoreType Evaluate(const Position& position, NNUEdata** nnueData = nullptr);
bool CheckInsufficientMaterial(const Position& position);
