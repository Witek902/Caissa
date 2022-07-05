#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include "nnue-probe/nnue.h"

#include <math.h>
#include <algorithm>

struct PieceScore
{
    int16_t mg;
    int16_t eg;
};

static constexpr PieceScore c_pawnValue     = { 82, 94 };
static constexpr PieceScore c_knightValue   = { 337, 281 };
static constexpr PieceScore c_bishopValue   = { 365, 297 };
static constexpr PieceScore c_rookValue     = { 477, 512 };
static constexpr PieceScore c_queenValue    = { 1025, 936 };

static constexpr PieceScore c_pieceValues[] =
{
    {0,0},
    c_pawnValue,
    c_knightValue,
    c_bishopValue,
    c_rookValue,
    c_queenValue,
    {UINT16_MAX,UINT16_MAX},
};

void InitEvaluation();

// convert evaluation score (in pawns) into win probability
inline float PawnToWinProbability(float pawnsDifference)
{
    return 1.0f / (1.0f + powf(10.0, -pawnsDifference / 4.0f));
}

inline float CentiPawnToWinProbability(int32_t centiPawnsDifference)
{
    return PawnToWinProbability(static_cast<float>(centiPawnsDifference) * 0.01f);
}

// convert win probability into evaluation score (in pawns)
inline float WinProbabilityToPawns(float w)
{
    w = std::clamp(w, 0.0f, 1.0f);
    return 4.0f * log10f(w / (1.0f - w));
}

ScoreType Evaluate(const Position& position, NNUEdata** nnueData = nullptr);
bool CheckInsufficientMaterial(const Position& position);
