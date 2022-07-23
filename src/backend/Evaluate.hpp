#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include <math.h>
#include <algorithm>

extern const char* c_DefaultEvalFile;

struct PieceScore
{
    int16_t mg;
    int16_t eg;
};

static constexpr PieceScore c_pawnValue     = { 127,    184 };
static constexpr PieceScore c_knightValue   = { 344,    474 };
static constexpr PieceScore c_bishopValue   = { 374,    508 };
static constexpr PieceScore c_rookValue     = { 548,    808 };
static constexpr PieceScore c_queenValue    = { 1064,   1513 };

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

bool TryLoadingDefaultEvalFile();
bool LoadMainNeuralNetwork(const char* path);


// scaling factor when converting from neural network output (logistic space) to centipawn value
// equal to 400/ln(10) = 173.7177...
static constexpr int32_t c_nnOutputToCentiPawns = 174;

// convert evaluation score (in pawns) to win probability
inline float PawnToWinProbability(float pawnsDifference)
{
    return 1.0f / (1.0f + powf(10.0, -pawnsDifference / 4.0f));
}

// convert evaluation score (in centipawns) to win probability
inline float CentiPawnToWinProbability(int32_t centiPawnsDifference)
{
    return PawnToWinProbability(static_cast<float>(centiPawnsDifference) * 0.01f);
}

// convert win probability to evaluation score (in pawns)
inline float WinProbabilityToPawns(float w)
{
    w = std::clamp(w, 0.0f, 1.0f);
    return 4.0f * log10f(w / (1.0f - w));
}

ScoreType Evaluate(const Position& position, NodeInfo* node = nullptr);
bool CheckInsufficientMaterial(const Position& position);
