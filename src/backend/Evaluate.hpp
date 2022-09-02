#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include <math.h>
#include <algorithm>

extern const char* c_DefaultEvalFile;
extern const char* c_DefaultEndgameEvalFile;

struct PieceScore
{
    int16_t mg;
    int16_t eg;
};

extern const PieceScore PSQT[6][Square::NumSquares];

static constexpr PieceScore c_pawnValue     = { 127,    184 };
static constexpr PieceScore c_knightValue   = { 344,    474 };
static constexpr PieceScore c_bishopValue   = { 374,    508 };
static constexpr PieceScore c_rookValue     = { 548,    808 };
static constexpr PieceScore c_queenValue    = { 1064,   1513 };
static constexpr PieceScore c_kingValue     = { std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::max() };

static constexpr PieceScore c_pieceValues[] =
{
    {0,0},
    c_pawnValue,
    c_knightValue,
    c_bishopValue,
    c_rookValue,
    c_queenValue,
    c_kingValue,
};

// if abs(simpleEval) > nnTresholdMax, then we don't use NN at all
// if abs(simpleEval) < nnTresholdMin, then we use NN purely
// between the two values, the NN eval and simple eval are blended smoothly
static constexpr int32_t c_nnTresholdMin = 256;
static constexpr int32_t c_nnTresholdMax = 1024;

bool TryLoadingDefaultEvalFile();
bool TryLoadingDefaultEndgameEvalFile();

bool LoadMainNeuralNetwork(const char* path);
bool LoadEndgameNeuralNetwork(const char* path);


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
    ASSERT(w >= 0.0f && w <= 1.0f);
    w = std::clamp(w, 0.0f, 1.0f);
    return 4.0f * log10f(w / (1.0f - w));
}

// convert win probability to evaluation score (in pawns)
inline int32_t WinProbabilityToCentiPawns(float w)
{
    return (int32_t)std::round(100.0f * WinProbabilityToPawns(w));
}

ScoreType Evaluate(const Position& position, NodeInfo* node = nullptr, bool useNN = true);
bool CheckInsufficientMaterial(const Position& position);
