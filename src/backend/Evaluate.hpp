#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include <math.h>
#include <algorithm>

extern const char* c_DefaultEvalFile;
extern const char* c_DefaultEndgameEvalFile;

template<typename T>
struct TPieceScore
{
    T mg;
    T eg;

    template<typename T2>
    TPieceScore& operator += (const TPieceScore<T2>& rhs)
    {
        mg += rhs.mg;
        eg += rhs.eg;
        return *this;
    }

    template<typename T2>
    TPieceScore& operator -= (const TPieceScore<T2>& rhs)
    {
        mg -= rhs.mg;
        eg -= rhs.eg;
        return *this;
    }

    TPieceScore<int32_t> operator - (const TPieceScore rhs) const
    {
        return { mg - rhs.mg, eg - rhs.eg };
    }

    TPieceScore<int32_t> operator * (const int32_t rhs) const
    {
        return { mg * rhs, eg * rhs };
    }
};

using PieceScore = TPieceScore<int16_t>;

extern const PieceScore PSQT[6][Square::NumSquares];

extern const PieceScore c_ourPawnDistanceBonus[8];
extern const PieceScore c_ourKnightDistanceBonus[8];
extern const PieceScore c_ourBishopDistanceBonus[8];
extern const PieceScore c_ourRookDistanceBonus[8];
extern const PieceScore c_ourQueenDistanceBonus[8];
extern const PieceScore c_theirPawnDistanceBonus[8];
extern const PieceScore c_theirKnightDistanceBonus[8];
extern const PieceScore c_theirBishopDistanceBonus[8];
extern const PieceScore c_theirRookDistanceBonus[8];
extern const PieceScore c_theirQueenDistanceBonus[8];

static constexpr PieceScore c_pawnValue     = {   95, 161};
static constexpr PieceScore c_knightValue   = {  427, 312};
static constexpr PieceScore c_bishopValue   = {  420, 358};
static constexpr PieceScore c_rookValue     = {  572, 621};
static constexpr PieceScore c_queenValue    = { 1326,1059 };
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
static constexpr int32_t c_nnTresholdMin = 512;
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
