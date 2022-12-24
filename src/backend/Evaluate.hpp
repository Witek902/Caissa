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

    TPieceScore() = default;
    INLINE constexpr TPieceScore(const T _mg, const T _eg) : mg(_mg), eg(_eg) { }
    INLINE TPieceScore(const T* ptr) : mg(ptr[0]), eg(ptr[1]) { }

    template<typename T2>
    INLINE TPieceScore& operator += (const TPieceScore<T2>& rhs)
    {
        mg += rhs.mg;
        eg += rhs.eg;
        return *this;
    }

    template<typename T2>
    INLINE TPieceScore& operator -= (const TPieceScore<T2>& rhs)
    {
        mg -= rhs.mg;
        eg -= rhs.eg;
        return *this;
    }

    INLINE TPieceScore<int32_t> operator - (const TPieceScore rhs) const
    {
        return { mg - rhs.mg, eg - rhs.eg };
    }

    INLINE TPieceScore<int32_t> operator * (const int32_t rhs) const
    {
        return { mg * rhs, eg * rhs };
    }
};

using PieceScore = TPieceScore<int16_t>;

// not using array of PieceScore, because Visual Studio compiler can't pack that nicely as data section of EXE,
// but generates ugly initialization code instead
using KingsPerspectivePSQT = int16_t[10][2 * Square::NumSquares];
extern const KingsPerspectivePSQT PSQT[Square::NumSquares / 2];

static constexpr PieceScore c_pawnValue     = {  73,  141 };
static constexpr PieceScore c_knightValue   = { 291,  380 };
static constexpr PieceScore c_bishopValue   = { 322,  391 };
static constexpr PieceScore c_rookValue     = { 397,  639 };
static constexpr PieceScore c_queenValue    = { 931, 1139 };
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
static constexpr int32_t c_nnTresholdMax = 768;

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
