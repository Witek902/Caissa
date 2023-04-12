#pragma once

#include "Position.hpp"
#include "Move.hpp"
#include "Score.hpp"

#include <math.h>
#include <algorithm>

// TODO re-enable once new net is generated
// #define USE_ENDGAME_NEURAL_NETWORK

struct DirtyPiece;

extern const char* c_DefaultEvalFile;
extern const char* c_DefaultEndgameEvalFile;

// not using array of PieceScore, because Visual Studio compiler can't pack that nicely as data section of EXE,
// but generates ugly initialization code instead
using KingsPerspectivePSQT = int16_t[10][2 * Square::NumSquares];
extern const KingsPerspectivePSQT PSQT[Square::NumSquares / 2];

static constexpr PieceScore c_pawnValue     = {   97, 166 };
static constexpr PieceScore c_knightValue   = {  455, 371 };
static constexpr PieceScore c_bishopValue   = {  494, 385 };
static constexpr PieceScore c_rookValue     = {  607, 656 };
static constexpr PieceScore c_queenValue    = { 1427,1086 };
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
static constexpr int32_t c_nnTresholdMax = 768;

bool TryLoadingDefaultEvalFile();
bool LoadMainNeuralNetwork(const char* path);

#ifdef USE_ENDGAME_NEURAL_NETWORK
bool TryLoadingDefaultEndgameEvalFile();
bool LoadEndgameNeuralNetwork(const char* path);
#endif // USE_ENDGAME_NEURAL_NETWORK

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
    if (w > 0.99999f)
        return INT32_MAX;
    else if (w < 0.00001f)
        return -INT32_MAX;
    else
        return (int32_t)std::round(100.0f * WinProbabilityToPawns(w));
}

const TPieceScore<int32_t> ComputePSQT(const Position& pos);
void ComputeIncrementalPSQT(TPieceScore<int32_t>& score, const Position& pos, const DirtyPiece* dirtyPieces, uint32_t numDirtyPieces);

ScoreType Evaluate(const Position& position, NodeInfo* node = nullptr, bool useNN = true);
bool CheckInsufficientMaterial(const Position& position);
