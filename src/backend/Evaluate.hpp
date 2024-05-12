#pragma once

#include "Position.hpp"
#include "Move.hpp"
#include "Score.hpp"

#include <math.h>
#include <algorithm>
#include <memory>

namespace nn
{
class PackedNeuralNetwork;
}
using PackedNeuralNetworkPtr = std::unique_ptr<nn::PackedNeuralNetwork>;

struct DirtyPiece;
struct AccumulatorCache;

extern const char* c_DefaultEvalFile;

extern PackedNeuralNetworkPtr g_mainNeuralNetwork;

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

bool TryLoadingDefaultEvalFile();
bool LoadMainNeuralNetwork(const char* path);

// scaling factor when converting from neural network output (logistic space) to centipawn value
static constexpr int32_t c_nnOutputToCentiPawns = 512;

namespace wld
{
    // WLD model by Vondele
    // coefficients computed with https://github.com/vondele/WLD_model on 60+0.6s games
    constexpr double as[] = {   1.88054041,    2.39467539,   -3.78019886,  153.49644002 };
    constexpr double bs[] = {  -4.24993229,   31.21455804,  -59.09168702,   68.89719592 };
    constexpr int32_t NormalizeToPawnValue = 153;
}

inline int32_t NormalizeEval(int32_t eval)
{
    if (eval >= KnownWinValue || eval <= -KnownWinValue)
        return eval;
    else
        return eval * 100 / wld::NormalizeToPawnValue;
}

// convert evaluation score (in pawns) to win probability
inline float EvalToWinProbability(float eval, uint32_t ply)
{
    using namespace wld;
    const double m = std::min(240u, ply) / 64.0;
    const double a = ((as[0] * m + as[1]) * m + as[2]) * m + as[3];
    const double b = ((bs[0] * m + bs[1]) * m + bs[2]) * m + bs[3];
    return static_cast<float>(1.0 / (1.0 + exp((a - 100.0 * eval) / b)));
}

// convert evaluation score (in pawns) to draw probability
inline float EvalToDrawProbability(float eval, uint32_t ply)
{
    const float winProb = EvalToWinProbability(eval, ply);
    const float lossProb = EvalToWinProbability(-eval, ply);
    return 1.0f - winProb - lossProb;
}

// convert evaluation score (in pawns) to expected game score
inline float EvalToExpectedGameScore(float eval)
{
    return 1.0f / (1.0f + powf(10.0, -eval / 4.0f));
}

// convert evaluation score (in centipawns) to expected game score
inline float InternalEvalToExpectedGameScore(int32_t eval)
{
    return EvalToExpectedGameScore(static_cast<float>(eval) * 0.01f);
}

// convert expected game score to evaluation score (in pawns)
inline float ExpectedGameScoreToEval(float score)
{
    ASSERT(score >= 0.0f && score <= 1.0f);
    score = std::clamp(score, 0.0f, 1.0f);
    return 4.0f * log10f(score / (1.0f - score));
}

// convert expected game score to evaluation score
inline ScoreType ExpectedGameScoreToInternalEval(float score)
{
    if (score > 0.99999f)
        return KnownWinValue - 1;
    else if (score < 0.00001f)
        return -KnownWinValue + 1;
    else
        return (ScoreType)std::clamp((int32_t)std::round(100.0f * ExpectedGameScoreToEval(score)),
            -KnownWinValue + 1, KnownWinValue - 1);
}

ScoreType Evaluate(const Position& position);
ScoreType Evaluate(NodeInfo& node, AccumulatorCache& cache);

void EnsureAccumulatorUpdated(NodeInfo& node, AccumulatorCache& cache);

bool CheckInsufficientMaterial(const Position& position);
