#pragma once

#include "Position.hpp"
#include "Move.hpp"
#include "Score.hpp"

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
static constexpr PieceScore c_kingValue     = { INT16_MAX, INT16_MAX };

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
// equal to 400/ln(10) = 173.7177...
static constexpr int32_t c_nnOutputToCentiPawns = 174;

namespace wld
{
    // WLD model by Vondele
    // coefficients computed with https://github.com/vondele/WLD_model on 60+0.6s games
    constexpr double as[] = {   1.88054041,    2.39467539,   -3.78019886,  153.49644002 };
    constexpr double bs[] = {  -4.24993229,   31.21455804,  -59.09168702,   68.89719592 };
    constexpr int32_t NormalizeToPawnValue = 153;
}

int32_t NormalizeEval(int32_t eval);

// convert evaluation score (in pawns) to win probability
float EvalToWinProbability(float eval, uint32_t ply);

// convert evaluation score (in pawns) to draw probability
float EvalToDrawProbability(float eval, uint32_t ply);

// convert evaluation score (in pawns) to expected game score
float EvalToExpectedGameScore(float eval);

// convert evaluation score (in centipawns) to expected game score
float InternalEvalToExpectedGameScore(int32_t eval);

// convert expected game score to evaluation score (in pawns)
float ExpectedGameScoreToEval(float score);

// convert expected game score to evaluation score
ScoreType ExpectedGameScoreToInternalEval(float score);

ScoreType Evaluate(const Position& position);
ScoreType Evaluate(NodeInfo& node, AccumulatorCache& cache);

void EnsureAccumulatorUpdated(NodeInfo& node, AccumulatorCache& cache);

bool CheckInsufficientMaterial(const Position& position);
