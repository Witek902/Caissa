#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"

#include "net/SparseBinaryInputNode.hpp"
#include "net/FullyConnectedNode.hpp"
#include "net/ConcatenationNode.hpp"
#include "net/SumNode.hpp"
#include "net/ActivationNode.hpp"
#include "net/WeightsStorage.hpp"

#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Material.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/Waitable.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>
#include <cmath>

#define USE_PACKED_NET
// #define USE_VIRTUAL_FEATURES

using namespace threadpool;

static const uint32_t cMaxIterations = 1'000'000'000;
static const uint32_t cNumTrainingVectorsPerIteration = 512 * 1024;
static const uint32_t cNumValidationVectorsPerIteration = 128 * 1024;
static const uint32_t cMinBatchSize = 32 * 1024;
static const uint32_t cMaxBatchSize = 32 * 1024;
#ifdef USE_VIRTUAL_FEATURES
static const uint32_t cNumVirtualFeatures = 12 * 64;
#endif // USE_VIRTUAL_FEATURES

class NetworkTrainer
{
public:

    NetworkTrainer()
        : m_randomGenerator(m_randomDevice())
        , m_trainingLog("training.log")
    {
        m_validationSet.resize(cNumTrainingVectorsPerIteration);
        m_trainingSet.resize(cNumTrainingVectorsPerIteration);
        m_trainingSetCopy.resize(cNumTrainingVectorsPerIteration);
        m_validationPerThreadData.resize(ThreadPool::GetInstance().GetNumThreads());
    }

    void InitNetwork();

    bool Train();

private:

    struct ValidationStats
    {
        float nnMinError = std::numeric_limits<float>::max();
        float nnMaxError = 0.0f, nnErrorSum = 0.0f;

#ifdef USE_PACKED_NET
        float nnPackedQuantizationErrorSum = 0.0f;
        float nnPackedMinError = std::numeric_limits<float>::max();
        float nnPackedMaxError = 0.0f, nnPackedErrorSum = 0.0f;
#endif // USE_PACKED_NET

        float evalMinError = std::numeric_limits<float>::max();
        float evalMaxError = 0.0f, evalErrorSum = 0.0f;
    };

    struct alignas(CACHELINE_SIZE) ValidationPerThreadData
    {
        ValidationStats stats;
        nn::NeuralNetworkRunContext networkRunContext;
        uint8_t __padding[CACHELINE_SIZE];
    };

    TrainingDataLoader m_dataLoader;

    nn::WeightsStoragePtr m_featureTransformerWeights;
    nn::WeightsStoragePtr m_lastLayerWeights;

    nn::NeuralNetwork m_network;
    nn::NeuralNetworkRunContext m_runCtx;
    nn::NeuralNetworkTrainer m_trainer;
#ifdef USE_PACKED_NET
    nn::PackedNeuralNetwork m_packedNet;
#endif // USE_PACKED_NET

    std::vector<TrainingEntry> m_validationSet;
    std::vector<TrainingEntry> m_trainingSet;
    std::vector<TrainingEntry> m_trainingSetCopy; // copy, because training set generation runs in parallel with training
    std::vector<ValidationPerThreadData> m_validationPerThreadData;

    alignas(CACHELINE_SIZE)
        std::atomic<uint64_t> m_numTrainingVectorsPassed = 0;

    alignas(CACHELINE_SIZE)
        std::mutex m_mutex;

    std::random_device m_randomDevice;
    std::mt19937 m_randomGenerator;

    std::ofstream m_trainingLog;

    bool GenerateTrainingSet(std::vector<TrainingEntry>& outEntries, uint64_t kingBucketMask, float baseLambda);

    void Validate(size_t iteration);

    void BlendLastLayerWeights();

    bool PackNetwork();
    bool UnpackNetwork();
};

void NetworkTrainer::InitNetwork()
{
    const uint32_t accumulatorSize = nn::AccumulatorSize;
    const uint32_t networkInputs = nn::NumNetworkInputs
#ifdef USE_VIRTUAL_FEATURES
        + cNumVirtualFeatures
#endif // USE_VIRTUAL_FEATURES
        ;

    m_featureTransformerWeights = std::make_shared<nn::WeightsStorage>(networkInputs, accumulatorSize, 1);
    m_featureTransformerWeights->m_isSparse = true;
    // divide by number of active input features to avoid accumulator overflow
    m_featureTransformerWeights->m_weightsRange = (float)std::numeric_limits<nn::FirstLayerWeightType>::max() / 32 / nn::InputLayerWeightQuantizationScale;
    m_featureTransformerWeights->m_biasRange = (float)std::numeric_limits<nn::FirstLayerBiasType>::max() / 32 / nn::InputLayerBiasQuantizationScale;
    m_featureTransformerWeights->Init(32u, 0.0f);

    //nn::WeightsStoragePtr layer1Weights = std::make_shared<nn::WeightsStorage>(2u * accumulatorSize, 1);
    //layer1Weights->m_weightsRange = (float)std::numeric_limits<nn::HiddenLayerWeightType>::max() / nn::HiddenLayerWeightQuantizationScale;
    //layer1Weights->m_biasRange = (float)std::numeric_limits<nn::HiddenLayerWeightType>::max() / nn::HiddenLayerBiasQuantizationScale;

    m_lastLayerWeights = std::make_shared<nn::WeightsStorage>(2u * accumulatorSize, 1, nn::NumVariants);
    m_lastLayerWeights->m_weightsRange = (float)std::numeric_limits<nn::LastLayerWeightType>::max() / nn::OutputLayerWeightQuantizationScale;
    m_lastLayerWeights->m_biasRange = (float)std::numeric_limits<nn::LastLayerBiasType>::max() / nn::OutputLayerBiasQuantizationScale;
    m_lastLayerWeights->Init(2 * nn::AccumulatorSize);

    nn::NodePtr inputNodeA = std::make_shared<nn::SparseBinaryInputNode>(networkInputs, accumulatorSize, m_featureTransformerWeights);
    nn::NodePtr inputNodeB = std::make_shared<nn::SparseBinaryInputNode>(networkInputs, accumulatorSize, m_featureTransformerWeights);
    nn::NodePtr concatenationNode = std::make_shared<nn::ConcatenationNode>(inputNodeA, inputNodeB);
    nn::NodePtr activationNode = std::make_shared<nn::ActivationNode>(concatenationNode, nn::ActivationFunction::CReLU);
    nn::NodePtr hiddenNode = std::make_shared<nn::FullyConnectedNode>(activationNode, 2u * accumulatorSize, 1, m_lastLayerWeights);
    nn::NodePtr outputNode = std::make_shared<nn::ActivationNode>(hiddenNode, nn::ActivationFunction::Sigmoid);

    std::vector<nn::NodePtr> nodes =
    {
        inputNodeA,
        inputNodeB,

        concatenationNode,
        activationNode,
        hiddenNode,

        outputNode,
    };

    m_network.Init(nodes);
    m_trainer.Init(m_network);
    m_runCtx.Init(m_network);

    for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
    {
        m_validationPerThreadData[i].networkRunContext.Init(m_network);
    }
}

static void PositionToTrainingEntry(const Position& pos, TrainingEntry& outEntry)
{
    ASSERT(pos.GetSideToMove() == Color::White);

    constexpr uint32_t maxFeatures = 64;
#ifdef USE_VIRTUAL_FEATURES
    constexpr bool useVirtualFeatures = true;
#else
    constexpr bool useVirtualFeatures = false;
#endif // USE_VIRTUAL_FEATURES

    uint16_t whiteFeatures[maxFeatures];
    uint32_t numWhiteFeatures = PositionToFeaturesVector<useVirtualFeatures>(pos, whiteFeatures, pos.GetSideToMove());
    ASSERT(numWhiteFeatures <= maxFeatures);

    uint16_t blackFeatures[maxFeatures];
    uint32_t numBlackFeatures = PositionToFeaturesVector<useVirtualFeatures>(pos, blackFeatures, GetOppositeColor(pos.GetSideToMove()));
    ASSERT(numBlackFeatures == numWhiteFeatures);

    outEntry.whiteFeatures.clear();
    outEntry.whiteFeatures.reserve(numWhiteFeatures);
    for (uint32_t i = 0; i < numWhiteFeatures; ++i)
        outEntry.whiteFeatures.emplace_back(whiteFeatures[i]);

    outEntry.blackFeatures.clear();
    outEntry.blackFeatures.reserve(numBlackFeatures);
    for (uint32_t i = 0; i < numBlackFeatures; ++i)
        outEntry.blackFeatures.emplace_back(blackFeatures[i]);

    outEntry.networkVariant = GetNetworkVariant(pos);
}

static void TrainingEntryToNetworkInput(const TrainingEntry& entry, nn::InputDesc& inputDesc)
{
    inputDesc.variant = entry.networkVariant;

    inputDesc.inputs[0].mode = nn::InputMode::SparseBinary;
    inputDesc.inputs[0].binaryFeatures = entry.whiteFeatures.data();
    inputDesc.inputs[0].numFeatures = static_cast<uint32_t>(entry.whiteFeatures.size());

    inputDesc.inputs[1].mode = nn::InputMode::SparseBinary;
    inputDesc.inputs[1].binaryFeatures = entry.blackFeatures.data();
    inputDesc.inputs[1].numFeatures = static_cast<uint32_t>(entry.blackFeatures.size());
}

bool NetworkTrainer::GenerateTrainingSet(std::vector<TrainingEntry>& outEntries, uint64_t kingBucketMask, float baseLambda)
{
    Position pos;
    PositionEntry entry;

    for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
    {
        if (!m_dataLoader.FetchNextPosition(m_randomGenerator, entry, pos, kingBucketMask))
            return false;

        // flip the board randomly in pawnless positions
        if (pos.Whites().pawns == 0 && pos.Blacks().pawns == 0)
        {
            if (std::uniform_int_distribution<>(0, 1)(m_randomGenerator) != 0)
                pos.MirrorVertically();
            if (std::uniform_int_distribution<>(0, 1)(m_randomGenerator) != 0)
                pos.FlipDiagonally();
        }

        // make game score more important for high move count
        const float wdlLambda = baseLambda * expf(-(float)pos.GetMoveCount() / 120.0f);

        const Game::Score gameScore = (Game::Score)entry.wdlScore;
        const Game::Score tbScore = (Game::Score)entry.tbScore;
        float score = InternalEvalToExpectedGameScore(entry.score);

        if (gameScore != Game::Score::Unknown)
        {
            const float wdlScore = gameScore == Game::Score::WhiteWins ? 1.0f : (gameScore == Game::Score::BlackWins ? 0.0f : 0.5f);
            score = std::lerp(wdlScore, score, wdlLambda);
        }

        if (tbScore == Game::Score::Draw)
        {
            const float tbDrawLambda = 0.05f;
            score = std::lerp(0.5f, score, tbDrawLambda);
        }
        else if (tbScore != Game::Score::Unknown)
        {
            const float tbLambda = 0.25f;
            const float wdlScore = tbScore == Game::Score::WhiteWins ? 1.0f : (tbScore == Game::Score::BlackWins ? 0.0f : 0.5f);
            score = std::lerp(wdlScore, score, tbLambda);
        }

        PositionToTrainingEntry(pos, outEntries[i]);
        outEntries[i].output = score;
        outEntries[i].pos = pos;
    }

    return true;
}

static void ParallelFor(const char* debugName, uint32_t arraySize, const threadpool::ParallelForTaskFunction& func, uint32_t maxThreads = 0)
{
    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);
        taskBuilder.ParallelFor(debugName, arraySize, func, maxThreads);
    }
    waitable.Wait();
}

#ifdef USE_PACKED_NET
static float EvalPackedNetwork(const TrainingEntry& entry, const nn::PackedNeuralNetwork& net)
{
    uint32_t numWhiteFeatures = 0;
    uint32_t numBlackFeatures = 0;

    // skip virtual features
    for (size_t i = 0; i < entry.whiteFeatures.size(); ++i)
    {
        if (entry.whiteFeatures[i] >= nn::NumNetworkInputs) break;
        ++numWhiteFeatures;
    }
    for (size_t i = 0; i < entry.blackFeatures.size(); ++i)
    {
        if (entry.blackFeatures[i] >= nn::NumNetworkInputs) break;
        ++numBlackFeatures;
    }

    const int32_t packedNetworkOutput = net.Run(
        entry.whiteFeatures.data(), numWhiteFeatures,
        entry.blackFeatures.data(), numBlackFeatures,
        entry.networkVariant);
    const float scaledPackedNetworkOutput = (float)packedNetworkOutput / (float)nn::OutputScale * c_nnOutputToCentiPawns / 100.0f;
    return EvalToExpectedGameScore(scaledPackedNetworkOutput);
}
#endif // USE_PACKED_NET

void NetworkTrainer::Validate(size_t iteration)
{
    // reset stats
    for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
    {
        m_validationPerThreadData[i].stats = ValidationStats();
    }

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);
        taskBuilder.ParallelFor("Validate", cNumValidationVectorsPerIteration, [this](const TaskContext& ctx, uint32_t i)
        {
            ValidationPerThreadData& threadData = m_validationPerThreadData[ctx.threadId];

            const TrainingEntry& entry = m_validationSet[i];

            const float expectedValue = entry.output;

            const ScoreType evalValue = Evaluate(entry.pos);

#ifdef USE_PACKED_NET
            const float nnPackedValue = EvalPackedNetwork(entry, m_packedNet);
#endif // USE_PACKED_NET

            nn::InputDesc inputDesc;
            TrainingEntryToNetworkInput(entry, inputDesc);

            const nn::Values& networkOutput = m_network.Run(inputDesc, threadData.networkRunContext);
            const float nnValue = networkOutput[0];

            if (i + 1 == cNumValidationVectorsPerIteration)
            {
                std::cout
                    << entry.pos.ToFEN() << std::endl << entry.pos.Print() << std::endl
                    << "True Score:     " << expectedValue << " (" << ExpectedGameScoreToInternalEval(expectedValue) << ")" << std::endl
                    << "NN eval:        " << nnValue << " (" << ExpectedGameScoreToInternalEval(nnValue) << ")" << std::endl
#ifdef USE_PACKED_NET
                    << "Packed NN eval: " << nnPackedValue << " (" << ExpectedGameScoreToInternalEval(nnPackedValue) << ")" << std::endl
#endif // USE_PACKED_NET
                    << "Static eval:    " << InternalEvalToExpectedGameScore(evalValue) << " (" << evalValue << ")" << std::endl
                    << std::endl;
            }

            ValidationStats& stats = threadData.stats;
            {
                const float error = expectedValue - nnValue;
                const float errorDiff = std::abs(error);
                stats.nnErrorSum += error * error;
                stats.nnMinError = std::min(stats.nnMinError, errorDiff);
                stats.nnMaxError = std::max(stats.nnMaxError, errorDiff);
            }
            {
                const float error = expectedValue - InternalEvalToExpectedGameScore(evalValue);
                const float errorDiff = std::abs(error);
                stats.evalErrorSum += error * error;
                stats.evalMinError = std::min(stats.evalMinError, errorDiff);
                stats.evalMaxError = std::max(stats.evalMaxError, errorDiff);
            }
#ifdef USE_PACKED_NET
            stats.nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            {
                const float error = expectedValue - nnPackedValue;
                const float errorDiff = std::abs(error);
                stats.nnPackedErrorSum += error * error;
                stats.nnPackedMinError = std::min(stats.nnPackedMinError, errorDiff);
                stats.nnPackedMaxError = std::max(stats.nnPackedMaxError, errorDiff);
            }
#endif // USE_PACKED_NET

        });
    }

    waitable.Wait();

    // accumulate stats
    ValidationStats stats;
    for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
    {
        const ValidationStats& threadStats = m_validationPerThreadData[i].stats;

        stats.nnErrorSum += threadStats.nnErrorSum;
        stats.nnMinError = std::min(stats.nnMinError, threadStats.nnMinError);
        stats.nnMaxError = std::max(stats.nnMaxError, threadStats.nnMaxError);
#ifdef USE_PACKED_NET
        stats.nnPackedQuantizationErrorSum += threadStats.nnPackedQuantizationErrorSum;
        stats.nnPackedErrorSum += threadStats.nnPackedErrorSum;
        stats.nnPackedMinError = std::min(stats.nnPackedMinError, threadStats.nnPackedMinError);
        stats.nnPackedMaxError = std::max(stats.nnPackedMaxError, threadStats.nnPackedMaxError);
#endif // USE_PACKED_NET
        stats.evalErrorSum += threadStats.evalErrorSum;
        stats.evalMinError = std::min(stats.evalMinError, threadStats.evalMinError);
        stats.evalMaxError = std::max(stats.evalMaxError, threadStats.evalMaxError);
    }

    stats.nnErrorSum = sqrtf(stats.nnErrorSum / cNumValidationVectorsPerIteration);
    stats.evalErrorSum = sqrtf(stats.evalErrorSum / cNumValidationVectorsPerIteration);
#ifdef USE_PACKED_NET
    stats.nnPackedErrorSum = sqrtf(stats.nnPackedErrorSum / cNumValidationVectorsPerIteration);
    stats.nnPackedQuantizationErrorSum = sqrtf(stats.nnPackedQuantizationErrorSum / cNumValidationVectorsPerIteration);
#endif // USE_PACKED_NET


    std::cout
        << "NN avg/min/max error:   " << std::setprecision(5) << stats.nnErrorSum << " " << std::setprecision(4) << stats.nnMinError << " " << std::setprecision(4) << stats.nnMaxError << std::endl
#ifdef USE_PACKED_NET
        << "PNN avg/min/max error:  " << std::setprecision(5) << stats.nnPackedErrorSum << " " << std::setprecision(4) << stats.nnPackedMinError << " " << std::setprecision(4) << stats.nnPackedMaxError << std::endl
        << "Quantization error:     " << std::setprecision(5) << stats.nnPackedQuantizationErrorSum << std::endl
#endif // USE_PACKED_NET
        << "Eval avg/min/max error: " << std::setprecision(5) << stats.evalErrorSum << " " << std::setprecision(4) << stats.evalMinError << " " << std::setprecision(4) << stats.evalMaxError << std::endl;

    {
        const char* s_testPositions[] =
        {
            Position::InitPositionFEN,
            "rnbq1bnr/pppppppp/8/8/5k2/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",         // black king in the center
            "r1bq1rk1/1pp2ppp/8/4pn2/B6b/1PN2P2/PBPP1P2/RQ2R1K1 w - - 1 12",
            "k7/ppp5/8/8/8/8/P7/K7 w - - 0 1",  // should be at least -200
            "7k/ppp5/8/8/8/8/P7/7K w - - 0 1",  // should be at least -200
            "7k/pp6/8/8/8/8/PP6/7K w - - 0 1",   // should be 0
            "k7/pp6/8/8/8/8/P7/K7 w - - 0 1",   // should be 0
            "r6k/7p/8/8/8/8/7P/1R5K w - - 0 1", // should be 0
            "8/7p/8/6k1/3q3p/4R3/5PK1/8 w - - 0 1", // should be 0
            "8/1k6/1p6/1R6/2P5/1P6/1K6/4q3 w - - 0 1", // should be 0
            "8/8/5k2/6p1/8/1P2R3/2q2P2/6K1 w - - 0 1", // should be 0
            "4k3/5p2/2K1p3/1Q1rP3/8/8/8/8 w - - 0 1", // should be 0
            "8/8/8/5B1p/5p1r/4kP2/6K1/8 w - - 0 1", // should be 0
            "8/8/8/p7/K5R1/1n6/1k1r4/8 w - - 0 1", // should be 0
            "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNk w Q - 0 1",
        };

        for (const char* testPosition : s_testPositions)
        {
            Position pos(testPosition);

            TrainingEntry entry;
            PositionToTrainingEntry(pos, entry);

            nn::InputDesc inputDesc;
            TrainingEntryToNetworkInput(entry, inputDesc);

            const float nnValue = m_network.Run(inputDesc, m_runCtx)[0];

#ifdef USE_PACKED_NET
            const float scaledPackedNetworkOutput = EvalPackedNetwork(entry, m_packedNet);
#endif // USE_PACKED_NET

            std::cout
                << "TEST " << testPosition
                << "  nn=" << ExpectedGameScoreToInternalEval(nnValue)
#ifdef USE_PACKED_NET
                << "  pnn=" << ExpectedGameScoreToInternalEval(scaledPackedNetworkOutput)
#endif // USE_PACKED_NET
                << std::endl;
        }
    }

    m_trainingLog
        << iteration << "\t"
        << stats.nnErrorSum
#ifdef USE_PACKED_NET
        << "\t" << stats.nnPackedErrorSum
#endif // USE_PACKED_NET
        << std::endl;

    m_network.PrintStats();
}

void NetworkTrainer::BlendLastLayerWeights()
{
    const float blendFactor = 1.0e-5f;

    for (uint32_t i = 0; i < nn::AccumulatorSize * 2; ++i)
    {
        // load weights
        float weights[nn::NumVariants];
        for (uint32_t variant = 0; variant < nn::NumVariants; ++variant)
        {
            weights[variant] = m_lastLayerWeights->m_variants[variant].m_weights[i];
        }

        // blend weights with neighboring buckets
        float blendedWeights[nn::NumVariants];
        for (uint32_t materialGroup = 0; materialGroup < 2; ++materialGroup)
        {
            const uint32_t offset = nn::NumPieceCountBuckets * materialGroup;

            blendedWeights[offset] = (weights[offset] + weights[offset + 1] * blendFactor) / (1.0f + blendFactor);
            blendedWeights[offset + nn::NumPieceCountBuckets - 1] = (weights[offset + nn::NumPieceCountBuckets - 1] + weights[offset + nn::NumPieceCountBuckets - 2] * blendFactor) / (1.0f + blendFactor);

            for (uint32_t j = 1; j + 1 < nn::NumPieceCountBuckets; ++j)
            {
                blendedWeights[offset + j] = (weights[offset + j] + (weights[offset + j + 1] + weights[offset + j - 1]) * blendFactor) / (1.0f + 2.0f * blendFactor);
            }
        }

        // store blended weights
        for (uint32_t variant = 0; variant < nn::NumVariants; ++variant)
        {
            m_lastLayerWeights->m_variants[variant].m_weights[i] = blendedWeights[variant];
        }
    }
}

template<typename WeightType, typename BiasType>
static void PackWeights(const nn::Values& weights, uint32_t numInputs, uint32_t numOutputs, WeightType* outWeights, BiasType* outBiases, float weightScale, float biasScale, bool transpose)
{
    // weights
    for (uint32_t j = 0; j < numInputs; j++)
    {
        for (uint32_t i = 0; i < numOutputs; i++)
        {
            const float weight = weights[j * numOutputs + i];
            const int32_t quantizedWeight = (int32_t)std::round(weight * weightScale);
            ASSERT(quantizedWeight <= std::numeric_limits<WeightType>::max());
            ASSERT(quantizedWeight >= std::numeric_limits<WeightType>::min());

            if (transpose)
                outWeights[numOutputs * j + i] = (WeightType)quantizedWeight;
            else
                outWeights[numInputs * i + j] = (WeightType)quantizedWeight;
        }
    }

    // biases
    for (uint32_t i = 0; i < numOutputs; i++)
    {
        const float bias = weights[numInputs * numOutputs + i];
        const int32_t quantizedBias = (int32_t)std::round(bias * biasScale);
        ASSERT(quantizedBias <= std::numeric_limits<BiasType>::max());
        ASSERT(quantizedBias >= std::numeric_limits<BiasType>::min());
        outBiases[i] = (BiasType)quantizedBias;
    }
}

template<typename WeightType, typename BiasType>
static void UnpackWeights(nn::Values& outWeights, uint32_t numInputs, uint32_t numOutputs, const WeightType* weights, const BiasType* biases, float weightScale, float biasScale, bool transpose)
{
    // weights
    for (uint32_t j = 0; j < numInputs; j++)
    {
        for (uint32_t i = 0; i < numOutputs; i++)
        {
            outWeights[j * numOutputs + i] = transpose ?
                (float)weights[numOutputs * j + i] / weightScale :
                (float)weights[numInputs * i + j] / weightScale;
        }
    }

    // biases
    for (uint32_t i = 0; i < numOutputs; i++)
    {
        outWeights[numInputs * numOutputs + i] = (float)biases[i] / biasScale;
    }
}

bool NetworkTrainer::PackNetwork()
{
    {
        const std::vector<uint32_t> layerSizes = { nn::NumNetworkInputs, 2u * nn::AccumulatorSize };
        const std::vector<uint32_t> layerVariants = { 1, nn::NumVariants };

        if (!m_packedNet.Resize(layerSizes, layerVariants))
        {
            return false;
        }
    }

    // feature transformer
    {
#ifdef USE_VIRTUAL_FEATURES
        nn::Values weights((nn::NumNetworkInputs + 1u) * nn::AccumulatorSize, 0.0f);

        const nn::Values& originalWeights = m_featureTransformerWeights->m_variants.front().m_weights;

        // distribute weights of virtual features to all king buckets
        for (uint32_t kingBucket = 0; kingBucket < nn::NumKingBuckets; ++kingBucket)
        {
            for (uint32_t featureIndex = 0; featureIndex < 12 * 64; ++featureIndex)
            {
                for (uint32_t accumIndex = 0; accumIndex < nn::AccumulatorSize; ++accumIndex)
                {
                    weights[accumIndex + (kingBucket * 12 * 64 + featureIndex) * nn::AccumulatorSize] =
                        originalWeights[accumIndex + (kingBucket * 12 * 64 + featureIndex) * nn::AccumulatorSize] +
                        originalWeights[accumIndex + (nn::NumNetworkInputs + featureIndex) * nn::AccumulatorSize];
                }
            }
        }

        // copy biases
        for (uint32_t accumIndex = 0; accumIndex < nn::AccumulatorSize; ++accumIndex)
        {
            weights[accumIndex + nn::NumNetworkInputs * nn::AccumulatorSize] =
                originalWeights[accumIndex + (nn::NumNetworkInputs + 12 * 64) * nn::AccumulatorSize];
        }
#else // !USE_VIRTUAL_FEATURES
        const nn::Values weights = m_featureTransformerWeights->m_variants.front().m_weights;
#endif // USE_VIRTUAL_FEATURES

        PackWeights(
            weights,
            nn::NumNetworkInputs,
            nn::AccumulatorSize,
            const_cast<nn::FirstLayerWeightType*>(m_packedNet.GetAccumulatorWeights()),
            const_cast<nn::FirstLayerBiasType*>(m_packedNet.GetAccumulatorBiases()),
            nn::InputLayerWeightQuantizationScale,
            nn::InputLayerBiasQuantizationScale,
            true);
    }

    /*
    // hidden layers
    for (uint32_t i = 1; i + 1 < nodes.size(); ++i)
    {
        for (uint32_t variantIdx = 0; variantIdx < nodes[i].variants.size(); ++variantIdx)
        {
            PackWeights(nodes[i],
                variantIdx,
                const_cast<nn::HiddenLayerWeightType*>(outNetwork.GetLayerWeights<HiddenLayerWeightType>(uint32_t(i), variantIdx)),
                const_cast<nn::HiddenLayerBiasType*>(outNetwork.GetLayerBiases<HiddenLayerBiasType>(uint32_t(i), variantIdx)),
                nn::HiddenLayerWeightQuantizationScale,
                nn::HiddenLayerBiasQuantizationScale,
                false);
        }
    }
    */

    // last layer
    const uint32_t lastLayerIndex = 1;
    for (uint32_t variantIdx = 0; variantIdx < nn::NumVariants; ++variantIdx)
    {
        PackWeights(
            m_lastLayerWeights->m_variants[variantIdx].m_weights,
            m_lastLayerWeights->m_inputSize,
            1u,
            const_cast<nn::LastLayerWeightType*>(m_packedNet.GetLayerWeights<nn::LastLayerWeightType>(lastLayerIndex, variantIdx)),
            const_cast<nn::LastLayerBiasType*>(m_packedNet.GetLayerBiases<nn::LastLayerBiasType>(lastLayerIndex, variantIdx)),
            nn::OutputLayerWeightQuantizationScale,
            nn::OutputLayerBiasQuantizationScale,
            false);
    }

    return true;
}

bool NetworkTrainer::UnpackNetwork()
{
    // feature transformer
    {
        UnpackWeights(
            m_featureTransformerWeights->m_variants.front().m_weights,
            nn::NumNetworkInputs,
            nn::AccumulatorSize,
            m_packedNet.GetAccumulatorWeights(),
            m_packedNet.GetAccumulatorBiases(),
            nn::InputLayerWeightQuantizationScale,
            nn::InputLayerBiasQuantizationScale,
            true);
    }

    /*
    // hidden layers
    for (uint32_t i = 1; i + 1 < nodes.size(); ++i)
    {
        for (uint32_t variantIdx = 0; variantIdx < nodes[i].variants.size(); ++variantIdx)
        {
            UnpackWeights(nodes[i],
                variantIdx,
                const_cast<nn::HiddenLayerWeightType*>(outNetwork.GetLayerWeights<HiddenLayerWeightType>(uint32_t(i), variantIdx)),
                const_cast<nn::HiddenLayerBiasType*>(outNetwork.GetLayerBiases<HiddenLayerBiasType>(uint32_t(i), variantIdx)),
                nn::HiddenLayerWeightQuantizationScale,
                nn::HiddenLayerBiasQuantizationScale,
                false);
        }
    }
    */

    // last layer
    const uint32_t lastLayerIndex = 1;
    for (uint32_t variantIdx = 0; variantIdx < nn::NumVariants; ++variantIdx)
    {
        UnpackWeights(
            m_lastLayerWeights->m_variants[variantIdx].m_weights,
            m_lastLayerWeights->m_inputSize,
            1u,
            m_packedNet.GetLayerWeights<nn::LastLayerWeightType>(lastLayerIndex, variantIdx),
            m_packedNet.GetLayerBiases<nn::LastLayerBiasType>(lastLayerIndex, variantIdx),
            nn::OutputLayerWeightQuantizationScale,
            nn::OutputLayerBiasQuantizationScale,
            false);
    }

    return true;
}

bool NetworkTrainer::Train()
{
    InitNetwork();

    if (!m_packedNet.Load("eval-21-1.pnn"))
    {
        std::cout << "ERROR: Failed to load packed network" << std::endl;
        return false;
    }
    UnpackNetwork();

    if (!m_dataLoader.Init(m_randomGenerator))
    {
        std::cout << "ERROR: Failed to initialize data loader" << std::endl;
        return false;
    }

    std::ofstream weightsFile("weights.txt", std::ios::out);

    std::vector<nn::TrainingVector> batch(cNumTrainingVectorsPerIteration);

    TimePoint prevIterationStartTime = TimePoint::GetCurrent();

    const float maxLearningRate = 0.25f;
    const float minLearningRate = 0.1f;
    const float maxLambda = 0.2f;
    const float minLambda = 0.1f;

    //uint64_t kingBucketMask = (1 << 4) | (1 << 3) | (1 << 2);
    uint64_t kingBucketMask = UINT64_MAX;

    GenerateTrainingSet(m_validationSet, kingBucketMask, maxLambda);

    size_t epoch = 0;
    for (size_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        const float warmup = iteration < 10.0f ? (float)(iteration + 1) / 10.0f : 1.0f;
        const float learningRate = warmup * std::lerp(minLearningRate, maxLearningRate, expf(-0.0005f * (float)iteration));
        const float lambda = std::lerp(minLambda, maxLambda, expf(-0.0005f * (float)iteration));

        if (iteration == 0)
        {
            if (!GenerateTrainingSet(m_trainingSet, kingBucketMask, lambda))
                return false;
        }

        TimePoint iterationStartTime = TimePoint::GetCurrent();
        float iterationTime = (iterationStartTime - prevIterationStartTime).ToSeconds();
        prevIterationStartTime = iterationStartTime;

        BlendLastLayerWeights();

        // use validation set from previous iteration as training set in the current one
        ParallelFor("PrepareBatch", cNumTrainingVectorsPerIteration, [&batch, this](const TaskContext&, uint32_t i)
        {
            m_trainingSetCopy[i] = m_trainingSet[i];

            const TrainingEntry& entry = m_trainingSetCopy[i];

            nn::TrainingVector& trainingVector = batch[i];
            trainingVector.input.variant = entry.networkVariant;
            trainingVector.output.mode = nn::OutputMode::Single;
            trainingVector.output.singleValue = entry.output;

            TrainingEntryToNetworkInput(entry, trainingVector.input);
        });

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder{ waitable };
            taskBuilder.Task("GenerateSet", [&](const TaskContext&)
            {
                GenerateTrainingSet(m_trainingSet, kingBucketMask, lambda);
            });

            taskBuilder.Task("Train", [this, iteration, kingBucketMask, &epoch, &batch, learningRate](const TaskContext& ctx)
            {
                nn::TrainParams params;
                params.optimizer = nn::Optimizer::Adadelta;
                params.iteration = epoch;
                params.batchSize = iteration < 5 ? cMinBatchSize : cMaxBatchSize;
                params.learningRate = learningRate;
                params.weightDecay = 1.0e-5f;

                if (kingBucketMask != UINT64_MAX)
                {
                    // set weights mask based on king bucket
                    m_lastLayerWeights->m_updateWeights = false;
                    std::fill(m_featureTransformerWeights->m_weightsMask.begin(), m_featureTransformerWeights->m_weightsMask.end(), 0.0f);
                    {
                        uint64_t mask = kingBucketMask;
                        while (mask)
                        {
                            const uint32_t index = FirstBitSet(mask);
                            mask &= ~(1ull << index);

                            std::fill(
                                m_featureTransformerWeights->m_weightsMask.begin() + 12 * 64 * nn::AccumulatorSize * index,
                                m_featureTransformerWeights->m_weightsMask.begin() + 12 * 64 * nn::AccumulatorSize * (index + 1),
                                1.0f);
                        };
                    }
                }

                TaskBuilder taskBuilder{ ctx };
                epoch += m_trainer.Train(m_network, batch, params, &taskBuilder);
            });
        }
        waitable.Wait();

#ifdef USE_PACKED_NET
        PackNetwork();
        ASSERT(m_packedNet.IsValid());
#endif // USE_PACKED_NET

        m_numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        std::cout
            << "Lambda:                 " << lambda << std::endl
            << "Iteration:              " << iteration << std::endl
            << "Epoch:                  " << epoch << std::endl
            << "Num training vectors:   " << std::setprecision(3) << m_numTrainingVectorsPassed / 1.0e9f << "B" << std::endl
            << "Learning rate:          " << learningRate << std::endl;

        Validate(iteration);

        std::cout << "Iteration time:   " << 1000.0f * iterationTime << " ms" << std::endl;
        std::cout << "Training rate :   " << ((float)cNumTrainingVectorsPerIteration / iterationTime) << " pos/sec" << std::endl << std::endl;

        if (iteration % 10 == 0)
        {
            const std::string name = "eval";
            m_network.Save((name + ".nn").c_str());
#ifdef USE_PACKED_NET
            m_packedNet.Save((name + ".pnn").c_str());
#endif // USE_PACKED_NET
        }
    }

    return true;
}


bool TrainNetwork()
{
    NetworkTrainer trainer;
    return trainer.Train();
}
