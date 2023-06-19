#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"

#include "net/SparseBinaryInputNode.hpp"
#include "net/FullyConnectedNode.hpp"
#include "net/ConcatenationNode.hpp"
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

using namespace threadpool;

static const uint32_t cMaxIterations = 1000000000;
static const uint32_t cNumTrainingVectorsPerIteration = 128 * 1024;
static const uint32_t cNumValidationVectorsPerIteration = 64 * 1024;
static const uint32_t cMinBatchSize = 4096;
static const uint32_t cMaxBatchSize = 4096;
static const uint32_t cNumNetworkInputs = 32 + 9 * 64 + 2 * 48; // 704
static const uint32_t cNumVariants = 16;


class NetworkTrainer
{
public:

    NetworkTrainer()
        : m_randomGenerator(m_randomDevice())
        , m_trainingLog("training.log")
    {
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

    nn::NeuralNetwork m_network;
    nn::NeuralNetworkRunContext m_runCtx;
    nn::NeuralNetworkTrainer m_trainer;
#ifdef USE_PACKED_NET
    nn::PackedNeuralNetwork m_packedNet;
#endif // USE_PACKED_NET

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

    bool GenerateTrainingSet(std::vector<TrainingEntry>& outEntries);

    void Validate(size_t iteration);
};

void NetworkTrainer::InitNetwork()
{
    const uint32_t hiddenLayerSize = 768;

    nn::WeightsStoragePtr featureTransformerWeights = std::make_shared<nn::WeightsStorage>(cNumNetworkInputs, hiddenLayerSize);
    featureTransformerWeights->m_isSparse = true;
    // divide by number of active input features to avoid accumulator overflow
    featureTransformerWeights->m_weightsRange = (float)std::numeric_limits<nn::FirstLayerWeightType>::max() / 64 / nn::InputLayerWeightQuantizationScale;
    featureTransformerWeights->m_biasRange = (float)std::numeric_limits<nn::FirstLayerBiasType>::max() / 64 / nn::InputLayerBiasQuantizationScale;

    //nn::WeightsStoragePtr layer1Weights = std::make_shared<nn::WeightsStorage>(2u * hiddenLayerSize, 1);
    //layer1Weights->m_weightsRange = (float)std::numeric_limits<nn::HiddenLayerWeightType>::max() / nn::HiddenLayerWeightQuantizationScale;
    //layer1Weights->m_biasRange = (float)std::numeric_limits<nn::HiddenLayerWeightType>::max() / nn::HiddenLayerBiasQuantizationScale;

    nn::WeightsStoragePtr lastLayerWeights = std::make_shared<nn::WeightsStorage>(2u * hiddenLayerSize, 1);
    lastLayerWeights->m_weightsRange = (float)std::numeric_limits<nn::LastLayerWeightType>::max() / nn::OutputLayerWeightQuantizationScale;
    lastLayerWeights->m_biasRange = (float)std::numeric_limits<nn::LastLayerBiasType>::max() / nn::OutputLayerBiasQuantizationScale;

    featureTransformerWeights->Init();
    lastLayerWeights->Init();

    nn::NodePtr inputNodeA = std::make_shared<nn::SparseBinaryInputNode>(cNumNetworkInputs, hiddenLayerSize, featureTransformerWeights);
    nn::NodePtr inputNodeB = std::make_shared<nn::SparseBinaryInputNode>(cNumNetworkInputs, hiddenLayerSize, featureTransformerWeights);
    nn::NodePtr concatenationNode = std::make_shared<nn::ConcatenationNode>(inputNodeA, inputNodeB);
    nn::NodePtr activationNode = std::make_shared<nn::ActivationNode>(concatenationNode, nn::ActivationFunction::CReLU);
    nn::NodePtr hiddenNode = std::make_shared<nn::FullyConnectedNode>(activationNode, 2u * hiddenLayerSize, 1, lastLayerWeights);
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

    /*
    const uint32_t hiddenLayerSize = 1024;

    nn::WeightsStoragePtr layer1Weights = std::make_shared<nn::WeightsStorage>(cNumNetworkInputs, hiddenLayerSize);
    nn::WeightsStoragePtr layer2Weights = std::make_shared<nn::WeightsStorage>(hiddenLayerSize, 1);

    layer1Weights->Init();
    layer2Weights->Init();

    nn::NodePtr inputNode = std::make_shared<nn::SparseBinaryInputNode>(cNumNetworkInputs, hiddenLayerSize, layer1Weights);
    nn::NodePtr activationNode = std::make_shared<nn::ActivationNode>(inputNode, nn::ActivationFunction::CReLU);
    nn::NodePtr hiddenNode = std::make_shared<nn::FullyConnectedNode>(activationNode, hiddenLayerSize, 1, layer2Weights);
    nn::NodePtr outputNode = std::make_shared<nn::ActivationNode>(hiddenNode, nn::ActivationFunction::Sigmoid);

    std::vector<nn::NodePtr> nodes =
    {
        inputNode,
        activationNode,
        hiddenNode,
        outputNode,
    };
    */

    m_network.Init(nodes);
    m_trainer.Init(m_network);
    m_runCtx.Init(m_network);

    for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
    {
        m_validationPerThreadData[i].networkRunContext.Init(m_network);
    }
}

static uint32_t GetNetworkVariant(const Position& pos)
{
    const uint32_t numPieceCountBuckets = 8;
    const uint32_t pieceCountBucket = std::min(pos.GetNumPiecesExcludingKing() / 4u, numPieceCountBuckets - 1u);
    const uint32_t queenPresenceBucket = pos.Whites().queens || pos.Blacks().queens;
    return queenPresenceBucket * numPieceCountBuckets + pieceCountBucket;
}

bool NetworkTrainer::GenerateTrainingSet(std::vector<TrainingEntry>& outEntries)
{
    Position pos;
    PositionEntry entry;

    for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
    {
        if (!m_dataLoader.FetchNextPosition(m_randomGenerator, entry, pos))
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
        const float wdlLambda = 1.0f; // std::lerp(0.95f, 0.1f, 1.0f - expf(-(float)pos.GetMoveCount() / 80.0f));

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
            const float tbDrawLambda = 0.1f;
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
        outEntries[i].networkVariant = GetNetworkVariant(pos);
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

            const TrainingEntry& entry = m_trainingSetCopy[i];

            const float expectedValue = entry.output;

            const ScoreType psqtValue = Evaluate(entry.pos, nullptr, false);
            const ScoreType evalValue = Evaluate(entry.pos);

#ifdef USE_PACKED_NET
            const int32_t packedNetworkOutput = m_packedNet.Run(features.data(), (uint32_t)features.size(), variant);
            const float scaledPackedNetworkOutput = (float)packedNetworkOutput / (float)nn::OutputScale * c_nnOutputToCentiPawns / 100.0f;
            const float nnPackedValue = EvalToExpectedGameScore(scaledPackedNetworkOutput /*+ psqtValue / 100.0f*/);
#endif // USE_PACKED_NET

            nn::InputDesc inputDesc;
            inputDesc.variant = entry.networkVariant;
            inputDesc.inputs[0].mode = nn::InputMode::SparseBinary;
            inputDesc.inputs[0].binaryFeatures = entry.whiteFeatures.data();
            inputDesc.inputs[0].numFeatures = static_cast<uint32_t>(entry.whiteFeatures.size());
            inputDesc.inputs[1].mode = nn::InputMode::SparseBinary;
            inputDesc.inputs[1].binaryFeatures = entry.blackFeatures.data();
            inputDesc.inputs[1].numFeatures = static_cast<uint32_t>(entry.blackFeatures.size());

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
                    << "PSQT eval:      " << InternalEvalToExpectedGameScore(psqtValue) << " (" << psqtValue << ")" << std::endl
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
            "r1bq1rk1/1pp2ppp/8/4pn2/B6b/1PN2P2/PBPP1P2/RQ2R1K1 b - - 1 12",
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
            "3k4/3B4/8/8/7p/7P/8/5K1B w - - 0 1", // should be 0
        };

        for (const char* testPosition : s_testPositions)
        {
            Position pos(testPosition);

            TrainingEntry entry;
            PositionToTrainingEntry(pos, entry);

            nn::InputDesc inputDesc;
            inputDesc.variant = GetNetworkVariant(pos);
            inputDesc.inputs[0].mode = nn::InputMode::SparseBinary;
            inputDesc.inputs[0].binaryFeatures = entry.whiteFeatures.data();
            inputDesc.inputs[0].numFeatures = static_cast<uint32_t>(entry.whiteFeatures.size());
            inputDesc.inputs[1].mode = nn::InputMode::SparseBinary;
            inputDesc.inputs[1].binaryFeatures = entry.blackFeatures.data();
            inputDesc.inputs[1].numFeatures = static_cast<uint32_t>(entry.blackFeatures.size());

            const float nnValue = m_network.Run(inputDesc, m_runCtx)[0];

#ifdef USE_PACKED_NET
            const int32_t packedNetworkOutput = m_packedNet.Run(vec.sparseBinaryInputs.data(), (uint32_t)vec.sparseBinaryInputs.size(), inputDesc.variant);
            const float scaledPackedNetworkOutput = (float)packedNetworkOutput / (float)nn::OutputScale * c_nnOutputToCentiPawns / 100.0f;
            const float nnPackedValue = scaledPackedNetworkOutput /* + psqtValue / 100.0f*/;
#endif // USE_PACKED_NET

            std::cout << "TEST " << testPosition << "  " << ExpectedGameScoreToInternalEval(nnValue) << std::endl;
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

/*
template<typename WeightType, typename BiasType>
static void PackWeights(const nn::WeightsStorage& weights, uint32_t variantIdx, WeightType* outWeights, BiasType* outBiases, float weightScale, float biasScale, bool transpose)
{
    const INode::Variant& variant = node.variants[variantIdx];

    // weights
    for (uint32_t j = 0; j < node.numInputs; j++)
    {
        uint32_t i = 0;
#ifdef USE_AVX2
        const float* weightsPtr = variant.weights.data() + j * node.numOutputs;
        for (; i + 8 < node.numOutputs; i += 8)
        {
            const __m256i quantizedWeights =
                _mm256_cvtps_epi32(_mm256_round_ps(
                    _mm256_mul_ps(_mm256_load_ps(weightsPtr + i), _mm256_set1_ps(weightScale)),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            if (transpose)
            {
                outWeights[node.numOutputs * j + (i + 0)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 0);
                outWeights[node.numOutputs * j + (i + 1)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 1);
                outWeights[node.numOutputs * j + (i + 2)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 2);
                outWeights[node.numOutputs * j + (i + 3)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 3);
                outWeights[node.numOutputs * j + (i + 4)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 4);
                outWeights[node.numOutputs * j + (i + 5)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 5);
                outWeights[node.numOutputs * j + (i + 6)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 6);
                outWeights[node.numOutputs * j + (i + 7)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 7);
            }
            else
            {
                outWeights[node.numInputs * (i + 0) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 0);
                outWeights[node.numInputs * (i + 1) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 1);
                outWeights[node.numInputs * (i + 2) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 2);
                outWeights[node.numInputs * (i + 3) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 3);
                outWeights[node.numInputs * (i + 4) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 4);
                outWeights[node.numInputs * (i + 5) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 5);
                outWeights[node.numInputs * (i + 6) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 6);
                outWeights[node.numInputs * (i + 7) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 7);
            }
        }
#endif // USE_AVX2

        for (; i < node.numOutputs; i++)
        {
            const float weight = variant.weights[j * node.numOutputs + i];
            const int32_t quantizedWeight = (int32_t)std::round(weight * weightScale);
            ASSERT(quantizedWeight <= std::numeric_limits<WeightType>::max());
            ASSERT(quantizedWeight >= std::numeric_limits<WeightType>::min());

            if (transpose)
            {
                outWeights[node.numOutputs * j + i] = (WeightType)quantizedWeight;
            }
            else
            {
                outWeights[node.numInputs * i + j] = (WeightType)quantizedWeight;
            }
        }
    }

    // biases
    for (uint32_t i = 0; i < node.numOutputs; i++)
    {
        const float bias = variant.weights[node.numInputs * node.numOutputs + i];
        const int32_t quantizedBias = (int32_t)std::round(bias * biasScale);
        ASSERT(quantizedBias <= std::numeric_limits<BiasType>::max());
        ASSERT(quantizedBias >= std::numeric_limits<BiasType>::min());
        outBiases[i] = (BiasType)quantizedBias;
    }
}

static bool PackNetwork(const nn::NeuralNetwork& net, PackedNeuralNetwork& outNetwork) const
{
    ASSERT(nodes.size() <= PackedNeuralNetwork::MaxNumLayers);
    ASSERT(nodes[0].numOutputs == AccumulatorSize);
    ASSERT(nodes[1].numInputs == AccumulatorSize);
    ASSERT(nodes.back().numOutputs == 1);
    ASSERT(nodes.front().variants.size() == 1);

    {
        std::vector<uint32_t> layerSizes, layerVariants;
        for (const INode& node : nodes)
        {
            layerSizes.push_back(node.numInputs);
            layerVariants.push_back((uint32_t)node.variants.size());
        }

        if (!outNetwork.Resize(layerSizes, layerVariants))
        {
            return false;
        }
    }

    // first node
    PackLayerWeights(nodes.front(),
        0,
        const_cast<FirstLayerWeightType*>(outNetwork.GetAccumulatorWeights()),
        const_cast<FirstLayerBiasType*>(outNetwork.GetAccumulatorBiases()),
        InputLayerWeightQuantizationScale,
        InputLayerBiasQuantizationScale,
        true);

    // hidden nodes
    for (uint32_t i = 1; i + 1 < nodes.size(); ++i)
    {
        for (uint32_t variantIdx = 0; variantIdx < nodes[i].variants.size(); ++variantIdx)
        {
            PackLayerWeights(nodes[i],
                variantIdx,
                const_cast<HiddenLayerWeightType*>(outNetwork.GetLayerWeights<HiddenLayerWeightType>(uint32_t(i), variantIdx)),
                const_cast<HiddenLayerBiasType*>(outNetwork.GetLayerBiases<HiddenLayerBiasType>(uint32_t(i), variantIdx)),
                HiddenLayerWeightQuantizationScale,
                HiddenLayerBiasQuantizationScale,
                false);
        }
    }

    // last node
    const uint32_t lastLayerIndex = (uint32_t)nodes.size() - 1;
    for (uint32_t variantIdx = 0; variantIdx < nodes.back().variants.size(); ++variantIdx)
    {
        PackLayerWeights(nodes.back(),
            variantIdx,
            const_cast<LastLayerWeightType*>(outNetwork.GetLayerWeights<LastLayerWeightType>(lastLayerIndex, variantIdx)),
            const_cast<LastLayerBiasType*>(outNetwork.GetLayerBiases<LastLayerBiasType>(lastLayerIndex, variantIdx)),
            OutputLayerWeightQuantizationScale,
            OutputLayerBiasQuantizationScale,
            false);
    }

    return true;
}
*/

static float CosineAnnealingLR(float phase, float baseLR)
{
    float maxLR = baseLR;
    float minLR = baseLR / 10.0f;
    float annealingFactor = (1.0f + cosf(phase)) / 2.0f;
    return minLR + annealingFactor * (maxLR - minLR);
}

bool NetworkTrainer::Train()
{
    InitNetwork();

    if (!m_dataLoader.Init(m_randomGenerator))
    {
        std::cout << "ERROR: Failed to initialize data loader" << std::endl;
        return false;
    }

    std::vector<nn::TrainingVector> batch(cNumTrainingVectorsPerIteration);

    TimePoint prevIterationStartTime = TimePoint::GetCurrent();

    size_t epoch = 0;
    for (size_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        const float baseLearningRate = 0.75f * expf(-0.00005f * (float)iteration);
        //const float learningRate = CosineAnnealingLR((float)iteration / (float)200.0f, baseLearningRate);
        const float learningRate = baseLearningRate;

        if (iteration == 0)
        {
            if (!GenerateTrainingSet(m_trainingSet))
                return false;
        }

        TimePoint iterationStartTime = TimePoint::GetCurrent();
        float iterationTime = (iterationStartTime - prevIterationStartTime).ToSeconds();
        prevIterationStartTime = iterationStartTime;
        
        // use validation set from previous iteration as training set in the current one
        ParallelFor("PrepareBatch", cNumTrainingVectorsPerIteration, [&batch, this](const TaskContext&, uint32_t i)
        {
            m_trainingSetCopy[i] = m_trainingSet[i];

            const TrainingEntry& entry = m_trainingSetCopy[i];

            nn::TrainingVector& trainingVector = batch[i];
            trainingVector.input.variant = entry.networkVariant;
            trainingVector.output.mode = nn::OutputMode::Single;
            trainingVector.output.singleValue = entry.output;
            trainingVector.input.inputs[0].mode = nn::InputMode::SparseBinary;
            trainingVector.input.inputs[0].numFeatures = static_cast<uint32_t>(entry.whiteFeatures.size());
            trainingVector.input.inputs[0].binaryFeatures = entry.whiteFeatures.data();
            trainingVector.input.inputs[1].mode = nn::InputMode::SparseBinary;
            trainingVector.input.inputs[1].numFeatures = static_cast<uint32_t>(entry.blackFeatures.size());
            trainingVector.input.inputs[1].binaryFeatures = entry.blackFeatures.data();
        });

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder{ waitable };
            taskBuilder.Task("GenerateSet", [&](const TaskContext&)
            {
                GenerateTrainingSet(m_trainingSet);
            });

            taskBuilder.Task("Train", [this, iteration, &epoch, &batch, &learningRate](const TaskContext& ctx)
            {
                nn::TrainParams params;
                params.iteration = epoch;
                params.batchSize = std::min<size_t>(cMinBatchSize + iteration * cMinBatchSize, cMaxBatchSize);
                params.learningRate = learningRate;
                params.weightDecay = 1.0e-6f;

                TaskBuilder taskBuilder{ ctx };
                epoch += m_trainer.Train(m_network, batch, params, &taskBuilder);
            });
        }
        waitable.Wait();

#ifdef USE_PACKED_NET
        m_network.ToPackedNetwork(m_packedNet);
        ASSERT(m_packedNet.IsValid());
#endif // USE_PACKED_NET

        m_numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        std::cout
            << "Iteration:              " << iteration << std::endl
            << "Epoch:                  " << epoch << std::endl
            << "Num training vectors:   " << m_numTrainingVectorsPassed << std::endl
            << "Learning rate:          " << learningRate << " (" << baseLearningRate << " base)" << std::endl;

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
