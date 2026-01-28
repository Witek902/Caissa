#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"

#include "trainer/CudaNetwork.hpp"
#include "trainer/CudaCommon.hpp"

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

#include "minitrace/minitrace.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>
#include <cmath>

#define USE_PACKED_NET

using namespace threadpool;

static const uint32_t cMaxIterations = 1'000'000'000;
static const uint32_t cNumTrainingVectorsPerIteration = 512 * 1024;
static const uint32_t cNumValidationVectorsPerIteration = 256 * 1024;
static const uint32_t cBatchSize = 32 * 1024;

class CudaNetworkTrainer
{
public:
    CudaNetworkTrainer()
        : m_randomGenerator(m_randomDevice())
        , m_trainingLog("training.log")
    {
        m_packedNet = std::make_unique<nn::PackedNeuralNetwork>();

        m_validationSet.resize(cNumTrainingVectorsPerIteration);
        m_trainingSet_Read.resize(cNumTrainingVectorsPerIteration);
        m_trainingSet_Write.resize(cNumTrainingVectorsPerIteration);
        m_validationPerThreadData.resize(ThreadPool::GetInstance().GetNumThreads());

        // Initialize CUDA batch data
        m_cudaBatchData.Allocate(cBatchSize);

        // Choose which GPU to run on, change this on a multi-GPU system.
        CUDA_CHECK(cudaSetDevice(0));

        // Print CUDA device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device name: %s\n", prop.name);
    }

    void InitNetwork();

    bool Train();

private:

    struct ValidationStats
    {
#ifdef USE_PACKED_NET
        float nnPackedMinError = std::numeric_limits<float>::max();
        float nnPackedMaxError = 0.0f;
        double nnPackedErrorSum = 0.0f;
#endif // USE_PACKED_NET

        float evalMinError = std::numeric_limits<float>::max();
        float evalMaxError = 0.0f;
        double evalErrorSum = 0.0f;
    };

    struct alignas(CACHELINE_SIZE) ValidationPerThreadData
    {
        ValidationStats stats;
        uint8_t __padding[CACHELINE_SIZE];
    };

    TrainingDataLoader m_dataLoader;

    nn::WeightsStoragePtr m_featureTransformerWeights;
    nn::WeightsStoragePtr m_lastLayerWeights;

    nn::cuda::CudaNeuralNetwork m_cudaNetwork;
    nn::cuda::CudaBatchData m_cudaBatchData;
#ifdef USE_PACKED_NET
    std::unique_ptr<nn::PackedNeuralNetwork> m_packedNet;
#endif // USE_PACKED_NET

    std::vector<TrainingEntry> m_validationSet; // TODO remove
    std::vector<TrainingEntry> m_trainingSet_Write; // training set written to by IO
    std::vector<TrainingEntry> m_trainingSet_Read; // training set used by trainer
    std::vector<ValidationPerThreadData> m_validationPerThreadData;

    alignas(CACHELINE_SIZE)
    std::atomic<uint64_t> m_numTrainingVectorsPassed = 0;

    alignas(CACHELINE_SIZE)
    std::mutex m_mutex;

    std::random_device m_randomDevice;
    std::mt19937 m_randomGenerator;

    std::ofstream m_trainingLog;

    bool GenerateTrainingSet(std::vector<TrainingEntry>& outEntries, uint64_t kingBucketMask, float baseLambda);

    void Validate(const TaskContext& ctx, size_t iteration);

    void BlendLastLayerWeights();

    bool PackNetwork();
    bool UnpackNetwork();

    // CUDA-specific methods
    void PrepareCudaBatch(const std::vector<TrainingEntry>& entries, uint32_t startIndex, uint32_t batchSize);
    void RunCudaTrainingIteration(float learningRate, size_t iteration);
};

void CudaNetworkTrainer::InitNetwork()
{
    const uint32_t accumulatorSize = nn::AccumulatorSize;
    const uint32_t networkInputs = nn::NumNetworkInputs;

    m_featureTransformerWeights = std::make_shared<nn::WeightsStorage>(networkInputs, accumulatorSize, 1);
    m_featureTransformerWeights->m_isSparse = true;
    // divide by number of active input features to avoid accumulator overflow
    m_featureTransformerWeights->m_weightsRange = (float)std::numeric_limits<nn::FirstLayerWeightType>::max() / 16 / nn::InputLayerWeightQuantizationScale;
    m_featureTransformerWeights->m_biasRange = (float)std::numeric_limits<nn::FirstLayerBiasType>::max() / 16 / nn::InputLayerBiasQuantizationScale;
    m_featureTransformerWeights->Init(32u, 0.0f);

    m_lastLayerWeights = std::make_shared<nn::WeightsStorage>(2u * accumulatorSize, 1, nn::NumVariants);
    m_lastLayerWeights->m_weightsRange = (float)std::numeric_limits<nn::LastLayerWeightType>::max() / nn::OutputLayerWeightQuantizationScale;
    m_lastLayerWeights->m_biasRange = (float)std::numeric_limits<nn::LastLayerBiasType>::max() / nn::OutputLayerBiasQuantizationScale;
    m_lastLayerWeights->Init(2 * nn::AccumulatorSize);

    // Initialize CUDA network
    m_cudaNetwork.Init(m_featureTransformerWeights, m_lastLayerWeights);
}

static void PositionToTrainingEntry(const Position& pos, TrainingEntry& outEntry)
{
    ASSERT(pos.GetSideToMove() == White);

    constexpr uint32_t maxFeatures = 64;

    uint16_t whiteFeatures[maxFeatures];
    uint32_t numWhiteFeatures = PositionToFeaturesVector<false>(pos, whiteFeatures, pos.GetSideToMove());
    ASSERT(numWhiteFeatures <= maxFeatures);

    uint16_t blackFeatures[maxFeatures];
    uint32_t numBlackFeatures = PositionToFeaturesVector<false>(pos, blackFeatures, pos.GetSideToMove() ^ 1);
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

bool CudaNetworkTrainer::GenerateTrainingSet(std::vector<TrainingEntry>& outEntries, uint64_t kingBucketMask, float baseLambda)
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
            const float tbDrawLambda = 0.0f;
            score = std::lerp(0.5f, score, tbDrawLambda);
        }
        else if (tbScore != Game::Score::Unknown)
        {
            const float tbLambda = 0.0f;
            const float wdlScore = tbScore == Game::Score::WhiteWins ? 1.0f : (tbScore == Game::Score::BlackWins ? 0.0f : 0.5f);
            score = std::lerp(wdlScore, score, tbLambda);
        }

        PositionToTrainingEntry(pos, outEntries[i]);
        outEntries[i].output = score;
        outEntries[i].pos = pos;
    }

    return true;
}

void CudaNetworkTrainer::PrepareCudaBatch(const std::vector<TrainingEntry>& entries, uint32_t startIndex, uint32_t batchSize)
{
    std::vector<nn::cuda::CudaTrainingVector> hostBatch(batchSize);

    for (uint32_t i = 0; i < batchSize; ++i)
    {
        const TrainingEntry& entry = entries[startIndex + i];
        nn::cuda::CudaTrainingVector& cudaEntry = hostBatch[i];

        cudaEntry.variant = entry.networkVariant;
        cudaEntry.targetOutput = entry.output;

        cudaEntry.numWhiteFeatures = std::min(32u, (uint32_t)entry.whiteFeatures.size());
        cudaEntry.numBlackFeatures = std::min(32u, (uint32_t)entry.blackFeatures.size());

        std::fill_n(cudaEntry.whiteFeatures, 32, (uint16_t)0);
        std::fill_n(cudaEntry.blackFeatures, 32, (uint16_t)0);

        for (uint32_t j = 0; j < cudaEntry.numWhiteFeatures; ++j)
            cudaEntry.whiteFeatures[j] = entry.whiteFeatures[j];
        for (uint32_t j = 0; j < cudaEntry.numBlackFeatures; ++j)
            cudaEntry.blackFeatures[j] = entry.blackFeatures[j];
    }

    m_cudaBatchData.trainingVectors.CopyFromHost(hostBatch.data(), batchSize);
}

void CudaNetworkTrainer::RunCudaTrainingIteration(float learningRate, size_t iteration)
{
    // Process training set in batches
    for (uint32_t batchStart = 0; batchStart < cNumTrainingVectorsPerIteration; batchStart += cBatchSize)
    {
        const uint32_t currentBatchSize = std::min(cBatchSize, cNumTrainingVectorsPerIteration - batchStart);

        // Prepare batch data for GPU
        PrepareCudaBatch(m_trainingSet_Read, batchStart, currentBatchSize);

        // Update batch size in CUDA structure
        m_cudaBatchData.batchSize = currentBatchSize;

        // Forward pass
        m_cudaNetwork.Forward(m_cudaBatchData);

        // Backward pass
        m_cudaNetwork.Backward(m_cudaBatchData, learningRate, 1.0f / 2048.0f, iteration);
    }

    // Copy weights from CUDA to host
    m_cudaNetwork.CopyWeightsToHost(m_featureTransformerWeights, m_lastLayerWeights);
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
    const float scaledPackedNetworkOutput = (float)packedNetworkOutput / (float)(nn::OutputScale * nn::WeightScale) * c_nnOutputToCentiPawns / 100.0f;
    return EvalToExpectedGameScore(scaledPackedNetworkOutput);
}
#endif // USE_PACKED_NET

void CudaNetworkTrainer::Validate(const TaskContext& ctx, size_t iteration)
{
    // reset stats
    for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
    {
        m_validationPerThreadData[i].stats = ValidationStats();
    }

    TaskBuilder taskBuilder(ctx);
    taskBuilder.ParallelFor("Validate", cNumValidationVectorsPerIteration, [this](const TaskContext& ctx, uint32_t i)
    {
        ValidationPerThreadData& threadData = m_validationPerThreadData[ctx.threadId];

        const TrainingEntry& entry = m_validationSet[i];

        const float expectedValue = entry.output;

        const ScoreType evalValue = Evaluate(entry.pos);

#ifdef USE_PACKED_NET
        const float nnPackedValue = EvalPackedNetwork(entry, *m_packedNet);
#endif // USE_PACKED_NET

        ValidationStats& stats = threadData.stats;
        {
            const float error = expectedValue - InternalEvalToExpectedGameScore(evalValue);
            const float errorDiff = std::abs(error);
            stats.evalErrorSum += (double)error * (double)error;
            stats.evalMinError = std::min(stats.evalMinError, errorDiff);
            stats.evalMaxError = std::max(stats.evalMaxError, errorDiff);
        }
#ifdef USE_PACKED_NET
        {
            const float error = expectedValue - nnPackedValue;
            const float errorDiff = std::abs(error);
            stats.nnPackedErrorSum += (double)error * (double)error;
            stats.nnPackedMinError = std::min(stats.nnPackedMinError, errorDiff);
            stats.nnPackedMaxError = std::max(stats.nnPackedMaxError, errorDiff);
        }
#endif // USE_PACKED_NET
    });

    taskBuilder.Fence();

    taskBuilder.Task("PrintValidationStats", [this, iteration](const TaskContext&)
    {
        // accumulate stats
        ValidationStats stats;
        for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
        {
            const ValidationStats& threadStats = m_validationPerThreadData[i].stats;

#ifdef USE_PACKED_NET
            stats.nnPackedErrorSum += threadStats.nnPackedErrorSum;
            stats.nnPackedMinError = std::min(stats.nnPackedMinError, threadStats.nnPackedMinError);
            stats.nnPackedMaxError = std::max(stats.nnPackedMaxError, threadStats.nnPackedMaxError);
#endif // USE_PACKED_NET
            stats.evalErrorSum += threadStats.evalErrorSum;
            stats.evalMinError = std::min(stats.evalMinError, threadStats.evalMinError);
            stats.evalMaxError = std::max(stats.evalMaxError, threadStats.evalMaxError);
        }

        stats.evalErrorSum = sqrt(stats.evalErrorSum / cNumValidationVectorsPerIteration);
#ifdef USE_PACKED_NET
        stats.nnPackedErrorSum = sqrt(stats.nnPackedErrorSum / cNumValidationVectorsPerIteration);
#endif // USE_PACKED_NET


        std::cout
            << "-------------------------------------------------------------------------\n"
#ifdef USE_PACKED_NET
            << "PNN avg/min/max error:  " << std::setprecision(6) << stats.nnPackedErrorSum << " " << std::setprecision(5) << stats.nnPackedMinError << " " << std::setprecision(5) << stats.nnPackedMaxError << '\n'
#endif // USE_PACKED_NET
            << "Eval avg/min/max error: " << std::setprecision(6) << stats.evalErrorSum << " " << std::setprecision(5) << stats.evalMinError << " " << std::setprecision(5) << stats.evalMaxError << '\n';

        {
            const char* s_testPositions[] =
            {
                Position::InitPositionFEN,
                "rnbq1bnr/pppppppp/8/8/5k2/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",         // black king in the center
                "r1bq1rk1/1pp2ppp/8/4pn2/B6b/1PN2P2/PBPP1P2/RQ2R1K1 w - - 1 12",
                "8/1kN5/8/2B5/4K1bN/8/8/8 w - - 0 1", // should be 1
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
                "8/8/2k3N1/8/Nn2N3/4K3/8/7n w - - 0 1", // should be 1
                "rnbqk1nr/3p1pbp/p1pPp1p1/PpP5/1P6/8/4PPPP/1NBQKBNR w kq - 1 9", // should be 1?
                "rn1qkbnr/pbp1p3/1p1pPp1p/5PpP/6P1/8/PPPP4/RNBQKBN1 w Qkq - 1 9", // should be 1?
            };

            for (const char* testPosition : s_testPositions)
            {
                Position pos(testPosition);

                TrainingEntry entry;
                PositionToTrainingEntry(pos, entry);

#ifdef USE_PACKED_NET
                const float scaledPackedNetworkOutput = EvalPackedNetwork(entry, *m_packedNet);
#endif // USE_PACKED_NET

                std::cout
                    << "TEST " << testPosition
#ifdef USE_PACKED_NET
                    << "  pnn=" << ExpectedGameScoreToInternalEval(scaledPackedNetworkOutput)
#endif // USE_PACKED_NET
                    << '\n';
            }
        }

        m_trainingLog
            << iteration << "\t"
#ifdef USE_PACKED_NET
            << "\t" << std::setprecision(8) << stats.nnPackedErrorSum
#endif // USE_PACKED_NET
            << std::endl;
    });
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

bool CudaNetworkTrainer::PackNetwork()
{
    MTR_BEGIN("CudaNetworkTrainer", "PackNetwork");

    // feature transformer
    {
        const nn::Values weights = m_featureTransformerWeights->m_variants.front().m_weights;

        PackWeights(
            weights,
            nn::NumNetworkInputs,
            nn::AccumulatorSize,
            const_cast<nn::FirstLayerWeightType*>(m_packedNet->accumulatorWeights),
            const_cast<nn::FirstLayerBiasType*>(m_packedNet->accumulatorBiases),
            nn::InputLayerWeightQuantizationScale,
            nn::InputLayerBiasQuantizationScale,
            true);
    }

    // last layer
    for (uint32_t variantIdx = 0; variantIdx < nn::NumVariants; ++variantIdx)
    {
        PackWeights(
            m_lastLayerWeights->m_variants[variantIdx].m_weights,
            m_lastLayerWeights->m_inputSize,
            1u,
            const_cast<nn::LastLayerWeightType*>(m_packedNet->lastLayerVariants[variantIdx].weights),
            const_cast<nn::LastLayerBiasType*>(&m_packedNet->lastLayerVariants[variantIdx].bias),
            nn::OutputLayerWeightQuantizationScale,
            nn::OutputLayerBiasQuantizationScale,
            false);
    }

    MTR_END("CudaNetworkTrainer", "PackNetwork");
    return true;
}

bool CudaNetworkTrainer::UnpackNetwork()
{
    constexpr float OldActivationRangeScaling = 256;
    constexpr int32_t OldWeightScaleShift = 8; // TODO should be 6 if we clamp weights to [-2,2] range
    constexpr int32_t OldWeightScale = 1 << OldWeightScaleShift;
    constexpr int32_t OldOutputScaleShift = 10;
    constexpr int32_t OldOutputScale = 1 << OldOutputScaleShift;
    constexpr float OldInputLayerWeightQuantizationScale = OldActivationRangeScaling;
    constexpr float OldInputLayerBiasQuantizationScale = OldActivationRangeScaling;
    constexpr float OldOutputLayerWeightQuantizationScale = OldWeightScale * OldOutputScale / OldActivationRangeScaling;
    constexpr float OldOutputLayerBiasQuantizationScale = OldWeightScale * OldOutputScale;

    // feature transformer
    {
        UnpackWeights(
            m_featureTransformerWeights->m_variants.front().m_weights,
            nn::NumNetworkInputs,
            nn::AccumulatorSize,
            m_packedNet->accumulatorWeights,
            m_packedNet->accumulatorBiases,
            OldInputLayerWeightQuantizationScale,
            OldInputLayerBiasQuantizationScale,
            true);
    }

    // last layer
    for (uint32_t variantIdx = 0; variantIdx < nn::NumVariants; ++variantIdx)
    {
        UnpackWeights(
            m_lastLayerWeights->m_variants[variantIdx].m_weights,
            m_lastLayerWeights->m_inputSize,
            1u,
            m_packedNet->lastLayerVariants[variantIdx].weights,
            &m_packedNet->lastLayerVariants[variantIdx].bias,
            OldOutputLayerWeightQuantizationScale,
            OldOutputLayerBiasQuantizationScale,
            false);
    }

    return true;
}

static const float g_warmupTime = 20.0f;
static volatile float g_learningRateScale = 1.0f;
static volatile float g_lambdaScale = 0.0f;

bool CudaNetworkTrainer::Train()
{
    InitNetwork();

    if (!m_packedNet->LoadFromFile("eval-69-44B.pnn"))
    {
        std::cout << "ERROR: Failed to load packed network" << std::endl;
        return false;
    }
    UnpackNetwork();

    // Copy unpacked weights to CUDA
    m_cudaNetwork.CopyWeightsFromHost(m_featureTransformerWeights, m_lastLayerWeights);

    if (!m_dataLoader.Init(m_randomGenerator))
    {
        std::cout << "ERROR: Failed to initialize data loader" << std::endl;
        return false;
    }

    TimePoint prevIterationStartTime = TimePoint::GetCurrent();

    const float baseLearningRate = 1.0e-4f;
    const float maxLambda = 1.0f;
    const float minLambda = 1.0f;

    uint64_t kingBucketMask = UINT64_MAX;

    GenerateTrainingSet(m_validationSet, kingBucketMask, maxLambda);

    size_t epoch = 0;
    for (size_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        const float warmup = g_warmupTime > 0.0f ? (iteration < g_warmupTime ? (float)(iteration + 1) / g_warmupTime : 1.0f) : 1.0f;
        const float learningRate = g_learningRateScale * warmup * baseLearningRate;
        const float lambda = g_lambdaScale * std::lerp(minLambda, maxLambda, expf(-0.0005f * (float)iteration));

        TimePoint iterationStartTime = TimePoint::GetCurrent();
        float iterationTime = (iterationStartTime - prevIterationStartTime).ToSeconds();
        prevIterationStartTime = iterationStartTime;

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder{ waitable };
            taskBuilder.Task("GenerateSet", [this, kingBucketMask, lambda](const TaskContext&)
            {
                GenerateTrainingSet(m_trainingSet_Write, kingBucketMask, lambda);
            });

            // skip training in the first iteration, as the data is not ready yet
            if (iteration > 0)
            {
                taskBuilder.Task("CudaTrain", [this, learningRate, &epoch, iteration](const TaskContext&)
                {
                    RunCudaTrainingIteration(learningRate, iteration);
                    epoch += cNumTrainingVectorsPerIteration / cBatchSize;
                });
            }

            if (iteration > 1)
            {
                taskBuilder.Task("Validate", [this, learningRate, &epoch, iteration](const TaskContext& ctx)
                {
                    Validate(ctx, iteration);
                });
            }
        }
        waitable.Wait();

        // swap read and write buffers
        std::swap(m_trainingSet_Write, m_trainingSet_Read);

#ifdef USE_PACKED_NET
        PackNetwork();
#endif // USE_PACKED_NET

        m_numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        /*
        // print weights stats
        {
            std::cout << "FT weights stats: ";
            m_featureTransformerWeights->PrintStats();

            std::cout << "LL weights stats: ";
            m_lastLayerWeights->PrintStats();
        }
        */

        std::cout
            << "Lambda:                 " << lambda << '\n'
            << "Iteration:              " << iteration << '\n'
            << "Epoch:                  " << epoch << '\n'
            << "Num training vectors:   " << std::setprecision(3) << m_numTrainingVectorsPassed / 1.0e9f << "B" << '\n'
            << "Learning rate:          " << learningRate << '\n';

        std::cout << "Iteration time:   " << 1000.0f * iterationTime << " ms" << '\n';
        std::cout << "Training rate :   " << ((float)cNumTrainingVectorsPerIteration / iterationTime) << " pos/sec" << std::endl;

        if (iteration % 10 == 0)
        {
            const std::string name = "eval";
#ifdef USE_PACKED_NET
            m_packedNet->SaveToFile((name + ".pnn").c_str());
#endif // USE_PACKED_NET
        }
    }

    return true;
}

bool TrainCudaNetwork()
{
    CudaNetworkTrainer trainer;
    return trainer.Train();
}
