#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"
#include "cudaTrainer/CudaNetwork.hpp"
#include "../backend/Search.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Waitable.hpp"
#include "minitrace/minitrace.h"

#include <ctime>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>

#define USE_PACKED_NET_VALIDATION
// #define USE_EVAL_VALIDATION

using namespace threadpool;

static constexpr uint32_t cNumTrainingVectorsPerIteration = 2 * 1024 * 1024;
static constexpr uint32_t cNumValidationVectorsPerIteration = 256 * 1024;
// The float reference network is evaluated on a prefix of the validation set only - it runs the
// dense feature transformer on the CPU, so it is far too slow for the whole set.
static constexpr uint32_t cNumFloatReferenceVectors = 4 * 1024;

// A packed net and a full training checkpoint are kept every this many training positions
static constexpr uint64_t cCheckpointInterval = 10'000'000'000ull;

// iterations of linear learning rate warmup when starting from an existing net
static const float cWarmupTime = 50.0f;

// seconds between progress reports
static constexpr float cReportInterval = 10.0f;

// weight statistics and test position evals are written to files every this many training positions
static constexpr uint64_t cDetailsInterval = 1'000'000'000ull;

// e.g. "1:23:45"
static std::string FormatDuration(double seconds)
{
    const uint64_t totalSeconds = (uint64_t)seconds;
    char buf[32];
    snprintf(buf, sizeof(buf), "%llu:%02u:%02u", (unsigned long long)(totalSeconds / 3600), (uint32_t)(totalSeconds / 60 % 60), (uint32_t)(totalSeconds % 60));
    return buf;
}

static const char* c_layerNames[] = { "FT", "L1", "L2", "L3" };

static const char* c_testPositions[] =
{
    Position::InitPositionFEN,
    "rnbq1bnr/pppppppp/8/8/5k2/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",         // black king in the center
    "r1bq1rk1/1pp2ppp/8/4pn2/B6b/1PN2P2/PBPP1P2/RQ2R1K1 w - - 1 12",
    "8/1kN5/8/2B5/4K1bN/8/8/8 w - - 0 1", // should be 1
    "k7/ppp5/8/8/8/8/P7/K7 w - - 0 1",  // should be at least -200
    "7k/pp6/8/8/8/8/PP6/7K w - - 0 1",   // should be 0
    "k7/pp6/8/8/8/8/P7/K7 w - - 0 1",   // should be 0
    "8/7p/8/6k1/3q3p/4R3/5PK1/8 w - - 0 1", // should be 0
    "8/1k6/1p6/1R6/2P5/1P6/1K6/4q3 w - - 0 1", // should be 0
    "8/8/5k2/6p1/8/1P2R3/2q2P2/6K1 w - - 0 1", // should be 0
    "4k3/5p2/2K1p3/1Q1rP3/8/8/8/8 w - - 0 1", // should be 0
    "8/8/8/5B1p/5p1r/4kP2/6K1/8 w - - 0 1", // should be 0
    "8/8/8/p7/K5R1/1n6/1k1r4/8 w - - 0 1", // should be 0
    "8/8/2k3N1/8/Nn2N3/4K3/8/7n w - - 0 1", // should be 1
    "8/8/8/8/3r4/6p1/1RP2k2/1K6 w - - 0 1", // should be 1
    "8/8/8/8/3r4/6p1/1RP2k2/1K6 w - - 0 1", // should be 1
    "8/2p5/pp6/4P1p1/8/2k5/2P5/3K4 w - - 0 1", // should be 1
    "1Q6/5p2/4k3/6p1/1K6/8/4q3/8 w - - 0 1", // should be 1
    "8/8/k1K5/3Q4/8/8/4q2q/8 w - - 0 1", // should be 1
    "rnbqk1nr/3p1pbp/p1pPp1p1/PpP5/1P6/8/4PPPP/1NBQKBNR w kq - 1 9", // should be 1?
    "rn1qkbnr/pbp1p3/1p1pPp1p/5PpP/6P1/8/PPPP4/RNBQKBN1 w Qkq - 1 9", // should be 1?
};

// AdamW decoupled weight decay of the output subnet (applied to weights only, not biases)
static constexpr float cOutputSubnetWeightDecay = 0.0f;

struct Options
{
    // full command line, recorded in the run settings file
    std::string commandLine;

    // empty = train from scratch
    std::string startNetPath;

    // checkpoint to continue an interrupted run from
    std::string resumePath;

    // output directory and file name prefix
    std::string name = "eval";

    // number of training iterations to run; after reaching this the trainer exits
    size_t maxIterations = std::numeric_limits<size_t>::max();

    // seed for weights initialization
    uint32_t seed = 12345;

    // cosine learning rate decay: starts at startLearningRate and reaches cEndLearningRate after the
    // 'training length', then stays constant for the rest of the training
    float startLearningRate = 1.0e-4f;
    float endLearningRate = 2.5e-6f;

    // number of training positions to process in whole LR schedule (in billions)
    uint64_t trainingLength = 150;

    // start a new schedule from a checkpoint: the restored weights and optimizer state are kept,
    // but the position counter is reset, so the learning rate and milestones restart
    bool restartSchedule = false;

    // fine-tune the output subnets only, keeping the feature transformer fixed
    bool freezeFeatureTransformer = false;

    // re-seed output subnet neurons that can no longer learn, see ReviveDeadNeurons
    bool reviveDeadNeurons = false;

    // probability that a training position picks its bucket as if it had one piece more or one piece
    // less, so positions at a bucket range edge also train the neighbouring subnet
    float bucketLeak = 0.0f;

    // blend of the training targets: 0 = pure game result, 1 = evaluation (decaying toward the
    // game result with move count). The validation set always uses 1.
    float startLambda = 0.0f;   // at the beginning of training
    float endLambda = 0.0f;     // at the end of training

    // positions per optimizer step; must divide cNumTrainingVectorsPerIteration
    uint32_t batchSize = 32 * 1024;

    TrainingDataLoader::Options loaderOptions;

    // AdamW
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float featureTransformerWeightDecay = 0.0025f; // applied to weights only, not biases
};

class CudaNetworkTrainer
{
public:
    CudaNetworkTrainer(const Options& options)
        : m_options(options)
        , m_deterministicRng(options.seed)
    {
        std::filesystem::create_directories(options.name);

        // a resumed run appends to the files of the interrupted one
        const bool append = !options.resumePath.empty();
        OpenTsvFile(m_progressFile, "-progress.tsv", append,
            "iteration\tpositions\telapsed_s\tlearning_rate\tlambda\ttrain_rmse\tval_rmse\tval_float_subset_rmse\tval_pnn_subset_rmse"
            "\ttrain_ms\tgpu_ms\tloader_ms\tvalidation_ms");
        OpenTsvFile(m_weightsFile, "-weights.tsv", append,
            "positions\tlayer\tw_min\tw_max\tw_avg\tw_std\tb_min\tb_max\tb_avg\tb_std");
        OpenTsvFile(m_testPositionsFile, "-test-positions.tsv", append,
            "positions\tindex\teval\tfen");

        m_packedNet = std::make_unique<nn::PackedNeuralNetwork>();

        m_validationSet.resize(cNumTrainingVectorsPerIteration);
        m_trainingSet_Read.resize(cNumTrainingVectorsPerIteration);
        m_trainingSet_Write.resize(cNumTrainingVectorsPerIteration);
        m_validationPerThreadData.resize(ThreadPool::GetInstance().GetNumThreads());

        for (uint32_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
        {
            std::seed_seq seedSeq{ options.seed, i };
            m_randomGenerators.emplace_back(seedSeq);
        }

        // Initialize CUDA batch data
        m_cudaBatchData.Allocate(options.batchSize);

        // Choose which GPU to run on, change this on a multi-GPU system.
        CUDA_CHECK(cudaSetDevice(0));

        // Print CUDA device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device name: %s\n", prop.name);
        m_deviceName = prop.name;

        // Pin the training set buffers so per-batch host->device copies are truly asynchronous.
        // Both are registered: they are only ever std::swap'd, never reallocated after this point.
        CUDA_CHECK(cudaHostRegister(m_trainingSet_Read.data(), m_trainingSet_Read.size() * sizeof(TrainingEntry), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(m_trainingSet_Write.data(), m_trainingSet_Write.size() * sizeof(TrainingEntry), cudaHostRegisterDefault));
    }

    ~CudaNetworkTrainer()
    {
        cudaHostUnregister(m_trainingSet_Read.data());
        cudaHostUnregister(m_trainingSet_Write.data());
    }

    void InitNetwork();

    bool Train();

private:

    struct ValidationStats
    {
#ifdef USE_PACKED_NET_VALIDATION
        double nnPackedErrorSum = 0.0f;
        // same positions as floatErrorSum, so the two are directly comparable
        double nnPackedSubsetErrorSum = 0.0f;
        double floatErrorSum = 0.0f;
#endif // USE_PACKED_NET_VALIDATION
#ifdef USE_EVAL_VALIDATION
        float evalMinError = std::numeric_limits<float>::max();
        float evalMaxError = 0.0f;
        double evalErrorSum = 0.0f;
#endif // USE_EVAL_VALIDATION
    };

    struct alignas(CACHELINE_SIZE) ValidationPerThreadData
    {
        ValidationStats stats;
        uint8_t __padding[CACHELINE_SIZE];
    };

    TrainingDataLoader m_dataLoader;

    nn::WeightsStoragePtr m_featureTransformerWeights;
    nn::WeightsStoragePtr m_l1Weights;
    nn::WeightsStoragePtr m_l2Weights;
    nn::WeightsStoragePtr m_l3Weights;

    nn::cuda::CudaNeuralNetwork m_cudaNetwork;
    nn::cuda::CudaBatchData m_cudaBatchData;
#ifdef USE_PACKED_NET_VALIDATION
    std::unique_ptr<nn::PackedNeuralNetwork> m_packedNet;
#endif // USE_PACKED_NET_VALIDATION

    TrainingDataSet m_validationSet; // TODO remove
    TrainingDataSet m_trainingSet_Write; // training set written to by IO
    TrainingDataSet m_trainingSet_Read; // training set used by trainer
    std::vector<ValidationPerThreadData> m_validationPerThreadData;

    alignas(CACHELINE_SIZE)
    std::atomic<uint64_t> m_numTrainingVectorsPassed = 0;

    // GPU time of the last training iteration, excluding the concurrent data generation
    float m_lastIterationGpuTimeMs = 0.0f;

    // training-set RMSE of the last training iteration; written by the training task, read by Validate
    std::atomic<float> m_lastTrainingRmse = 0.0f;

    // wall times of the last iteration's concurrent tasks; the slowest one limits the training speed
    float m_lastTrainTaskMs = 0.0f;
    float m_lastLoaderMs = 0.0f;
    float m_lastValidationMs = 0.0f;
    TimePoint m_validationStartTime;

#ifdef USE_PACKED_NET_VALIDATION
    // validation RMSEs of the last iteration; written by Validate, read after the iteration's tasks finish
    float m_lastPackedNetRmse = 0.0f;
    float m_lastPackedNetSubsetRmse = 0.0f;
    float m_lastFloatNetSubsetRmse = 0.0f;
#endif // USE_PACKED_NET_VALIDATION

    alignas(CACHELINE_SIZE)
    std::mutex m_mutex;

    Options m_options;
    std::vector<std::mt19937> m_randomGenerators; // per-thread RNGs
    std::mt19937 m_deterministicRng; // validation set generation

    std::string m_deviceName;

    // tab-separated files in the output directory, one header row each
    std::ofstream m_progressFile;
    std::ofstream m_weightsFile;
    std::ofstream m_testPositionsFile;

    struct NetworkDetails
    {
        nn::WeightsStorage::Stats layerStats[std::size(c_layerNames)];
        int32_t testPositionEvals[std::size(c_testPositions)] = {};
    };

    void OpenTsvFile(std::ofstream& file, const char* suffix, bool append, const char* header) const
    {
        const std::string path = OutputPath(suffix);
        const bool writeHeader = !append || !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
        file.open(path, append ? std::ios::app : std::ios::trunc);
        if (writeHeader)
            file << header << std::endl;
    }

    // run settings, so a net can be traced back to the options and data it was trained with
    void WriteRunSettings(uint64_t startPositions) const;

    void GenerateTrainingEntry(std::mt19937& rng, TrainingEntry& outEntry, uint64_t kingBucketMask, float lambda, float bucketLeak);

    // deterministic: single-threaded with a dedicated RNG, so the set depends only on the seed
    void GenerateTrainingSet(TrainingDataSet& outSet, TaskBuilder& builder, uint64_t kingBucketMask, float lambda, float bucketLeak, bool deterministic = false);

    void Validate(const TaskContext& ctx);

    NetworkDetails CollectNetworkDetails() const;
    void WriteNetworkDetails(const NetworkDetails& details, uint64_t numPositions);
    void PrintNetworkDetails(const NetworkDetails& details, int indent) const;

    // Forward pass on the float weights, matching what the CUDA trainer optimizes
    float EvalFloatNetwork(const TrainingEntry& entry) const;

    // "<name>/<name><suffix>", e.g. "eval-86/eval-86-10B.pnn"
    std::string OutputPath(const std::string& suffix) const
    {
        return m_options.name + "/" + m_options.name + suffix;
    }

    bool PackNetwork();

    // Loads a packed net. Version 12 has no output subnet, so outHasOutputSubnet reports whether
    // the caller still needs to initialize one.
    bool UnpackNetwork(const char* path, bool& outHasOutputSubnet);

    void ReviveDeadNeurons();

    // CUDA-specific methods
    void RunCudaTrainingIteration(float learningRate);
};

void CudaNetworkTrainer::InitNetwork()
{
    const uint32_t accumulatorSize = nn::AccumulatorSize;

    m_featureTransformerWeights = std::make_shared<nn::WeightsStorage>(nn::cuda::FeatureTransformerInputs, accumulatorSize, 1);
    m_featureTransformerWeights->m_isSparse = true;
    // divide by number of active input features to avoid accumulator overflow
    m_featureTransformerWeights->m_weightsRange = 6.0f; // (float)std::numeric_limits<nn::FirstLayerWeightType>::max() / 16 / nn::InputLayerWeightQuantizationScale;
    m_featureTransformerWeights->m_biasRange = 6.0f; //(float)std::numeric_limits<nn::FirstLayerBiasType>::max() / 16 / nn::InputLayerBiasQuantizationScale;
    m_featureTransformerWeights->Init(32u, 0.0f);

    // L1/L2 weights are int8; the range is exactly what the quantization scale can represent.
    // Their biases are int32, so the range only has to keep the layer's int32 sum from overflowing.
    constexpr float hiddenWeightRange = INT8_MAX / nn::HiddenLayerWeightQuantizationScale;
    constexpr float hiddenBiasRange = 64.0f;

    m_l1Weights = std::make_shared<nn::WeightsStorage>(nn::L1InputSize, nn::L1Size, nn::NumVariants);
    m_l1Weights->m_weightsRange = hiddenWeightRange;
    m_l1Weights->m_biasRange = hiddenBiasRange;
    m_l1Weights->Init(nn::L1InputSize);

    m_l2Weights = std::make_shared<nn::WeightsStorage>(nn::L1Size, nn::L2Size, nn::NumVariants);
    m_l2Weights->m_weightsRange = hiddenWeightRange;
    m_l2Weights->m_biasRange = hiddenBiasRange;
    m_l2Weights->Init(nn::L1Size);

    m_l3Weights = std::make_shared<nn::WeightsStorage>(nn::L2Size, 1, nn::NumVariants);
    m_l3Weights->m_weightsRange = INT16_MAX / nn::OutputLayerWeightQuantizationScale;
    m_l3Weights->m_biasRange = hiddenBiasRange;
    m_l3Weights->Init(nn::L2Size);

    // Initialize CUDA network
    m_cudaNetwork.Init(m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights);

    // AdamW weight decay (QAT is enabled inside CudaNeuralNetwork::Init).
    m_cudaNetwork.SetWeightDecay(m_options.featureTransformerWeightDecay, cOutputSubnetWeightDecay);
    m_cudaNetwork.SetAdamBetas(m_options.beta1, m_options.beta2);
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

    outEntry.numWhiteFeatures = (uint8_t)numWhiteFeatures;
    outEntry.numBlackFeatures = (uint8_t)numBlackFeatures;
    for (uint32_t i = 0; i < numWhiteFeatures; ++i)
        outEntry.whiteFeatures[i] = whiteFeatures[i];
    for (uint32_t i = 0; i < numBlackFeatures; ++i)
        outEntry.blackFeatures[i] = blackFeatures[i];

    outEntry.variant = GetNetworkVariant(pos);
}

// King bucket pair of an entry, packed as (whiteBucket, blackBucket).
// PositionToFeaturesVector emits every feature as kingBucket * 12 * 64 + pieceIndex and each
// position always has at least the two kings, so the first feature carries the bucket.
INLINE static uint32_t GetKingBucketSortKey(const TrainingEntry& entry)
{
    const uint32_t whiteBucket = entry.whiteFeatures[0] / (12u * 64u);
    const uint32_t blackBucket = entry.blackFeatures[0] / (12u * 64u);
    ASSERT(whiteBucket < nn::NumKingBuckets && blackBucket < nn::NumKingBuckets);
    return whiteBucket * nn::NumKingBuckets + blackBucket;
}

// Group a batch's entries by king bucket pair (counting sort, stable).
// The feature transformer kernels assign 16 consecutive batch entries to a thread block, so
// grouping equal buckets confines a block's weight gathers and gradient atomics to a single
// bucket's 768-row window instead of scattering them over all NumNetworkInputs rows.
// The batch gradient is a sum, so reordering within a batch does not change the math. Sorting must
// stay inside one batch: sorting a whole iteration would make batches bucket-homogeneous and bias
// each batch's gradient toward one bucket.
static void SortBatchByKingBucket(TrainingEntry* entries, uint32_t count)
{
    constexpr uint32_t numKeys = nn::NumKingBuckets * nn::NumKingBuckets;

    static thread_local std::vector<TrainingEntry> scratch;
    scratch.resize(count);

    uint32_t offsets[numKeys] = { 0 };

    for (uint32_t i = 0; i < count; ++i)
        offsets[GetKingBucketSortKey(entries[i])]++;

    uint32_t sum = 0;
    for (uint32_t key = 0; key < numKeys; ++key)
    {
        const uint32_t keyCount = offsets[key];
        offsets[key] = sum;
        sum += keyCount;
    }

    for (uint32_t i = 0; i < count; ++i)
        scratch[offsets[GetKingBucketSortKey(entries[i])]++] = entries[i];

    memcpy(entries, scratch.data(), count * sizeof(TrainingEntry));
}

void CudaNetworkTrainer::GenerateTrainingEntry(std::mt19937& rng, TrainingEntry& outEntry, uint64_t kingBucketMask, float lambda, float bucketLeak)
{
    Position pos;
    PositionEntry entry;

    if (!m_dataLoader.FetchNextPosition(rng, entry, pos, kingBucketMask))
        return;

    // flip the board randomly in pawnless positions
    if (pos.Whites().pawns == 0 && pos.Blacks().pawns == 0)
    {
        if (std::uniform_int_distribution<>(0, 1)(rng) != 0)
            pos.MirrorVertically();
        if (std::uniform_int_distribution<>(0, 1)(rng) != 0)
            pos.FlipDiagonally();
    }

    // make game score more important for high move count
    const float wdlLambda = lambda * expf(-(float)pos.GetMoveCount() / 120.0f);

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

    PositionToTrainingEntry(pos, outEntry);
    outEntry.targetOutput = score;

    // training as if the position had one piece more or less only changes the bucket at a range edge
    if (bucketLeak > 0.0f && std::bernoulli_distribution(bucketLeak)(rng))
    {
        const int32_t shiftedPieces = (int32_t)pos.GetNumPiecesExcludingKing() + (std::bernoulli_distribution(0.5)(rng) ? 1 : -1);
        outEntry.variant = (uint8_t)std::min((uint32_t)std::max(shiftedPieces, 0) / 4u, nn::NumVariants - 1u);
    }
}

void CudaNetworkTrainer::GenerateTrainingSet(TrainingDataSet& outSet, TaskBuilder& builder, uint64_t kingBucketMask, float lambda, float bucketLeak, bool deterministic)
{
    if (deterministic)
    {
        builder.Task("GenerateSet", [this, &outSet, kingBucketMask, lambda, bucketLeak](const TaskContext&)
        {
            for (TrainingEntry& entry : outSet)
                GenerateTrainingEntry(m_deterministicRng, entry, kingBucketMask, lambda, bucketLeak);
        });
    }
    else
    {
        builder.ParallelFor("GenerateSet", static_cast<uint32_t>(outSet.size()),
            [this, &outSet, kingBucketMask, lambda, bucketLeak](const TaskContext& ctx, uint32_t index)
        {
            GenerateTrainingEntry(m_randomGenerators[ctx.threadId], outSet[index], kingBucketMask, lambda, bucketLeak);
        }, 0);
    }

    builder.Fence();

    const uint32_t batchSize = m_options.batchSize;
    const uint32_t numBatches = (static_cast<uint32_t>(outSet.size()) + batchSize - 1) / batchSize;
    builder.ParallelFor("SortSetByKingBucket", numBatches,
        [&outSet, batchSize](const TaskContext&, uint32_t batchIndex)
    {
        const uint32_t begin = batchIndex * batchSize;
        const uint32_t end = std::min(begin + batchSize, static_cast<uint32_t>(outSet.size()));
        SortBatchByKingBucket(outSet.data() + begin, end - begin);
    }, 0);
}

void CudaNetworkTrainer::RunCudaTrainingIteration(float learningRate)
{
    const uint32_t batchSize = m_options.batchSize;
    const uint32_t numBatches = cNumTrainingVectorsPerIteration / batchSize;
    m_cudaBatchData.batchSize = batchSize;

    m_cudaNetwork.BeginIterationTiming();

    m_cudaBatchData.lossSum.ClearAsync(m_cudaNetwork.GetStream().Get());

    // Prefetch the first batch (this copy cannot overlap anything; it is the only per-iteration
    // stall). Subsequent batches are prefetched while the previous batch's Adam updates run.
    m_cudaNetwork.CopyTrainingBatchAsync(m_cudaBatchData, m_trainingSet_Read.data(), batchSize);

    for (uint32_t b = 0; b < numBatches; ++b)
    {
        // Forward pass (waits for this batch's copy to complete)
        m_cudaNetwork.Forward(m_cudaBatchData);

        // Backward pass
        m_cudaNetwork.Backward(m_cudaBatchData, learningRate);

        // Prefetch the next batch's training vectors on the copy stream. It overlaps this batch's
        // Adam updates, which no longer read the training-vectors buffer.
        if (b + 1 < numBatches)
        {
            m_cudaNetwork.CopyTrainingBatchAsync(
                m_cudaBatchData,
                m_trainingSet_Read.data() + (b + 1) * batchSize,
                batchSize);
        }
    }

    m_lastIterationGpuTimeMs = m_cudaNetwork.EndIterationTimingMs();

    m_cudaNetwork.GetStream().Synchronize();

    {
        float squaredErrorSum = 0.0f;
        m_cudaBatchData.lossSum.CopyToHost(&squaredErrorSum, 1);
        m_lastTrainingRmse = sqrtf(squaredErrorSum / (float)(numBatches * batchSize));
    }

    // Copy weights from CUDA to host
    m_cudaNetwork.CopyWeightsToHost(m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights);
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

#ifdef USE_PACKED_NET_VALIDATION
static float EvalPackedNetwork(const TrainingEntry& entry, const nn::PackedNeuralNetwork& net)
{
    const int32_t packedNetworkOutput = net.Run(
        entry.whiteFeatures, entry.numWhiteFeatures,
        entry.blackFeatures, entry.numBlackFeatures,
        entry.variant);
    const float scaledPackedNetworkOutput = (float)packedNetworkOutput / (float)(nn::OutputScale * nn::WeightScale) * c_nnOutputToCentiPawns / 100.0f;
    return EvalToExpectedGameScore(scaledPackedNetworkOutput);
}
#endif // USE_PACKED_NET_VALIDATION

// Forward pass of one dense output-subnet layer on the float weights.
// Weights are input-major (weights[input * outputSize + output]), biases follow the matrix.
static void FloatDenseLayer(
    float* output, const float* input, const nn::Values& weights,
    uint32_t inputSize, uint32_t outputSize, bool applyCReLU)
{
    for (uint32_t o = 0; o < outputSize; ++o)
    {
        float sum = weights[inputSize * outputSize + o];
        for (uint32_t i = 0; i < inputSize; ++i)
            sum += input[i] * weights[i * outputSize + o];

        output[o] = applyCReLU ? std::clamp(sum, 0.0f, 1.0f) : sum;
    }
}

// Reference forward pass on the float (unquantized) weights the trainer actually optimizes.
// Comparing it against the packed network on the same positions tells a training problem apart
// from a packing / quantized-inference problem: only the latter moves the two numbers apart.
float CudaNetworkTrainer::EvalFloatNetwork(const TrainingEntry& entry) const
{
    constexpr uint32_t halfSize = nn::AccumulatorSize / 2;
    const nn::Values& ftWeights = m_featureTransformerWeights->m_variants.front().m_weights;

    float accumulators[2][nn::AccumulatorSize];
    for (uint32_t perspective = 0; perspective < 2; ++perspective)
    {
        const uint16_t* features = perspective == 0 ? entry.whiteFeatures : entry.blackFeatures;
        const uint32_t numFeatures = perspective == 0 ? entry.numWhiteFeatures : entry.numBlackFeatures;

        const float* biases = ftWeights.data() + nn::cuda::FeatureTransformerInputs * nn::AccumulatorSize;
        std::copy(biases, biases + nn::AccumulatorSize, accumulators[perspective]);

        for (uint32_t f = 0; f < numFeatures; ++f)
        {
            const uint32_t feature = features[f];
            const float* weights = ftWeights.data() + feature * nn::AccumulatorSize;
            for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
                accumulators[perspective][i] += weights[i];

#if USE_FACTORIZER
            const float* factorizer = ftWeights.data() +
                (nn::NumNetworkInputs + feature % nn::cuda::FactorizerInputs) * nn::AccumulatorSize;
            for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
                accumulators[perspective][i] += factorizer[i];
#endif // USE_FACTORIZER
        }
    }

    // pairwise activation, side to move first
    float l1Input[nn::L1InputSize];
    for (uint32_t perspective = 0; perspective < 2; ++perspective)
    {
        for (uint32_t i = 0; i < halfSize; ++i)
        {
            l1Input[perspective * halfSize + i] =
                std::clamp(accumulators[perspective][i], 0.0f, 1.0f) *
                std::clamp(accumulators[perspective][i + halfSize], 0.0f, 1.0f);
        }
    }

    float l2Input[nn::L1Size];
    FloatDenseLayer(l2Input, l1Input, m_l1Weights->m_variants[entry.variant].m_weights,
        nn::L1InputSize, nn::L1Size, true);

    float l3Input[nn::L2Size];
    FloatDenseLayer(l3Input, l2Input, m_l2Weights->m_variants[entry.variant].m_weights,
        nn::L1Size, nn::L2Size, true);

    float output = 0.0f;
    FloatDenseLayer(&output, l3Input, m_l3Weights->m_variants[entry.variant].m_weights,
        nn::L2Size, 1u, false);

    return EvalToExpectedGameScore(output * c_nnOutputToCentiPawns / 100.0f);
}

void CudaNetworkTrainer::Validate(const TaskContext& ctx)
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
        const float expectedValue = entry.targetOutput;

        ValidationStats& stats = threadData.stats;
#ifdef USE_EVAL_VALIDATION
        // TrainingEntry has no Position (only features); eval validation would require reconstructing Position from features.
#error "USE_EVAL_VALIDATION not supported in CudaNetworkTrainer: TrainingEntry has no Position"
#endif // USE_EVAL_VALIDATION
#ifdef USE_PACKED_NET_VALIDATION
        {
            const float nnPackedValue = EvalPackedNetwork(entry, *m_packedNet);
            const float error = expectedValue - nnPackedValue;
            stats.nnPackedErrorSum += (double)error * (double)error;

            if (i < cNumFloatReferenceVectors)
            {
                const float floatError = expectedValue - EvalFloatNetwork(entry);
                stats.nnPackedSubsetErrorSum += (double)error * (double)error;
                stats.floatErrorSum += (double)floatError * (double)floatError;
            }
        }
#endif // USE_PACKED_NET_VALIDATION
    });

    taskBuilder.Fence();

    taskBuilder.Task("AccumulateValidationStats", [this](const TaskContext&)
    {
        ValidationStats stats;
        for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
        {
            const ValidationStats& threadStats = m_validationPerThreadData[i].stats;

#ifdef USE_PACKED_NET_VALIDATION
            stats.nnPackedErrorSum += threadStats.nnPackedErrorSum;
            stats.nnPackedSubsetErrorSum += threadStats.nnPackedSubsetErrorSum;
            stats.floatErrorSum += threadStats.floatErrorSum;
#endif // USE_PACKED_NET_VALIDATION
        }

#ifdef USE_PACKED_NET_VALIDATION
        m_lastPackedNetRmse = (float)sqrt(stats.nnPackedErrorSum / cNumValidationVectorsPerIteration);
        m_lastPackedNetSubsetRmse = (float)sqrt(stats.nnPackedSubsetErrorSum / cNumFloatReferenceVectors);
        m_lastFloatNetSubsetRmse = (float)sqrt(stats.floatErrorSum / cNumFloatReferenceVectors);
#endif // USE_PACKED_NET_VALIDATION

        m_lastValidationMs = 1000.0f * (TimePoint::GetCurrent() - m_validationStartTime).ToSeconds();
    });
}

CudaNetworkTrainer::NetworkDetails CudaNetworkTrainer::CollectNetworkDetails() const
{
    NetworkDetails details{};

    const nn::WeightsStorage* layers[] = { m_featureTransformerWeights.get(), m_l1Weights.get(), m_l2Weights.get(), m_l3Weights.get() };
    static_assert(std::size(layers) == std::size(c_layerNames));
    for (size_t i = 0; i < std::size(layers); ++i)
        details.layerStats[i] = layers[i]->ComputeStats();

#ifdef USE_PACKED_NET_VALIDATION
    for (size_t i = 0; i < std::size(c_testPositions); ++i)
    {
        TrainingEntry entry;
        PositionToTrainingEntry(Position(c_testPositions[i]), entry);
        details.testPositionEvals[i] = ExpectedGameScoreToInternalEval(EvalPackedNetwork(entry, *m_packedNet));
    }
#endif // USE_PACKED_NET_VALIDATION

    return details;
}

void CudaNetworkTrainer::WriteNetworkDetails(const NetworkDetails& details, uint64_t numPositions)
{
    for (size_t i = 0; i < std::size(c_layerNames); ++i)
    {
        const nn::WeightsStorage::Stats& stats = details.layerStats[i];
        m_weightsFile << numPositions << '\t' << c_layerNames[i] << std::setprecision(6)
            << '\t' << stats.minWeight << '\t' << stats.maxWeight << '\t' << stats.weightAvg << '\t' << stats.weightStdDev
            << '\t' << stats.minBias << '\t' << stats.maxBias << '\t' << stats.biasAvg << '\t' << stats.biasStdDev << '\n';
    }
    m_weightsFile.flush();

#ifdef USE_PACKED_NET_VALIDATION
    for (size_t i = 0; i < std::size(c_testPositions); ++i)
        m_testPositionsFile << numPositions << '\t' << i << '\t' << details.testPositionEvals[i] << '\t' << c_testPositions[i] << '\n';
    m_testPositionsFile.flush();
#endif // USE_PACKED_NET_VALIDATION
}

void CudaNetworkTrainer::PrintNetworkDetails(const NetworkDetails& details, int indent) const
{
    char buf[256];
    std::string str;

    snprintf(buf, sizeof(buf), "%*s%-5s %9s %8s %9s %7s | %9s %8s %9s %7s\n", indent, "",
        "layer", "w min", "w max", "w avg", "w std", "b min", "b max", "b avg", "b std");
    str += buf;

    for (size_t i = 0; i < std::size(c_layerNames); ++i)
    {
        const nn::WeightsStorage::Stats& stats = details.layerStats[i];
        snprintf(buf, sizeof(buf), "%*s%-5s %9.4f %8.4f %9.5f %7.4f | %9.4f %8.4f %9.5f %7.4f\n", indent, "", c_layerNames[i],
            stats.minWeight, stats.maxWeight, stats.weightAvg, stats.weightStdDev,
            stats.minBias, stats.maxBias, stats.biasAvg, stats.biasStdDev);
        str += buf;
    }

#ifdef USE_PACKED_NET_VALIDATION
    snprintf(buf, sizeof(buf), "%*stest positions (packed net eval):\n", indent, "");
    str += buf;

    for (size_t i = 0; i < std::size(c_testPositions); ++i)
    {
        snprintf(buf, sizeof(buf), "%*s%6d  %s\n", indent, "", details.testPositionEvals[i], c_testPositions[i]);
        str += buf;
    }
#endif // USE_PACKED_NET_VALIDATION

    std::cout << str << std::flush;
}

void CudaNetworkTrainer::WriteRunSettings(uint64_t startPositions) const
{
    const std::string path = startPositions > 0 ?
        OutputPath("-settings-resume-" + std::to_string(startPositions / 1'000'000'000ull) + "B.txt") :
        OutputPath("-settings.txt");

    std::ofstream f(path);
    if (!f)
    {
        std::cout << "ERROR: Failed to write run settings: " << path << std::endl;
        return;
    }

    const std::time_t now = std::time(nullptr);

    f << "command="                    << m_options.commandLine << "\n";
    f << "started="                    << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "\n";
    f << "build="                      << CAISSA_VERSION << " " << __DATE__ << " " << __TIME__ << "\n";
    f << "device="                     << m_deviceName << "\n";
    f << "threads="                    << ThreadPool::GetInstance().GetNumThreads() << "\n";
    f << "\n";

    f << "name="                       << m_options.name << "\n";
    f << "net="                        << m_options.startNetPath << "\n";
    f << "resume="                     << m_options.resumePath << "\n";
    f << "restartSchedule="            << (m_options.restartSchedule ? "true" : "false") << "\n";
    f << "startPositions="             << startPositions << "\n";
    f << "iterations="                 << m_options.maxIterations << "\n";
    f << "seed="                       << m_options.seed << "\n";
    f << "LR="                         << m_options.startLearningRate << "\n";
    f << "endLR="                      << m_options.endLearningRate << "\n";
    f << "trainingLength="             << m_options.trainingLength << "\n";
    f << "startLambda="                << m_options.startLambda << "\n";
    f << "endLambda="                  << m_options.endLambda << "\n";
    f << "batchSize="                  << m_options.batchSize << "\n";
    f << "beta1="                      << m_options.beta1 << "\n";
    f << "beta2="                      << m_options.beta2 << "\n";
    f << "ftDecay="                    << m_options.featureTransformerWeightDecay << "\n";
    f << "outputSubnetDecay="          << cOutputSubnetWeightDecay << "\n";
    f << "bucketLeak="                 << m_options.bucketLeak << "\n";
    f << "freezeFeatureTransformer="   << (m_options.freezeFeatureTransformer ? "true" : "false") << "\n";
    f << "reviveDeadNeurons="          << (m_options.reviveDeadNeurons ? "true" : "false") << "\n";
    f << "warmupIterations="           << cWarmupTime << "\n";
    f << "positionsPerIteration="      << cNumTrainingVectorsPerIteration << "\n";
    f << "validationPositions="        << cNumValidationVectorsPerIteration << "\n";
    f << "checkpointInterval="         << cCheckpointInterval << "\n";
    f << "\n";

    f << "dataPath="                   << m_dataLoader.GetDataPath() << "\n";
    f << "dataFiles="                  << m_dataLoader.GetNumFiles() << "\n";
    f << "dataBytes="                  << m_dataLoader.GetTotalDataSize() << "\n";
    f << "decisiveSkip="               << (m_options.loaderOptions.decisiveSkip ? "true" : "false") << "\n";
    f << "pieceCountTarget="           << (m_options.loaderOptions.pieceCountTarget ? "true" : "false") << "\n";
    f << "hmcSkipFrom20="              << (m_options.loaderOptions.hmcSkipFrom20 ? "true" : "false") << "\n";

    if (m_options.loaderOptions.pieceCountTarget)
    {
        // keep probability per pair of piece counts, starting at 0-1 pieces
        f << "pieceCountKeepProb=";
        const TrainingDataLoader::PiecePairKeepProb& keepProb = m_dataLoader.GetPiecePairKeepProb();
        for (size_t i = 0; i < keepProb.size(); ++i)
            f << (i > 0 ? "," : "") << keepProb[i];
        f << "\n";
    }

    std::cout << "Run settings written to: " << path << std::endl;
}

// How a layer's weights are laid out in the packed network
enum class WeightLayout
{
    OutputMajor,    // weights[output * numInputs + input]
    InputMajor,     // weights[input * numOutputs + output]
    Grouped,        // nn::HiddenWeightIndex
};

static uint32_t PackedWeightIndex(WeightLayout layout, uint32_t input, uint32_t output, uint32_t numInputs, uint32_t numOutputs)
{
    switch (layout)
    {
    case WeightLayout::InputMajor:  return input * numOutputs + output;
    case WeightLayout::Grouped:     return nn::HiddenWeightIndex(input, output, numOutputs);
    default:                        return output * numInputs + input;
    }
}

template<typename WeightType, typename BiasType>
static void PackWeights(
    const nn::Values& weights, uint32_t numInputs, uint32_t numOutputs,
    WeightType* outWeights, BiasType* outBiases,
    float weightScale, float biasScale,
    float maxWeightRange, float maxBiasRange,
    WeightLayout layout)
{
    (void)maxWeightRange;
    (void)maxBiasRange;

    // weights
    for (uint32_t j = 0; j < numInputs; j++)
    {
        for (uint32_t i = 0; i < numOutputs; i++)
        {
            const float weight = weights[j * numOutputs + i];
            ASSERT(weight <= maxWeightRange);
            ASSERT(weight >= -maxWeightRange);

            const int32_t quantizedWeight = (int32_t)std::round(weight * weightScale);
            ASSERT(quantizedWeight <= std::numeric_limits<WeightType>::max());
            ASSERT(quantizedWeight >= std::numeric_limits<WeightType>::min());

            outWeights[PackedWeightIndex(layout, j, i, numInputs, numOutputs)] = (WeightType)quantizedWeight;
        }
    }

    // biases
    for (uint32_t i = 0; i < numOutputs; i++)
    {
        const float bias = weights[numInputs * numOutputs + i];
        ASSERT(bias <= maxBiasRange);
        ASSERT(bias >= -maxBiasRange);

        const int32_t quantizedBias = (int32_t)std::round(bias * biasScale);
        ASSERT(quantizedBias <= std::numeric_limits<BiasType>::max());
        ASSERT(quantizedBias >= std::numeric_limits<BiasType>::min());

        outBiases[i] = (BiasType)quantizedBias;
    }
}

template<typename WeightType, typename BiasType>
static void UnpackWeights(nn::Values& outWeights, uint32_t numInputs, uint32_t numOutputs, const WeightType* weights, const BiasType* biases, float weightScale, float biasScale, WeightLayout layout)
{
    // weights
    for (uint32_t j = 0; j < numInputs; j++)
    {
        for (uint32_t i = 0; i < numOutputs; i++)
        {
            outWeights[j * numOutputs + i] = (float)weights[PackedWeightIndex(layout, j, i, numInputs, numOutputs)] / weightScale;
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
        const nn::Values& storedWeights = m_featureTransformerWeights->m_variants.front().m_weights;

        // packed layout: [NumNetworkInputs * AccumulatorSize weights][AccumulatorSize biases]
        nn::Values weights((nn::NumNetworkInputs + 1) * nn::AccumulatorSize);
        std::copy(storedWeights.begin(), storedWeights.begin() + nn::NumNetworkInputs * nn::AccumulatorSize, weights.begin());
        std::copy(
            storedWeights.begin() + nn::cuda::FeatureTransformerInputs * nn::AccumulatorSize,
            storedWeights.begin() + (nn::cuda::FeatureTransformerInputs + 1) * nn::AccumulatorSize,
            weights.begin() + nn::NumNetworkInputs * nn::AccumulatorSize);

#if USE_FACTORIZER
        // fold the factorizer into every king bucket
        for (uint32_t feature = 0; feature < nn::NumNetworkInputs; ++feature)
        {
            const float* factorizer = storedWeights.data() + (nn::NumNetworkInputs + feature % nn::cuda::FactorizerInputs) * nn::AccumulatorSize;
            float* target = weights.data() + feature * nn::AccumulatorSize;
            for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
                target[i] += factorizer[i];
        }
#endif // USE_FACTORIZER

        PackWeights(
            weights,
            nn::NumNetworkInputs, nn::AccumulatorSize,
            const_cast<nn::FirstLayerWeightType*>(m_packedNet->accumulatorWeights),
            const_cast<nn::FirstLayerBiasType*>(m_packedNet->accumulatorBiases),
            nn::InputLayerWeightQuantizationScale, nn::InputLayerBiasQuantizationScale,
            m_featureTransformerWeights->m_weightsRange, m_featureTransformerWeights->m_biasRange,
            WeightLayout::InputMajor);
    }

    // output subnets: L1/L2 in the grouped layout the sparse inference reads, L3 as a plain row
    for (uint32_t variantIdx = 0; variantIdx < nn::NumVariants; ++variantIdx)
    {
        nn::PackedNeuralNetwork::OutputSubnetVariant& subnet = m_packedNet->outputSubnetVariants[variantIdx];

        PackWeights(
            m_l1Weights->m_variants[variantIdx].m_weights,
            nn::L1InputSize, nn::L1Size,
            subnet.l1Weights, subnet.l1Biases,
            nn::HiddenLayerWeightQuantizationScale, nn::HiddenLayerBiasQuantizationScale,
            m_l1Weights->m_weightsRange, m_l1Weights->m_biasRange,
            WeightLayout::Grouped);

        PackWeights(
            m_l2Weights->m_variants[variantIdx].m_weights,
            nn::L1Size, nn::L2Size,
            subnet.l2Weights, subnet.l2Biases,
            nn::HiddenLayerWeightQuantizationScale, nn::HiddenLayerBiasQuantizationScale,
            m_l2Weights->m_weightsRange, m_l2Weights->m_biasRange,
            WeightLayout::Grouped);

        PackWeights(
            m_l3Weights->m_variants[variantIdx].m_weights,
            nn::L2Size, 1u,
            subnet.l3Weights, &subnet.l3Bias,
            nn::OutputLayerWeightQuantizationScale, nn::OutputLayerBiasQuantizationScale,
            m_l3Weights->m_weightsRange, m_l3Weights->m_biasRange,
            WeightLayout::OutputMajor);
    }

    MTR_END("CudaNetworkTrainer", "PackNetwork");
    return true;
}

// Unpacks the feature transformer of a version 12 (single hidden layer) network. That format has
// no counterpart for the output subnet, so the caller must initialize it separately.
static bool UnpackNetworkV12(const char* path, nn::Values& featureTransformerWeights)
{
    constexpr uint32_t OldKingBuckets = 32;
    constexpr float OldActivationRangeScaling = 255;
    constexpr float OldInputLayerWeightQuantizationScale = OldActivationRangeScaling;
    constexpr float OldInputLayerBiasQuantizationScale = OldActivationRangeScaling;

    struct alignas(CACHELINE_SIZE) OldLastLayerVariant
    {
        nn::LastLayerWeightType weights[2 * nn::AccumulatorSize];
        nn::LastLayerBiasType bias;
        int32_t padding[15];
    };

    struct alignas(CACHELINE_SIZE) OldPackedNeuralNetwork
    {
        nn::PackedNeuralNetwork::Header header;
        nn::FirstLayerWeightType accumulatorWeights[768u * OldKingBuckets * nn::AccumulatorSize];
        nn::FirstLayerBiasType accumulatorBiases[nn::AccumulatorSize];
        OldLastLayerVariant lastLayerVariants[nn::NumVariants];
    };

    FILE* file = fopen(path, "rb");
    if (!file)
    {
        std::cerr << "Failed to load neural network: " << "cannot open file" << std::endl;
        return false;
    }

    auto oldPackedNet = std::make_unique<OldPackedNeuralNetwork>();
    if (1 != fread(oldPackedNet.get(), sizeof(OldPackedNeuralNetwork), 1, file))
    {
        fclose(file);
        std::cerr << "Failed to load neural network: " << "cannot read header" << std::endl;
        return false;
    }

    // feature transformer
    {
        nn::Values& weights = featureTransformerWeights;

        UnpackWeights(
            weights,
            OldKingBuckets * 768,
            nn::AccumulatorSize,
            oldPackedNet->accumulatorWeights,
            oldPackedNet->accumulatorBiases,
            OldInputLayerWeightQuantizationScale,
            OldInputLayerBiasQuantizationScale,
            WeightLayout::InputMajor);

#if USE_FACTORIZER
        // a packed net has the factorizer already folded in: move the biases behind the (zeroed)
        // factorizer rows
        std::copy(
            weights.begin() + nn::NumNetworkInputs * nn::AccumulatorSize,
            weights.begin() + (nn::NumNetworkInputs + 1) * nn::AccumulatorSize,
            weights.begin() + nn::cuda::FeatureTransformerInputs * nn::AccumulatorSize);
        std::fill(
            weights.begin() + nn::NumNetworkInputs * nn::AccumulatorSize,
            weights.begin() + nn::cuda::FeatureTransformerInputs * nn::AccumulatorSize,
            0.0f);
#endif // USE_FACTORIZER
    }

    fclose(file);
    return true;
}

bool CudaNetworkTrainer::UnpackNetwork(const char* path, bool& outHasOutputSubnet)
{
    nn::PackedNeuralNetwork::Header header{};
    {
        FILE* file = fopen(path, "rb");
        if (!file)
        {
            std::cerr << "Failed to load neural network: cannot open " << path << std::endl;
            return false;
        }
        const bool read = fread(&header, sizeof(header), 1, file) == 1;
        fclose(file);

        if (!read || header.magic != nn::MagicNumber)
        {
            std::cerr << "Failed to load neural network: " << path << " is not a network file" << std::endl;
            return false;
        }
    }

    if (header.version == 12)
    {
        outHasOutputSubnet = false;
        return UnpackNetworkV12(path, m_featureTransformerWeights->m_variants.front().m_weights);
    }

    // loads the current version and upgrades version 13 in place
    auto packedNet = std::make_unique<nn::PackedNeuralNetwork>();
    if (!packedNet->LoadFromFile(path))
        return false;

    // feature transformer
    {
        nn::Values& weights = m_featureTransformerWeights->m_variants.front().m_weights;

        UnpackWeights(
            weights,
            nn::NumNetworkInputs,
            nn::AccumulatorSize,
            packedNet->accumulatorWeights,
            packedNet->accumulatorBiases,
            nn::InputLayerWeightQuantizationScale,
            nn::InputLayerBiasQuantizationScale,
            WeightLayout::InputMajor);

#if USE_FACTORIZER
        // a packed net has the factorizer already folded in: move the biases behind the (zeroed)
        // factorizer rows
        std::copy(
            weights.begin() + nn::NumNetworkInputs * nn::AccumulatorSize,
            weights.begin() + (nn::NumNetworkInputs + 1) * nn::AccumulatorSize,
            weights.begin() + nn::cuda::FeatureTransformerInputs * nn::AccumulatorSize);
        std::fill(
            weights.begin() + nn::NumNetworkInputs * nn::AccumulatorSize,
            weights.begin() + nn::cuda::FeatureTransformerInputs * nn::AccumulatorSize,
            0.0f);
#endif // USE_FACTORIZER
    }

    // output subnets
    for (uint32_t variantIdx = 0; variantIdx < nn::NumVariants; ++variantIdx)
    {
        const nn::PackedNeuralNetwork::OutputSubnetVariant& subnet = packedNet->outputSubnetVariants[variantIdx];

        UnpackWeights(
            m_l1Weights->m_variants[variantIdx].m_weights,
            nn::L1InputSize, nn::L1Size,
            subnet.l1Weights, subnet.l1Biases,
            nn::HiddenLayerWeightQuantizationScale, nn::HiddenLayerBiasQuantizationScale,
            WeightLayout::Grouped);

        UnpackWeights(
            m_l2Weights->m_variants[variantIdx].m_weights,
            nn::L1Size, nn::L2Size,
            subnet.l2Weights, subnet.l2Biases,
            nn::HiddenLayerWeightQuantizationScale, nn::HiddenLayerBiasQuantizationScale,
            WeightLayout::Grouped);

        UnpackWeights(
            m_l3Weights->m_variants[variantIdx].m_weights,
            nn::L2Size, 1u,
            subnet.l3Weights, &subnet.l3Bias,
            nn::OutputLayerWeightQuantizationScale, nn::OutputLayerBiasQuantizationScale,
            WeightLayout::OutputMajor);
    }

    outHasOutputSubnet = true;
    return true;
}

// if non-zero, overrides the learning rate scheduler (for tweaking under a debugger)
static volatile float g_learningRateScale = 0.0f;

static float GetScheduledLearningRate(const Options& options, float trainingProgress)
{
    constexpr float pi = 3.14159265358979323846f;
    return options.endLearningRate + 0.5f * (options.startLearningRate - options.endLearningRate) * (1.0f + cosf(pi * trainingProgress));
}

// A hidden neuron is dead when it can never activate (zero incoming weights and a non-positive bias) or when the
// next layer ignores it (zero outgoing weights). Either way it gets no gradient, so both sides are re-seeded.
void CudaNetworkTrainer::ReviveDeadNeurons()
{
    std::mt19937 gen(m_options.seed + 1000);

    const auto revive = [&gen](nn::WeightsStorage& layer, nn::WeightsStorage& nextLayer, float nextWeightScale, const char* name)
    {
        const uint32_t numInputs = layer.m_inputSize;
        const uint32_t numOutputs = layer.m_outputSize;
        const uint32_t nextNumOutputs = nextLayer.m_outputSize;
        std::normal_distribution<float> inputDist(0.0f, sqrtf(2.0f / (float)numInputs));
        std::normal_distribution<float> outputDist(0.0f, sqrtf(2.0f / (float)numOutputs));

        uint32_t numRevived = 0;
        for (size_t variantIdx = 0; variantIdx < layer.m_variants.size(); ++variantIdx)
        {
            nn::Values& weights = layer.m_variants[variantIdx].m_weights;
            nn::Values& nextWeights = nextLayer.m_variants[variantIdx].m_weights;

            for (uint32_t output = 0; output < numOutputs; ++output)
            {
                bool neverActive = std::round(weights[numInputs * numOutputs + output] * nn::HiddenLayerBiasQuantizationScale) <= 0.0f;
                for (uint32_t input = 0; neverActive && input < numInputs; ++input)
                    neverActive = std::round(weights[input * numOutputs + output] * nn::HiddenLayerWeightQuantizationScale) == 0.0f;

                bool ignored = true;
                for (uint32_t nextOutput = 0; ignored && nextOutput < nextNumOutputs; ++nextOutput)
                    ignored = std::round(nextWeights[output * nextNumOutputs + nextOutput] * nextWeightScale) == 0.0f;

                if (!neverActive && !ignored)
                    continue;

                for (uint32_t input = 0; input < numInputs; ++input)
                    weights[input * numOutputs + output] = std::clamp(inputDist(gen), -layer.m_weightsRange, layer.m_weightsRange);
                weights[numInputs * numOutputs + output] = 0.0f;

                // outgoing weights must survive quantization, otherwise the neuron is ignored again
                for (uint32_t nextOutput = 0; nextOutput < nextNumOutputs; ++nextOutput)
                {
                    float w;
                    do
                        w = std::clamp(outputDist(gen), -nextLayer.m_weightsRange, nextLayer.m_weightsRange);
                    while (std::round(w * nextWeightScale) == 0.0f);
                    nextWeights[output * nextNumOutputs + nextOutput] = w;
                }

                numRevived++;
            }
        }

        std::cout << "Revived dead " << name << " neurons: " << numRevived << std::endl;
    };

    revive(*m_l1Weights, *m_l2Weights, nn::HiddenLayerWeightQuantizationScale, "L1");
    revive(*m_l2Weights, *m_l3Weights, nn::OutputLayerWeightQuantizationScale, "L2");
}

bool CudaNetworkTrainer::Train()
{
    InitNetwork();

    const bool resuming = !m_options.resumePath.empty();
    const bool fromScratch = m_options.startNetPath.empty() && !resuming;
    std::cout << "Learning rate: " << m_options.startLearningRate << " -> " << m_options.endLearningRate << std::endl;
    std::cout << "Lambda: " << m_options.startLambda << " -> " << m_options.endLambda << std::endl;
    std::cout << "Training length: " << m_options.trainingLength << "B positions" << std::endl;
    std::cout << "Batch size: " << m_options.batchSize << std::endl;
    std::cout << "AdamW: beta1 " << m_options.beta1 << ", beta2 " << m_options.beta2
              << ", FT weight decay " << m_options.featureTransformerWeightDecay << std::endl;

    if (resuming)
    {
        uint64_t positionsPassed = 0;
        if (!m_cudaNetwork.LoadCheckpoint(m_options.resumePath.c_str(), positionsPassed))
            return false;

        m_numTrainingVectorsPassed = m_options.restartSchedule ? 0 : positionsPassed;
        m_cudaNetwork.CopyWeightsToHost(m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights);

        std::cout << "Resuming from checkpoint: " << m_options.resumePath
            << " at " << std::setprecision(4) << positionsPassed / 1.0e9f << "B positions" << std::endl;
        if (m_options.restartSchedule)
            std::cout << "Restarting the schedule from zero positions" << std::endl;
    }
    else if (fromScratch)
    {
        std::cout << "Training from scratch" << std::endl;
        m_cudaNetwork.InitRandomWeights(m_options.seed);
        m_cudaNetwork.CopyWeightsToHost(m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights);
    }
    else
    {
        bool hasOutputSubnet = false;
        if (!UnpackNetwork(m_options.startNetPath.c_str(), hasOutputSubnet))
            return false;

        std::cout << "Starting from net: " << m_options.startNetPath
            << (hasOutputSubnet ? "" : " (feature transformer only)") << std::endl;

        m_cudaNetwork.CopyWeightsFromHost(m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights);

        if (!hasOutputSubnet)
        {
            m_cudaNetwork.InitRandomOutputSubnetWeights(m_options.seed);
            m_cudaNetwork.CopyWeightsToHost(m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights);
        }
    }

    if (m_options.reviveDeadNeurons)
    {
        ReviveDeadNeurons();
        m_cudaNetwork.CopyWeightsFromHost(m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights);
    }

    if (m_options.freezeFeatureTransformer)
    {
        m_cudaNetwork.SetFeatureTransformerFrozen(true);
        std::cout << "Feature transformer frozen" << std::endl;
    }

    if (m_options.bucketLeak > 0.0f)
        std::cout << "Bucket leak: " << m_options.bucketLeak << std::endl;

    std::cout << "Loader: decisiveSkip " << (m_options.loaderOptions.decisiveSkip ? "on" : "off")
        << ", pieceCountTarget " << (m_options.loaderOptions.pieceCountTarget ? "on" : "off")
        << ", hmcSkipFrom20 " << (m_options.loaderOptions.hmcSkipFrom20 ? "on" : "off") << std::endl;

    if (!m_dataLoader.Init(m_randomGenerators[0], m_options.loaderOptions))
    {
        std::cout << "ERROR: Failed to initialize data loader" << std::endl;
        return false;
    }

    WriteRunSettings(m_numTrainingVectorsPassed);

    const float validationLambda = 1.0f;

    uint64_t kingBucketMask = UINT64_MAX;

    // resuming keeps the milestones already written on the previous run
    uint64_t lastCheckpoint = m_numTrainingVectorsPassed / cCheckpointInterval;
    uint64_t lastDetails = m_numTrainingVectorsPassed / cDetailsInterval;

    // initial training set generation
    {
        Waitable waitable;
        {
            TaskBuilder taskBuilder{ waitable };
            GenerateTrainingSet(m_validationSet, taskBuilder, kingBucketMask, validationLambda, 0.0f, true);
        }
        waitable.Wait();
    }

    const TimePoint startTime = TimePoint::GetCurrent();

    // progress since the last report
    TimePoint lastReportTime = startTime;
    uint64_t lastReportPositions = m_numTrainingVectorsPassed;
    uint32_t numReportIterations = 0;
    float gpuTimeMsSum = 0.0f;
    float trainTaskMsSum = 0.0f;
    float loaderMsSum = 0.0f;
    float validationMsSum = 0.0f;
    double trainingMseSum = 0.0;

    for (size_t iteration = 0; iteration < m_options.maxIterations; ++iteration)
    {
        const bool useWarmup = !fromScratch && (!resuming || m_options.restartSchedule) && cWarmupTime > 0.0f;
        const float warmup = useWarmup ? (iteration < cWarmupTime ? (float)(iteration + 1) / cWarmupTime : 1.0f) : 1.0f;
        const float t = std::min(1.0f, (float)((double)m_numTrainingVectorsPassed * 1.0e-9 / (double)m_options.trainingLength));
        const float learningRate = (g_learningRateScale != 0.0f) ? g_learningRateScale : warmup * GetScheduledLearningRate(m_options, t);
        const float lambda = m_options.startLambda + (m_options.endLambda - m_options.startLambda) * t;

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder{ waitable };

            // skip training in the first iteration, as the data is not ready yet
            if (iteration > 0)
            {
                taskBuilder.Task("CudaTrain", [this, learningRate](const TaskContext&)
                {
                    const TimePoint taskStartTime = TimePoint::GetCurrent();
                    RunCudaTrainingIteration(learningRate);
                    m_lastTrainTaskMs = 1000.0f * (TimePoint::GetCurrent() - taskStartTime).ToSeconds();
                });
            }

            if (iteration > 1)
            {
                taskBuilder.Task("Validate", [this](const TaskContext& ctx)
                {
                    m_validationStartTime = TimePoint::GetCurrent();
#ifdef USE_PACKED_NET_VALIDATION
                    PackNetwork();
#endif // USE_PACKED_NET_VALIDATION
                    Validate(ctx);
                });
            }

            taskBuilder.Task("GenerateTrainingSet", [this, kingBucketMask, lambda](const TaskContext& ctx)
            {
                const TimePoint taskStartTime = TimePoint::GetCurrent();
                TaskBuilder taskBuilder{ ctx };
                GenerateTrainingSet(m_trainingSet_Write, taskBuilder, kingBucketMask, lambda, m_options.bucketLeak);
                taskBuilder.Fence();
                taskBuilder.Task("GenerateTrainingSetDone", [this, taskStartTime](const TaskContext&)
                {
                    m_lastLoaderMs = 1000.0f * (TimePoint::GetCurrent() - taskStartTime).ToSeconds();
                });
            });
        }
        waitable.Wait();

        // swap read and write buffers
        std::swap(m_trainingSet_Write, m_trainingSet_Read);

        m_numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;
        const uint64_t numPositions = m_numTrainingVectorsPassed;

        const TimePoint now = TimePoint::GetCurrent();
        const float elapsedSeconds = (now - startTime).ToSeconds();
        const std::string timeTag = "[" + FormatDuration(elapsedSeconds) + "]";
        const int indent = (int)timeTag.size() + 1;

        // from the third iteration on, training, data loading and validation all run concurrently
        if (iteration > 1)
        {
            numReportIterations++;
            gpuTimeMsSum += m_lastIterationGpuTimeMs;
            trainTaskMsSum += m_lastTrainTaskMs;
            loaderMsSum += m_lastLoaderMs;
            validationMsSum += m_lastValidationMs;
            trainingMseSum += (double)m_lastTrainingRmse * (double)m_lastTrainingRmse;

            m_progressFile << iteration << '\t' << numPositions << '\t' << std::setprecision(6) << elapsedSeconds
                << '\t' << learningRate << '\t' << lambda << '\t' << m_lastTrainingRmse
#ifdef USE_PACKED_NET_VALIDATION
                << '\t' << m_lastPackedNetRmse << '\t' << m_lastFloatNetSubsetRmse << '\t' << m_lastPackedNetSubsetRmse
#else
                << "\t\t\t"
#endif // USE_PACKED_NET_VALIDATION
                << '\t' << m_lastTrainTaskMs << '\t' << m_lastIterationGpuTimeMs << '\t' << m_lastLoaderMs << '\t' << m_lastValidationMs << '\n';
        }

        const float secondsSinceReport = (now - lastReportTime).ToSeconds();
        if (secondsSinceReport >= cReportInterval && numReportIterations > 0)
        {
            const uint64_t totalPositions = m_options.trainingLength * 1'000'000'000ull;
            const double positionsPerSecond = (double)(numPositions - lastReportPositions) / secondsSinceReport;
            const auto taskPositionsPerSecond = [&](float msSum) { return msSum > 0.0f ? 1000.0 * numReportIterations * cNumTrainingVectorsPerIteration / msSum : 0.0; };
            const std::string eta = numPositions < totalPositions ? FormatDuration((double)(totalPositions - numPositions) / positionsPerSecond) : "done";

            char buf[256];
            std::string str;

            snprintf(buf, sizeof(buf), "%s %.1fB/%lluB (%.1f%%) ETA %s | %.2fM pos/s | loss train %.5f",
                timeTag.c_str(), numPositions * 1.0e-9, (unsigned long long)m_options.trainingLength, 100.0 * numPositions / totalPositions,
                eta.c_str(), positionsPerSecond * 1.0e-6, sqrt(trainingMseSum / numReportIterations));
            str += buf;
#ifdef USE_PACKED_NET_VALIDATION
            snprintf(buf, sizeof(buf), " val %.5f", m_lastPackedNetRmse);
            str += buf;
#endif // USE_PACKED_NET_VALIDATION

            snprintf(buf, sizeof(buf), "\n%*sLR %.3g lambda %.2f", indent, "", learningRate, lambda);
            str += buf;
#ifdef USE_PACKED_NET_VALIDATION
            snprintf(buf, sizeof(buf), " | float vs pnn %.5f / %.5f", m_lastFloatNetSubsetRmse, m_lastPackedNetSubsetRmse);
            str += buf;
#endif // USE_PACKED_NET_VALIDATION
            snprintf(buf, sizeof(buf), " | iteration %zu\n", iteration);
            str += buf;

            const char* limit = "training";
            if (loaderMsSum > trainTaskMsSum && loaderMsSum >= validationMsSum)
                limit = "data loading";
            else if (validationMsSum > trainTaskMsSum)
                limit = "validation";

            snprintf(buf, sizeof(buf), "%*sspeed M pos/s: GPU %.2f, training %.2f, data loading %.2f, validation %.2f | limited by %s\n", indent, "",
                taskPositionsPerSecond(gpuTimeMsSum) * 1.0e-6, taskPositionsPerSecond(trainTaskMsSum) * 1.0e-6,
                taskPositionsPerSecond(loaderMsSum) * 1.0e-6, taskPositionsPerSecond(validationMsSum) * 1.0e-6, limit);
            str += buf;

            std::cout << str << std::flush;
            m_progressFile.flush();

            lastReportTime = now;
            lastReportPositions = numPositions;
            numReportIterations = 0;
            gpuTimeMsSum = 0.0f;
            trainTaskMsSum = 0.0f;
            loaderMsSum = 0.0f;
            validationMsSum = 0.0f;
            trainingMseSum = 0.0;
        }

#ifdef USE_PACKED_NET_VALIDATION
        if (iteration % 50 == 2)
        {
            m_packedNet->SaveToFile(OutputPath(".pnn").c_str());
        }
#endif // USE_PACKED_NET_VALIDATION

        const uint64_t details = numPositions / cDetailsInterval;
        const uint64_t checkpoint = numPositions / cCheckpointInterval;
        const bool writeDetails = details > lastDetails && iteration > 1;
        const bool saveCheckpoint = checkpoint > lastCheckpoint;

        NetworkDetails networkDetails{};
        if (writeDetails || saveCheckpoint)
            networkDetails = CollectNetworkDetails();

        if (writeDetails)
        {
            lastDetails = details;
            WriteNetworkDetails(networkDetails, numPositions);
        }

        // keep a packed net and the full training state at every checkpoint interval
        if (saveCheckpoint)
        {
            lastCheckpoint = checkpoint;

            // labelled by the position count reached, matching the eval-<version>-<positions>B convention
            const std::string suffix = "-" + std::to_string(numPositions / 1'000'000'000ull) + "B";
#ifdef USE_PACKED_NET_VALIDATION
            m_packedNet->SaveToFile(OutputPath(suffix + ".pnn").c_str());
#endif // USE_PACKED_NET_VALIDATION
            m_cudaNetwork.SaveCheckpoint(OutputPath(suffix + ".ckpt").c_str(), numPositions);

            std::cout << timeTag << " Saved checkpoint: " << OutputPath(suffix) << ".{pnn,ckpt}" << std::endl;
            PrintNetworkDetails(networkDetails, indent);
        }
    }

    return true;
}

bool TrainCudaNetwork(const std::vector<std::string>& args)
{
    Options options;

    options.commandLine = "trainCudaNetwork";
    for (const std::string& arg : args)
        options.commandLine += " " + arg;
    for (size_t i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--restartSchedule")
            options.restartSchedule = true;
        else if (args[i] == "--freezeFeatureTransformer")
            options.freezeFeatureTransformer = true;
        else if (args[i] == "--reviveDeadNeurons")
            options.reviveDeadNeurons = true;
        else if (args[i] == "--decisiveSkip")
            options.loaderOptions.decisiveSkip = true;
        else if (args[i] == "--pieceCountTarget")
            options.loaderOptions.pieceCountTarget = true;
        else if (args[i] == "--hmcSkipFrom20")
            options.loaderOptions.hmcSkipFrom20 = true;
        else if (i + 1 >= args.size())
        {
            std::cerr << "Missing value for option: " << args[i] << std::endl;
            return false;
        }
        // every remaining option takes a value
        else if (args[i] == "--net")
            options.startNetPath = args[++i];
        else if (args[i] == "--iterations")
            options.maxIterations = std::stoull(args[++i]);
        else if (args[i] == "--seed")
            options.seed = (uint32_t)std::stoul(args[++i]);
        else if (args[i] == "--LR")
            options.startLearningRate = std::stof(args[++i]);
        else if (args[i] == "--endLR")
            options.endLearningRate = std::stof(args[++i]);
        else if (args[i] == "--startLambda")
            options.startLambda = std::stof(args[++i]);
        else if (args[i] == "--endLambda")
            options.endLambda = std::stof(args[++i]);
        else if (args[i] == "--name")
            options.name = args[++i];
        else if (args[i] == "--resume")
            options.resumePath = args[++i];
        else if (args[i] == "--trainingLength")
            options.trainingLength = std::stoull(args[++i]);
        else if (args[i] == "--bucketLeak")
            options.bucketLeak = std::stof(args[++i]);
        else if (args[i] == "--batchSize")
            options.batchSize = (uint32_t)std::stoul(args[++i]);
        else if (args[i] == "--beta1")
            options.beta1 = std::stof(args[++i]);
        else if (args[i] == "--beta2")
            options.beta2 = std::stof(args[++i]);
        else if (args[i] == "--ftDecay")
            options.featureTransformerWeightDecay = std::stof(args[++i]);
        else
        {
            std::cerr << "Unknown option: " << args[i] << std::endl;
            return false;
        }
    }

    if (options.batchSize == 0 || cNumTrainingVectorsPerIteration % options.batchSize != 0)
    {
        std::cerr << "Batch size must divide " << cNumTrainingVectorsPerIteration << std::endl;
        return false;
    }
    std::cout << "Seed: " << options.seed << std::endl;

    CudaNetworkTrainer trainer(options);
    return trainer.Train();
}
