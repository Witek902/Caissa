#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"
#include "TrainingEntry.hpp"
#include "cudaTrainer/CudaNetwork.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/NeuralNetworkEvaluator.hpp"
#include "../backend/ThreatInputs.hpp"
#include "../backend/Time.hpp"
#include "../backend/Waitable.hpp"

#include <ctime>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>


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

// Validation needs a weight download from the GPU. One download is in flight at a time, and a validation
// finishes before the next one starts.
static constexpr size_t cValidationInterval = 8;
static_assert(cValidationInterval >= 2, "a validation must finish before the next weight download starts");

// the packed net is saved every few validations
static constexpr uint32_t cPackedNetSaveInterval = 6;

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

static const char* c_layerNames[] = { "FT", "TI", "L1", "L2", "L3" };

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
    "8/2p5/pp6/4P1p1/8/2k5/2P5/3K4 w - - 0 1", // should be 1
    "1Q6/5p2/4k3/6p1/1K6/8/4q3/8 w - - 0 1", // should be 1
    "8/8/k1K5/3Q4/8/8/4q2q/8 w - - 0 1", // should be 1
    "rnbqk1nr/3p1pbp/p1pPp1p1/PpP5/1P6/8/4PPPP/1NBQKBNR w kq - 1 9", // should be 1?
    "rn1qkbnr/pbp1p3/1p1pPp1p/5PpP/6P1/8/PPPP4/RNBQKBN1 w Qkq - 1 9", // should be 1?
    // recaptures: the best move recaptures on the square just captured on
    "rn1qr1k1/ppp2ppp/5n2/3pbb2/N7/2PN1PP1/PP1PB2P/R1BQK2R w KQ - 0 12", // should be 0, Nxe5, bishop, opening
    "3r2k1/2br2p1/5q1p/1ppPpp2/2P5/1P2BnP1/5P1P/R2QR2K w - - 0 32", // should be 0, Qxf3, knight, middlegame
    "r2q1rk1/pp3ppp/8/1N2P3/PP6/1Q1p4/R2bbPPP/4R1K1 w - - 0 22", // should be 0, Rxd2, bishop, middlegame
    "8/1r3pk1/2nqpn1p/6p1/p2p4/P2B1PP1/1Q3PK1/4R3 w - - 0 35", // should be 0, Qxb7, rook
    "8/4n3/4k1p1/8/7P/4q1NK/5P2/8 w - - 0 65", // should be 0, fxe3, queen, into a minor-piece ending
    "3rk3/pp2p3/6p1/3P4/7p/3R2P1/PP6/6K1 w - - 0 32", // should be 0, gxh4, pawn, rook ending
    "8/8/4k3/2n2p1p/1P5P/P2R4/6r1/2K5 w - - 0 56", // should be 0, bxc5, knight, rook ending
};

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

    // cosine learning rate decay: starts at startLearningRate and reaches endLearningRate after the
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

    // add the shared factorizer weights to the piece weights; when off the factorizer rows are unused
    bool useFactorizer = true;

    // re-seed output subnet neurons that can no longer learn, see ReviveDeadNeurons
    bool reviveDeadNeurons = false;

    // probability that a training position picks its bucket as if it had one piece more or one piece
    // less, so positions at a bucket range edge also train the neighbouring subnet
    float bucketLeak = 0.0f;

    // output subnet variants (bit per variant) that keep their weights: positions that would train
    // them are drawn again, so with the feature transformer frozen they get no gradient at all
    uint32_t frozenVariantsMask = 0;

    // blend of the training targets: 0 = pure game result, 1 = evaluation (decaying toward the
    // game result with move count). The validation set always uses 1.
    float startLambda = 0.0f;   // at the beginning of training
    float endLambda = 0.0f;     // at the end of training

    // positions per optimizer step; must divide cNumTrainingVectorsPerIteration
    uint32_t batchSize = 32 * 1024;

    TrainingDataLoader::Options loaderOptions;

    // training data directories; empty = data/trainingData
    std::vector<std::string> dataPaths;

    // AdamW
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float featureTransformerWeightDecay = 0.0025f; // applied to weights only, not biases
};

// Training positions split into batches. Each batch owns TrainingThreatFeatureBudget threat feature slots
// per position, filled from the start, so a batch's threat features are uploaded in one piece.
struct TrainingSet
{
    std::vector<TrainingEntry> entries;
    std::vector<uint16_t> threatFeatures;
    std::vector<uint32_t> numBatchThreatFeatures;
    uint32_t batchSize = 0;

    // the queued iteration whose batch uploads read this set, SIZE_MAX if none
    size_t uploadingIteration = SIZE_MAX;

    void Allocate(uint32_t numEntries, uint32_t newBatchSize)
    {
        batchSize = newBatchSize;
        entries.resize(numEntries);
        numBatchThreatFeatures.resize(NumBatches());
        threatFeatures.resize((size_t)NumBatches() * batchSize * TrainingThreatFeatureBudget);
    }

    uint32_t NumBatches() const
    {
        return (static_cast<uint32_t>(entries.size()) + batchSize - 1) / batchSize;
    }

    uint32_t BatchBegin(uint32_t batchIndex) const { return batchIndex * batchSize; }
    uint32_t BatchEnd(uint32_t batchIndex) const { return std::min(BatchBegin(batchIndex) + batchSize, static_cast<uint32_t>(entries.size())); }

    uint16_t* BatchThreatFeatures(uint32_t batchIndex)
    {
        return threatFeatures.data() + (size_t)batchIndex * batchSize * TrainingThreatFeatureBudget;
    }

    const uint16_t* BatchThreatFeatures(uint32_t batchIndex) const
    {
        return threatFeatures.data() + (size_t)batchIndex * batchSize * TrainingThreatFeatureBudget;
    }

    // white threat features of an entry, followed by the black ones
    const uint16_t* EntryThreatFeatures(uint32_t index) const
    {
        return BatchThreatFeatures(index / batchSize) + entries[index].threatFeaturesOffset;
    }
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
            "\titeration_ms\tgpu_ms\tloader_ms\tvalidation_ms");
        OpenTsvFile(m_weightsFile, "-weights.tsv", append,
            "positions\tlayer\tw_min\tw_max\tw_avg\tw_std\tb_min\tb_max\tb_avg\tb_std");
        OpenTsvFile(m_testPositionsFile, "-test-positions.tsv", append,
            "positions\tindex\teval\tfen");

        m_packedNet = std::make_unique<nn::PackedNeuralNetwork>();

        m_validationSet.Allocate(cNumValidationVectorsPerIteration, options.batchSize);
        m_trainingSet_Read.Allocate(cNumTrainingVectorsPerIteration, options.batchSize);
        m_trainingSet_Write.Allocate(cNumTrainingVectorsPerIteration, options.batchSize);
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
        // They are only ever std::swap'd, never reallocated after this point.
        for (TrainingSet* set : { &m_trainingSet_Read, &m_trainingSet_Write })
        {
            CUDA_CHECK(cudaHostRegister(set->entries.data(), set->entries.size() * sizeof(TrainingEntry), cudaHostRegisterDefault));
            CUDA_CHECK(cudaHostRegister(set->threatFeatures.data(), set->threatFeatures.size() * sizeof(uint16_t), cudaHostRegisterDefault));
        }
    }

    ~CudaNetworkTrainer()
    {
        for (TrainingSet* set : { &m_trainingSet_Read, &m_trainingSet_Write })
        {
            cudaHostUnregister(set->entries.data());
            cudaHostUnregister(set->threatFeatures.data());
        }
    }

    void InitNetwork();

    bool Train();

private:

    struct ValidationStats
    {
        double nnPackedErrorSum = 0.0f;
        // same positions as floatErrorSum, so the two are directly comparable
        double nnPackedSubsetErrorSum = 0.0f;
        double floatErrorSum = 0.0f;
    };

    struct alignas(CACHELINE_SIZE) ValidationPerThreadData
    {
        ValidationStats stats;
        uint8_t __padding[CACHELINE_SIZE];
    };

    TrainingDataLoader m_dataLoader;

    nn::WeightsStoragePtr m_featureTransformerWeights;
    nn::WeightsStoragePtr m_threatWeights;
    nn::WeightsStoragePtr m_l1Weights;
    nn::WeightsStoragePtr m_l2Weights;
    nn::WeightsStoragePtr m_l3Weights;

    nn::cuda::CudaNeuralNetwork m_cudaNetwork;
    nn::cuda::CudaBatchData m_cudaBatchData;
    std::unique_ptr<nn::PackedNeuralNetwork> m_packedNet;

    TrainingSet m_validationSet;
    TrainingSet m_trainingSet_Write; // training set written to by IO
    TrainingSet m_trainingSet_Read; // training set used by trainer
    std::vector<ValidationPerThreadData> m_validationPerThreadData;

    uint64_t m_numTrainingVectorsPassed = 0;

    // set while the validation task reads or writes the host weights
    bool m_hostWeightsInUse = false;

    // positions drawn again because they have more pieces than a training entry holds or could not be read
    std::atomic<uint64_t> m_numSkippedPositions = 0;

    // positions drawn again because their threat features did not fit in the rest of the batch's list
    std::atomic<uint64_t> m_numThreatListFullPositions = 0;

    // threat weights out of the int8 range in the last packed net
    uint64_t m_numClippedThreatWeights = 0;

    // inputs never active in an evaluated position, zeroed in packed nets
    std::vector<bool> m_unreachablePieceInputs;
    std::vector<bool> m_unreachableThreatInputs;

    // wall times of the last data generation and validation, which run while the GPU trains
    float m_lastLoaderMs = 0.0f;
    float m_lastValidationMs = 0.0f;
    TimePoint m_validationStartTime;

    uint32_t m_numValidations = 0;

    // results of the last validation; written by the validation task, read after it finishes
    int32_t m_lastTestPositionEvals[std::size(c_testPositions)] = {};
    float m_lastPackedNetRmse = 0.0f;
    float m_lastPackedNetSubsetRmse = 0.0f;
    float m_lastFloatNetSubsetRmse = 0.0f;

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

    // outThreatFeatures receives the white threat features, then the black ones (2 * MaxTrainingThreatFeatures slots)
    void GenerateTrainingEntry(std::mt19937& rng, TrainingEntry& outEntry, uint16_t* outThreatFeatures, float lambda, float bucketLeak);
    // false if the position must be drawn again
    bool GenerateTrainingEntryAnyVariant(std::mt19937& rng, TrainingEntry& outEntry, uint16_t* outThreatFeatures, float lambda, float bucketLeak);

    void GenerateTrainingBatch(std::mt19937& rng, TrainingSet& set, uint32_t batchIndex, float lambda, float bucketLeak);

    // deterministic: single-threaded with a dedicated RNG, so the set depends only on the seed
    void GenerateTrainingSet(TrainingSet& outSet, TaskBuilder& builder, float lambda, float bucketLeak, bool deterministic = false);

    void Validate(const TaskContext& ctx);

    NetworkDetails CollectNetworkDetails() const;
    void EvalTestPositions(int32_t* outEvals) const;
    static std::string FormatTestPositions(const int32_t* evals, int indent);
    void WriteNetworkDetails(const NetworkDetails& details, uint64_t numPositions);
    void PrintNetworkDetails(const NetworkDetails& details, int indent) const;

    // Forward pass on the float weights, matching what the CUDA trainer optimizes
    float EvalFloatNetwork(const TrainingEntry& entry, const uint16_t* threatFeatures) const;

    // "<name>/<name><suffix>", e.g. "eval-86/eval-86-10B.pnn"
    std::string OutputPath(const std::string& suffix) const
    {
        return m_options.name + "/" + m_options.name + suffix;
    }

    void PackNetwork();

    // Loads a packed net into the float weights
    bool UnpackNetwork(const char* path);

    void ReviveDeadNeurons();

    // Queues a training iteration on the GPU without waiting for it. downloadWeights also queues a download
    // of the weights after it, for validation.
    void QueueTrainingIteration(size_t iteration, float learningRate, bool downloadWeights);

    // waits for the GPU, then saves the packed net and the full training state
    void SaveCheckpoint(uint64_t numPositions, const std::string& timeTag, int indent);
};

void CudaNetworkTrainer::InitNetwork()
{
    const uint32_t accumulatorSize = nn::AccumulatorSize;

    m_featureTransformerWeights = std::make_shared<nn::WeightsStorage>(nn::cuda::FeatureTransformerInputs, accumulatorSize, 1);
    m_featureTransformerWeights->m_weightsRange = 5.0f;
    m_featureTransformerWeights->m_biasRange = 5.0f;

    m_threatWeights = std::make_shared<nn::WeightsStorage>(nn::NumThreatInputs, accumulatorSize, 1);
    m_threatWeights->m_weightsRange = nn::cuda::ThreatWeightRange;
    m_threatWeights->m_biasRange = 0.0f; // the bias row is unused

    // L1/L2 weights are int8; the range is exactly what the quantization scale can represent.
    // Their biases are int32, so the range only has to keep the layer's int32 sum from overflowing.
    constexpr float hiddenWeightRange = INT8_MAX / nn::HiddenLayerWeightQuantizationScale;
    constexpr float hiddenBiasRange = 64.0f;

    m_l1Weights = std::make_shared<nn::WeightsStorage>(nn::L1InputSize, nn::L1Size, nn::NumVariants);
    m_l1Weights->m_weightsRange = hiddenWeightRange;
    m_l1Weights->m_biasRange = hiddenBiasRange;

    m_l2Weights = std::make_shared<nn::WeightsStorage>(nn::L1Size, nn::L2Size, nn::NumVariants);
    m_l2Weights->m_weightsRange = hiddenWeightRange;
    m_l2Weights->m_biasRange = hiddenBiasRange;

    m_l3Weights = std::make_shared<nn::WeightsStorage>(nn::L2Size, 1, nn::NumVariants);
    m_l3Weights->m_weightsRange = INT16_MAX / nn::OutputLayerWeightQuantizationScale;
    m_l3Weights->m_biasRange = hiddenBiasRange;

    // Initialize CUDA network
    m_cudaNetwork.Init(m_featureTransformerWeights, m_threatWeights, m_l1Weights, m_l2Weights, m_l3Weights);

    // AdamW weight decay (QAT is enabled inside CudaNeuralNetwork::Init).
    m_cudaNetwork.SetFeatureTransformerWeightDecay(m_options.featureTransformerWeightDecay);
    m_cudaNetwork.SetAdamBetas(m_options.beta1, m_options.beta2);

    m_unreachablePieceInputs = GetUnreachablePieceInputs();
    m_unreachableThreatInputs = GetUnreachableThreatInputs();
    std::cout << "Unreachable inputs zeroed in packed nets: "
        << std::count(m_unreachablePieceInputs.begin(), m_unreachablePieceInputs.end(), true) << " piece, "
        << std::count(m_unreachableThreatInputs.begin(), m_unreachableThreatInputs.end(), true) << " threat" << std::endl;
}

// False if the position has more pieces than a training entry holds. outThreatFeatures receives the white
// threat features, then the black ones, and needs 2 * MaxTrainingThreatFeatures slots.
static bool PositionToTrainingEntry(const Position& pos, TrainingEntry& outEntry, uint16_t* outThreatFeatures)
{
    ASSERT(pos.GetSideToMove() == White);

    constexpr uint32_t maxFeatures = 64;

    uint16_t whiteFeatures[maxFeatures];
    uint32_t numWhiteFeatures = PositionToFeaturesVector(pos, whiteFeatures, pos.GetSideToMove());
    ASSERT(numWhiteFeatures <= maxFeatures);

    uint16_t blackFeatures[maxFeatures];
    uint32_t numBlackFeatures = PositionToFeaturesVector(pos, blackFeatures, pos.GetSideToMove() ^ 1);
    ASSERT(numBlackFeatures == numWhiteFeatures);

    if (numWhiteFeatures > std::size(outEntry.whiteFeatures))
        return false;

    // ThreatRecordsToFeatures writes one slot per threat, at most MaxTrainingThreatFeatures in a position that fits
    static_assert(MaxTrainingThreatFeatures >= 8 * (std::size(TrainingEntry{}.whiteFeatures) - 2), "every position that fits must fit its threats");
    ThreatRecord threats[nn::MaxThreatRecords];
    const uint32_t numThreats = GetThreatRecords(pos, threats);
    ASSERT(numThreats <= MaxTrainingThreatFeatures);

    const uint32_t numWhiteThreatFeatures = ThreatRecordsToFeatures(threats, numThreats, GetThreatOrientation(pos, pos.GetSideToMove()), outThreatFeatures);
    const uint32_t numBlackThreatFeatures = ThreatRecordsToFeatures(threats, numThreats, GetThreatOrientation(pos, pos.GetSideToMove() ^ 1), outThreatFeatures + numWhiteThreatFeatures);
    ASSERT(numBlackThreatFeatures == numWhiteThreatFeatures);
    UNUSED(numBlackThreatFeatures);

    outEntry.numWhiteFeatures = (uint8_t)numWhiteFeatures;
    outEntry.numBlackFeatures = (uint8_t)numBlackFeatures;
    for (uint32_t i = 0; i < numWhiteFeatures; ++i)
        outEntry.whiteFeatures[i] = whiteFeatures[i];
    for (uint32_t i = 0; i < numBlackFeatures; ++i)
        outEntry.blackFeatures[i] = blackFeatures[i];

    outEntry.numThreatFeatures = (uint8_t)numWhiteThreatFeatures;
    outEntry.variant = GetNetworkVariant(pos);
    return true;
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

void CudaNetworkTrainer::GenerateTrainingEntry(std::mt19937& rng, TrainingEntry& outEntry, uint16_t* outThreatFeatures, float lambda, float bucketLeak)
{
    for (;;)
    {
        if (!GenerateTrainingEntryAnyVariant(rng, outEntry, outThreatFeatures, lambda, bucketLeak))
            m_numSkippedPositions++;
        else if (!(m_options.frozenVariantsMask & (1u << outEntry.variant)))
            break;
    }
}

bool CudaNetworkTrainer::GenerateTrainingEntryAnyVariant(std::mt19937& rng, TrainingEntry& outEntry, uint16_t* outThreatFeatures, float lambda, float bucketLeak)
{
    Position pos;
    PositionEntry entry;

    // the loader prints the error of a failed read
    if (!m_dataLoader.FetchNextPosition(rng, entry, pos))
        return false;

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

    // a tablebase result replaces the score
    if (tbScore != Game::Score::Unknown)
        score = tbScore == Game::Score::WhiteWins ? 1.0f : (tbScore == Game::Score::BlackWins ? 0.0f : 0.5f);

    if (!PositionToTrainingEntry(pos, outEntry, outThreatFeatures))
        return false;

    outEntry.targetOutput = score;

    // training as if the position had one piece more or less only changes the bucket at a range edge
    if (bucketLeak > 0.0f && std::bernoulli_distribution(bucketLeak)(rng))
    {
        const int32_t shiftedPieces = (int32_t)pos.GetNumPiecesExcludingKing() + (std::bernoulli_distribution(0.5)(rng) ? 1 : -1);
        const bool anyQueen = (pos.Whites().queens | pos.Blacks().queens) != 0;
        outEntry.variant = GetNetworkVariant((uint32_t)std::max(shiftedPieces, 0), anyQueen);
    }

    return true;
}

void CudaNetworkTrainer::GenerateTrainingBatch(std::mt19937& rng, TrainingSet& set, uint32_t batchIndex, float lambda, float bucketLeak)
{
    const uint32_t begin = set.BatchBegin(batchIndex);
    const uint32_t end = set.BatchEnd(batchIndex);
    uint16_t* batchThreatFeatures = set.BatchThreatFeatures(batchIndex);
    const uint32_t capacity = (end - begin) * TrainingThreatFeatureBudget;

    uint16_t threatFeatures[2 * MaxTrainingThreatFeatures];
    uint32_t numThreatFeatures = 0;
    for (uint32_t i = begin; i < end; ++i)
    {
        TrainingEntry& entry = set.entries[i];
        for (;;)
        {
            GenerateTrainingEntry(rng, entry, threatFeatures, lambda, bucketLeak);
            if (numThreatFeatures + 2u * entry.numThreatFeatures <= capacity)
                break;
            m_numThreatListFullPositions++;
        }

        entry.threatFeaturesOffset = numThreatFeatures;
        memcpy(batchThreatFeatures + numThreatFeatures, threatFeatures, 2u * entry.numThreatFeatures * sizeof(uint16_t));
        numThreatFeatures += 2u * entry.numThreatFeatures;
    }
    TRAINER_CHECK(numThreatFeatures <= capacity);
    set.numBatchThreatFeatures[batchIndex] = numThreatFeatures;

    // entries keep their threat feature offsets, so only they are reordered
    SortBatchByKingBucket(set.entries.data() + begin, end - begin);
}

void CudaNetworkTrainer::GenerateTrainingSet(TrainingSet& outSet, TaskBuilder& builder, float lambda, float bucketLeak, bool deterministic)
{
    if (deterministic)
    {
        builder.Task("GenerateSet", [this, &outSet, lambda, bucketLeak](const TaskContext&)
        {
            for (uint32_t batchIndex = 0; batchIndex < outSet.NumBatches(); ++batchIndex)
                GenerateTrainingBatch(m_deterministicRng, outSet, batchIndex, lambda, bucketLeak);
        });
    }
    else
    {
        builder.ParallelFor("GenerateSet", outSet.NumBatches(),
            [this, &outSet, lambda, bucketLeak](const TaskContext& ctx, uint32_t batchIndex)
        {
            GenerateTrainingBatch(m_randomGenerators[ctx.threadId], outSet, batchIndex, lambda, bucketLeak);
        }, 0);
    }
}

void CudaNetworkTrainer::QueueTrainingIteration(size_t iteration, float learningRate, bool downloadWeights)
{
    const uint32_t batchSize = m_options.batchSize;
    const uint32_t numBatches = cNumTrainingVectorsPerIteration / batchSize;
    TRAINER_CHECK(m_trainingSet_Read.NumBatches() == numBatches);
    TRAINER_CHECK(m_trainingSet_Read.uploadingIteration == SIZE_MAX);
    m_trainingSet_Read.uploadingIteration = iteration;
    m_cudaBatchData.batchSize = batchSize;

    const uint32_t slot = iteration % nn::cuda::CudaNeuralNetwork::NumIterationSlots;
    m_cudaNetwork.BeginIteration(slot, m_cudaBatchData);

    const auto copyBatch = [this, batchSize](uint32_t batchIndex)
    {
        const uint32_t numThreatFeatures = m_trainingSet_Read.numBatchThreatFeatures[batchIndex];
        TRAINER_CHECK(numThreatFeatures <= batchSize * TrainingThreatFeatureBudget);
#ifndef CONFIGURATION_FINAL
        for (uint32_t i = m_trainingSet_Read.BatchBegin(batchIndex); i < m_trainingSet_Read.BatchEnd(batchIndex); ++i)
        {
            const TrainingEntry& entry = m_trainingSet_Read.entries[i];
            ASSERT(entry.variant < nn::NumVariants);
            ASSERT(entry.numWhiteFeatures == entry.numBlackFeatures && entry.numWhiteFeatures <= std::size(entry.whiteFeatures));
            ASSERT(entry.numThreatFeatures <= MaxTrainingThreatFeatures);
            ASSERT(entry.threatFeaturesOffset + 2u * entry.numThreatFeatures <= numThreatFeatures);
            ASSERT(entry.targetOutput >= 0.0f && entry.targetOutput <= 1.0f);
        }
#endif // CONFIGURATION_FINAL

        m_cudaNetwork.CopyTrainingBatchAsync(
            m_cudaBatchData,
            m_trainingSet_Read.entries.data() + m_trainingSet_Read.BatchBegin(batchIndex),
            batchSize,
            m_trainingSet_Read.BatchThreatFeatures(batchIndex),
            numThreatFeatures);
    };

    // Prefetch the first batch (this copy cannot overlap anything; it is the only per-iteration
    // stall). Subsequent batches are prefetched while the previous batch trains.
    copyBatch(0);

    for (uint32_t b = 0; b < numBatches; ++b)
    {
        m_cudaNetwork.SwapTrainingBatch(m_cudaBatchData);

        // Forward pass (waits for this batch's copy to complete)
        m_cudaNetwork.Forward(m_cudaBatchData);

        // Backward pass
        m_cudaNetwork.Backward(m_cudaBatchData, learningRate);

        // Prefetch the next batch's training vectors into the other buffer on the copy stream
        if (b + 1 < numBatches)
            copyBatch(b + 1);
    }

    m_cudaNetwork.EndIteration(slot, m_cudaBatchData);

    if (downloadWeights)
        m_cudaNetwork.StartWeightsDownload();
}

void CudaNetworkTrainer::SaveCheckpoint(uint64_t numPositions, const std::string& timeTag, int indent)
{
    TRAINER_CHECK(!m_hostWeightsInUse);

    m_cudaNetwork.Synchronize();
    m_cudaNetwork.CopyWeightsToHost();

    // labelled by the position count reached, matching the eval-<version>-<positions>B convention
    const std::string suffix = "-" + std::to_string(numPositions / 1'000'000'000ull) + "B";
    PackNetwork();
    m_packedNet->SaveToFile(OutputPath(suffix + ".pnn").c_str());
    m_cudaNetwork.SaveCheckpoint(OutputPath(suffix + ".ckpt").c_str(), numPositions);

    std::cout << timeTag << " Saved checkpoint: " << OutputPath(suffix) << ".{pnn,ckpt}" << std::endl;
    PrintNetworkDetails(CollectNetworkDetails(), indent);
}

static float EvalPackedNetwork(const TrainingEntry& entry, const uint16_t* threatFeatures, const nn::PackedNeuralNetwork& net)
{
    const int32_t packedNetworkOutput = net.Run(
        entry.whiteFeatures, entry.numWhiteFeatures,
        entry.blackFeatures, entry.numBlackFeatures,
        threatFeatures, threatFeatures + entry.numThreatFeatures, entry.numThreatFeatures,
        entry.variant);
    const float scaledPackedNetworkOutput = (float)packedNetworkOutput / (float)(nn::OutputScale * nn::WeightScale) * c_nnOutputToCentiPawns / 100.0f;
    return EvalToExpectedGameScore(scaledPackedNetworkOutput);
}

static float FakeQuantize(float w, float scale)
{
    return std::round(w * scale) / scale;
}

// Forward pass of one dense output-subnet layer on the fake-quantized float weights.
// Weights are input-major (weights[input * outputSize + output]), biases follow the matrix.
static void FloatDenseLayer(
    float* output, const float* input, const nn::Values& weights,
    uint32_t inputSize, uint32_t outputSize, bool applyCReLU,
    float weightScale, float biasScale)
{
    for (uint32_t o = 0; o < outputSize; ++o)
    {
        float sum = FakeQuantize(weights[inputSize * outputSize + o], biasScale);
        for (uint32_t i = 0; i < inputSize; ++i)
            sum += input[i] * FakeQuantize(weights[i * outputSize + o], weightScale);

        output[o] = applyCReLU ? std::clamp(sum, 0.0f, 1.0f) : sum;
    }
}

// Reference forward pass on the float weights, fake-quantized like in the CUDA forward pass (the
// unquantized master weights drift off the grid in directions the quantized net does not see).
// Comparing it against the packed network on the same positions tells a training problem apart
// from a packing / quantized-inference problem: only the latter moves the two numbers apart.
float CudaNetworkTrainer::EvalFloatNetwork(const TrainingEntry& entry, const uint16_t* threatFeatures) const
{
    constexpr uint32_t halfSize = nn::AccumulatorSize / 2;
    const nn::Values& ftWeights = m_featureTransformerWeights->m_variants.front().m_weights;
    const nn::Values& threatWeights = m_threatWeights->m_variants.front().m_weights;

    float accumulators[2][nn::AccumulatorSize];
    for (uint32_t perspective = 0; perspective < 2; ++perspective)
    {
        const uint16_t* features = perspective == 0 ? entry.whiteFeatures : entry.blackFeatures;
        const uint32_t numFeatures = perspective == 0 ? entry.numWhiteFeatures : entry.numBlackFeatures;

        const float* biases = ftWeights.data() + nn::cuda::FeatureTransformerInputs * nn::AccumulatorSize;
        for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
            accumulators[perspective][i] = FakeQuantize(biases[i], nn::InputLayerBiasQuantizationScale);

        for (uint32_t f = 0; f < numFeatures; ++f)
        {
            const uint32_t feature = features[f];
            const float* weights = ftWeights.data() + feature * nn::AccumulatorSize;
            const float* factorizer = ftWeights.data() +
                (nn::NumNetworkInputs + feature % nn::cuda::FactorizerInputs) * nn::AccumulatorSize;

            // bucket and factorizer weight are quantized as a sum, like when they are packed
            for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
            {
                const float w = m_options.useFactorizer ? weights[i] + factorizer[i] : weights[i];
                accumulators[perspective][i] += FakeQuantize(w, nn::InputLayerWeightQuantizationScale);
            }
        }

        const uint16_t* perspectiveThreatFeatures = threatFeatures + perspective * entry.numThreatFeatures;
        for (uint32_t f = 0; f < entry.numThreatFeatures; ++f)
        {
            const float* weights = threatWeights.data() + perspectiveThreatFeatures[f] * nn::AccumulatorSize;
            for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
                accumulators[perspective][i] += FakeQuantize(weights[i], nn::InputLayerWeightQuantizationScale);
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
        nn::L1InputSize, nn::L1Size, true,
        nn::HiddenLayerWeightQuantizationScale, nn::HiddenLayerBiasQuantizationScale);

    float l3Input[nn::L2Size];
    FloatDenseLayer(l3Input, l2Input, m_l2Weights->m_variants[entry.variant].m_weights,
        nn::L1Size, nn::L2Size, true,
        nn::HiddenLayerWeightQuantizationScale, nn::HiddenLayerBiasQuantizationScale);

    float output = 0.0f;
    FloatDenseLayer(&output, l3Input, m_l3Weights->m_variants[entry.variant].m_weights,
        nn::L2Size, 1u, false,
        nn::OutputLayerWeightQuantizationScale, nn::OutputLayerBiasQuantizationScale);

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

        const TrainingEntry& entry = m_validationSet.entries[i];
        const uint16_t* threatFeatures = m_validationSet.EntryThreatFeatures(i);
        const float expectedValue = entry.targetOutput;

        ValidationStats& stats = threadData.stats;
        {
            const float nnPackedValue = EvalPackedNetwork(entry, threatFeatures, *m_packedNet);
            const float error = expectedValue - nnPackedValue;
            stats.nnPackedErrorSum += (double)error * (double)error;

            if (i < cNumFloatReferenceVectors)
            {
                const float floatError = expectedValue - EvalFloatNetwork(entry, threatFeatures);
                stats.nnPackedSubsetErrorSum += (double)error * (double)error;
                stats.floatErrorSum += (double)floatError * (double)floatError;
            }
        }
    });

    taskBuilder.Fence();

    taskBuilder.Task("AccumulateValidationStats", [this](const TaskContext&)
    {
        ValidationStats stats;
        for (size_t i = 0; i < ThreadPool::GetInstance().GetNumThreads(); ++i)
        {
            const ValidationStats& threadStats = m_validationPerThreadData[i].stats;

            stats.nnPackedErrorSum += threadStats.nnPackedErrorSum;
            stats.nnPackedSubsetErrorSum += threadStats.nnPackedSubsetErrorSum;
            stats.floatErrorSum += threadStats.floatErrorSum;
        }

        m_lastPackedNetRmse = (float)sqrt(stats.nnPackedErrorSum / cNumValidationVectorsPerIteration);
        m_lastPackedNetSubsetRmse = (float)sqrt(stats.nnPackedSubsetErrorSum / cNumFloatReferenceVectors);
        m_lastFloatNetSubsetRmse = (float)sqrt(stats.floatErrorSum / cNumFloatReferenceVectors);

        EvalTestPositions(m_lastTestPositionEvals);

        m_lastValidationMs = 1000.0f * (TimePoint::GetCurrent() - m_validationStartTime).ToSeconds();
        m_hostWeightsInUse = false;
    });
}

CudaNetworkTrainer::NetworkDetails CudaNetworkTrainer::CollectNetworkDetails() const
{
    NetworkDetails details{};

    const nn::WeightsStorage* layers[] = { m_featureTransformerWeights.get(), m_threatWeights.get(), m_l1Weights.get(), m_l2Weights.get(), m_l3Weights.get() };
    static_assert(std::size(layers) == std::size(c_layerNames));
    for (size_t i = 0; i < std::size(layers); ++i)
        details.layerStats[i] = layers[i]->ComputeStats();

    EvalTestPositions(details.testPositionEvals);

    return details;
}

void CudaNetworkTrainer::EvalTestPositions(int32_t* outEvals) const
{
    for (size_t i = 0; i < std::size(c_testPositions); ++i)
    {
        TrainingEntry entry;
        uint16_t threatFeatures[2 * MaxTrainingThreatFeatures];
        VERIFY(PositionToTrainingEntry(Position(c_testPositions[i]), entry, threatFeatures));
        outEvals[i] = ExpectedGameScoreToInternalEval(EvalPackedNetwork(entry, threatFeatures, *m_packedNet));
    }
}

std::string CudaNetworkTrainer::FormatTestPositions(const int32_t* evals, int indent)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%*stest positions (packed net eval):\n", indent, "");
    std::string str = buf;

    for (size_t i = 0; i < std::size(c_testPositions); ++i)
    {
        snprintf(buf, sizeof(buf), "%*s%6d  %s\n", indent, "", evals[i], c_testPositions[i]);
        str += buf;
    }

    return str;
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

    for (size_t i = 0; i < std::size(c_testPositions); ++i)
        m_testPositionsFile << numPositions << '\t' << i << '\t' << details.testPositionEvals[i] << '\t' << c_testPositions[i] << '\n';
    m_testPositionsFile.flush();
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

    str += FormatTestPositions(details.testPositionEvals, indent);

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
    f << "ftWeightRange="              << m_featureTransformerWeights->m_weightsRange << "\n";
    f << "bucketLeak="                 << m_options.bucketLeak << "\n";
    f << "frozenVariantsMask="         << m_options.frozenVariantsMask << "\n";
    f << "freezeFeatureTransformer="   << (m_options.freezeFeatureTransformer ? "true" : "false") << "\n";
    f << "factorizer="                 << (m_options.useFactorizer ? "true" : "false") << "\n";
    f << "reviveDeadNeurons="          << (m_options.reviveDeadNeurons ? "true" : "false") << "\n";
    f << "warmupIterations="           << cWarmupTime << "\n";
    f << "positionsPerIteration="      << cNumTrainingVectorsPerIteration << "\n";
    f << "validationPositions="        << cNumValidationVectorsPerIteration << "\n";
    f << "validationInterval="         << cValidationInterval << "\n";
    f << "checkpointInterval="         << cCheckpointInterval << "\n";
    f << "\n";

    f << "dataPath=";
    for (size_t i = 0; i < m_dataLoader.GetDataPaths().size(); ++i)
        f << (i > 0 ? ";" : "") << m_dataLoader.GetDataPaths()[i];
    f << "\n";
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
    UNUSED(maxWeightRange);
    UNUSED(maxBiasRange);

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

void CudaNetworkTrainer::PackNetwork()
{
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

        // fold the factorizer into every king bucket
        if (m_options.useFactorizer)
        {
            for (uint32_t feature = 0; feature < nn::NumNetworkInputs; ++feature)
            {
                const float* factorizer = storedWeights.data() + (nn::NumNetworkInputs + feature % nn::cuda::FactorizerInputs) * nn::AccumulatorSize;
                float* target = weights.data() + feature * nn::AccumulatorSize;
                for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
                    target[i] += factorizer[i];
            }
        }

        PackWeights(
            weights,
            nn::NumNetworkInputs, nn::AccumulatorSize,
            const_cast<nn::FirstLayerWeightType*>(m_packedNet->accumulatorWeights),
            const_cast<nn::FirstLayerBiasType*>(m_packedNet->accumulatorBiases),
            nn::InputLayerWeightQuantizationScale, nn::InputLayerBiasQuantizationScale,
            // bucket and factorizer weights are clipped separately, so their sum can exceed the bucket range
            m_featureTransformerWeights->m_weightsRange + (m_options.useFactorizer ? nn::cuda::FactorizerWeightRange : 0.0f),
            m_featureTransformerWeights->m_biasRange,
            WeightLayout::InputMajor);
    }

    // threat weights, input-major like the piece weights; Adam keeps them in the int8 range
    {
        const nn::Values& storedWeights = m_threatWeights->m_variants.front().m_weights;

        uint64_t numClipped = 0;
        for (size_t i = 0; i < (size_t)nn::NumThreatInputs * nn::AccumulatorSize; ++i)
        {
            const int32_t quantizedWeight = (int32_t)std::round(storedWeights[i] * nn::InputLayerWeightQuantizationScale);
            const int32_t clampedWeight = std::clamp(quantizedWeight, -(int32_t)INT8_MAX, (int32_t)INT8_MAX);
            numClipped += quantizedWeight != clampedWeight;
            m_packedNet->threatWeights[i] = (nn::ThreatWeightType)clampedWeight;
        }
        m_numClippedThreatWeights = numClipped;
    }

    for (uint32_t input = 0; input < nn::NumNetworkInputs; ++input)
    {
        if (m_unreachablePieceInputs[input])
            std::fill_n(m_packedNet->accumulatorWeights + (size_t)input * nn::AccumulatorSize, nn::AccumulatorSize, (nn::FirstLayerWeightType)0);
    }
    for (uint32_t input = 0; input < nn::NumThreatInputs; ++input)
    {
        if (m_unreachableThreatInputs[input])
            std::fill_n(m_packedNet->threatWeights + (size_t)input * nn::AccumulatorSize, nn::AccumulatorSize, (nn::ThreatWeightType)0);
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
}

bool CudaNetworkTrainer::UnpackNetwork(const char* path)
{
    auto packedNet = std::make_unique<nn::PackedNeuralNetwork>();
    if (!packedNet->LoadFromFile(path))
    {
        std::cerr << "Failed to load neural network: " << path << std::endl;
        return false;
    }

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
    }

    // threat weights
    {
        nn::Values& weights = m_threatWeights->m_variants.front().m_weights;
        for (size_t i = 0; i < (size_t)nn::NumThreatInputs * nn::AccumulatorSize; ++i)
            weights[i] = (float)packedNet->threatWeights[i] / nn::InputLayerWeightQuantizationScale;
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

    return true;
}

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
        m_cudaNetwork.CopyWeightsToHost();

        std::cout << "Resuming from checkpoint: " << m_options.resumePath
            << " at " << std::setprecision(4) << positionsPassed / 1.0e9f << "B positions" << std::endl;
        if (m_options.restartSchedule)
            std::cout << "Restarting the schedule from zero positions" << std::endl;
    }
    else if (fromScratch)
    {
        std::cout << "Training from scratch" << std::endl;
        m_cudaNetwork.InitRandomWeights(m_options.seed);
        m_cudaNetwork.CopyWeightsToHost();
    }
    else
    {
        if (!UnpackNetwork(m_options.startNetPath.c_str()))
            return false;

        std::cout << "Starting from net: " << m_options.startNetPath << std::endl;

        m_cudaNetwork.CopyWeightsFromHost();
    }

    if (m_options.reviveDeadNeurons)
    {
        ReviveDeadNeurons();
        m_cudaNetwork.CopyWeightsFromHost();
    }

    if (m_options.freezeFeatureTransformer)
    {
        m_cudaNetwork.SetFeatureTransformerFrozen(true);
        std::cout << "Feature transformer frozen" << std::endl;
    }

    m_cudaNetwork.SetFactorizerEnabled(m_options.useFactorizer);
    std::cout << "Factorizer " << (m_options.useFactorizer ? "on" : "off") << std::endl;

    if (m_options.bucketLeak > 0.0f)
        std::cout << "Bucket leak: " << m_options.bucketLeak << std::endl;

    if (m_options.frozenVariantsMask != 0)
    {
        std::cout << "Frozen output variants:";
        for (uint32_t v = 0; v < nn::NumVariants; ++v)
            if (m_options.frozenVariantsMask & (1u << v))
                std::cout << " " << v;
        std::cout << std::endl;
    }

    std::cout << "Loader: decisiveSkip " << (m_options.loaderOptions.decisiveSkip ? "on" : "off")
        << ", pieceCountTarget " << (m_options.loaderOptions.pieceCountTarget ? "on" : "off")
        << ", hmcSkipFrom20 " << (m_options.loaderOptions.hmcSkipFrom20 ? "on" : "off") << std::endl;

    if (!m_dataLoader.Init(m_randomGenerators[0], m_options.loaderOptions, m_options.dataPaths))
    {
        std::cout << "ERROR: Failed to initialize data loader" << std::endl;
        return false;
    }

    WriteRunSettings(m_numTrainingVectorsPassed);

    const float validationLambda = 1.0f;

    // resuming keeps the milestones already written on the previous run
    uint64_t lastCheckpoint = m_numTrainingVectorsPassed / cCheckpointInterval;
    uint64_t lastDetails = m_numTrainingVectorsPassed / cDetailsInterval;

    // initial training set generation
    {
        Waitable waitable;
        {
            TaskBuilder taskBuilder{ waitable };
            GenerateTrainingSet(m_validationSet, taskBuilder, validationLambda, 0.0f, true);
        }
        waitable.Wait();
    }

    const TimePoint startTime = TimePoint::GetCurrent();

    // Iteration k is queued on the GPU while iteration k - 1 still runs, and the data for k + 1 is generated
    // meanwhile. Every cValidationInterval iterations the weights are downloaded after an iteration and validated
    // in the background over the following iterations. Nothing in the loop waits for the GPU or the validation.
    struct QueuedIteration
    {
        size_t iteration;
        uint64_t numPositions;  // position count once the iteration is done, as in checkpoint names
        float learningRate;
        float lambda;
        bool downloadsWeights;
    };
    std::optional<QueuedIteration> previous;
    size_t lastFinishedIteration = SIZE_MAX;

    struct ProgressSums
    {
        uint32_t numIterations = 0;
        double trainingMseSum = 0.0;
        double gpuMsSum = 0.0;
        double periodMsSum = 0.0;
        double loaderMsSum = 0.0;
    };
    ProgressSums reportSums;    // since the last report
    ProgressSums rowSums;       // since the last progress file row
    TimePoint lastReportTime = startTime;
    uint64_t lastReportPositions = m_numTrainingVectorsPassed;

    // progress file row of the validated iteration, completed when its validation finishes
    struct PendingRow
    {
        size_t iteration;
        uint64_t numPositions;
        float elapsedSeconds;
        float learningRate;
        float lambda;
        ProgressSums sums;
    };
    std::optional<PendingRow> pendingRow;

    // results of the last finished validation, for the reports
    struct ValidationResults
    {
        float packedNetRmse = 0.0f;
        float floatNetSubsetRmse = 0.0f;
        float packedNetSubsetRmse = 0.0f;
        float validationMs = 0.0f;
        uint64_t numClippedThreatWeights = 0;
        int32_t testPositionEvals[std::size(c_testPositions)] = {};
    };
    ValidationResults validationResults;

    std::unique_ptr<Waitable> validation;

    const auto finishValidation = [&]()
    {
        if (!validation)
            return;

        validation->Wait();
        validation.reset();
        TRAINER_CHECK(!m_hostWeightsInUse);
        TRAINER_CHECK(pendingRow.has_value());

        m_numValidations++;
        validationResults.validationMs = m_lastValidationMs;
        TRAINER_CHECK(std::isfinite(m_lastPackedNetRmse) && std::isfinite(m_lastFloatNetSubsetRmse) && std::isfinite(m_lastPackedNetSubsetRmse));
        validationResults.packedNetRmse = m_lastPackedNetRmse;
        validationResults.floatNetSubsetRmse = m_lastFloatNetSubsetRmse;
        validationResults.packedNetSubsetRmse = m_lastPackedNetSubsetRmse;
        validationResults.numClippedThreatWeights = m_numClippedThreatWeights;
        std::copy(std::begin(m_lastTestPositionEvals), std::end(m_lastTestPositionEvals), validationResults.testPositionEvals);

        // averaged over the iterations since the previous row
        const double n = (double)pendingRow->sums.numIterations;
        TRAINER_CHECK(n > 0.0);
        m_progressFile << pendingRow->iteration << '\t' << pendingRow->numPositions << '\t' << std::setprecision(6) << pendingRow->elapsedSeconds
            << '\t' << pendingRow->learningRate << '\t' << pendingRow->lambda << '\t' << sqrt(pendingRow->sums.trainingMseSum / n)
            << '\t' << validationResults.packedNetRmse << '\t' << validationResults.floatNetSubsetRmse << '\t' << validationResults.packedNetSubsetRmse
            << '\t' << pendingRow->sums.periodMsSum / n << '\t' << pendingRow->sums.gpuMsSum / n << '\t' << pendingRow->sums.loaderMsSum / n
            << '\t' << validationResults.validationMs << '\n';

        if (m_numValidations % cPackedNetSaveInterval == 0)
            m_packedNet->SaveToFile(OutputPath(".pnn").c_str());

        const uint64_t details = pendingRow->numPositions / cDetailsInterval;
        if (details > lastDetails)
        {
            lastDetails = details;
            WriteNetworkDetails(CollectNetworkDetails(), pendingRow->numPositions);
        }

        pendingRow.reset();
    };

    for (size_t iteration = 0; iteration < m_options.maxIterations; ++iteration)
    {
        const bool useWarmup = !fromScratch && (!resuming || m_options.restartSchedule) && cWarmupTime > 0.0f;
        const float warmup = useWarmup ? (iteration < cWarmupTime ? (float)(iteration + 1) / cWarmupTime : 1.0f) : 1.0f;
        const float t = std::min(1.0f, (float)((double)m_numTrainingVectorsPassed * 1.0e-9 / (double)m_options.trainingLength));
        const float learningRate = warmup * GetScheduledLearningRate(m_options, t);
        const float lambda = m_options.startLambda + (m_options.endLambda - m_options.startLambda) * t;

        // a checkpoint is the exact state after the last queued iteration, so the GPU finishes it first
        if (previous && previous->numPositions / cCheckpointInterval > lastCheckpoint)
        {
            lastCheckpoint = previous->numPositions / cCheckpointInterval;
            finishValidation();
            const std::string timeTag = "[" + FormatDuration((TimePoint::GetCurrent() - startTime).ToSeconds()) + "]";
            SaveCheckpoint(previous->numPositions, timeTag, (int)timeTag.size() + 1);
        }

        // skip training in the first iteration, as the data is not ready yet
        std::optional<QueuedIteration> queued;
        if (iteration > 0)
        {
            queued = QueuedIteration{ iteration, m_numTrainingVectorsPassed + cNumTrainingVectorsPerIteration,
                learningRate, lambda, iteration % cValidationInterval == 0 };

            // the validation started cValidationInterval - 1 iterations ago has read the previous download
            if (queued->downloadsWeights)
                finishValidation();

            QueueTrainingIteration(iteration, learningRate, queued->downloadsWeights);
        }

        if (previous && previous->downloadsWeights)
        {
            TRAINER_CHECK(!validation);
            validation = std::make_unique<Waitable>();
            TaskBuilder taskBuilder{ *validation };
            taskBuilder.Task("Validate", [this](const TaskContext& ctx)
            {
                m_validationStartTime = TimePoint::GetCurrent();
                TRAINER_CHECK(!m_hostWeightsInUse);
                m_hostWeightsInUse = true;
                m_cudaNetwork.FinishWeightsDownload();
                PackNetwork();
                Validate(ctx);
            });
        }

        Waitable waitable;
        {
            TaskBuilder taskBuilder{ waitable };

            // the set written here was read by the previous iteration's batch uploads
            const size_t uploadingIteration = previous ? previous->iteration : SIZE_MAX;
            taskBuilder.Task("GenerateTrainingSet", [this, lambda, uploadingIteration](const TaskContext& ctx)
            {
                TRAINER_CHECK(m_trainingSet_Write.uploadingIteration == uploadingIteration);
                if (uploadingIteration != SIZE_MAX)
                {
                    m_cudaNetwork.WaitForIterationUploads(uploadingIteration % nn::cuda::CudaNeuralNetwork::NumIterationSlots);
                    m_trainingSet_Write.uploadingIteration = SIZE_MAX;
                }

                const TimePoint taskStartTime = TimePoint::GetCurrent();
                TaskBuilder taskBuilder{ ctx };
                GenerateTrainingSet(m_trainingSet_Write, taskBuilder, lambda, m_options.bucketLeak);
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

        const TimePoint now = TimePoint::GetCurrent();
        const float elapsedSeconds = (now - startTime).ToSeconds();
        const std::string timeTag = "[" + FormatDuration(elapsedSeconds) + "]";
        const int indent = (int)timeTag.size() + 1;

        if (previous)
        {
            // the GPU is busy with the iteration queued in this loop
            const uint32_t slot = previous->iteration % nn::cuda::CudaNeuralNetwork::NumIterationSlots;
            const uint32_t previousSlot = lastFinishedIteration != SIZE_MAX ? lastFinishedIteration % nn::cuda::CudaNeuralNetwork::NumIterationSlots : UINT32_MAX;
            const nn::cuda::CudaNeuralNetwork::IterationResult result = m_cudaNetwork.FinishIteration(slot, previousSlot);
            lastFinishedIteration = previous->iteration;

            const double trainingMse = result.lossSum / (double)cNumTrainingVectorsPerIteration;
            TRAINER_CHECK(std::isfinite(trainingMse));
            for (ProgressSums* sums : { &reportSums, &rowSums })
            {
                sums->numIterations++;
                sums->trainingMseSum += trainingMse;
                sums->gpuMsSum += result.gpuMs;
                sums->periodMsSum += result.periodMs;
                sums->loaderMsSum += m_lastLoaderMs;
            }

            if (previous->downloadsWeights)
            {
                TRAINER_CHECK(!pendingRow.has_value());
                pendingRow = PendingRow{ previous->iteration, previous->numPositions, elapsedSeconds, previous->learningRate, previous->lambda, rowSums };
                rowSums = ProgressSums();
            }
        }

        const float secondsSinceReport = (now - lastReportTime).ToSeconds();
        if (secondsSinceReport >= cReportInterval && reportSums.numIterations > 0 && m_numValidations > 0)
        {
            const uint64_t numPositions = previous->numPositions;
            const uint64_t totalPositions = m_options.trainingLength * 1'000'000'000ull;
            const double positionsPerSecond = (double)(numPositions - lastReportPositions) / secondsSinceReport;
            const auto taskPositionsPerSecond = [](double msSum, double count) { return msSum > 0.0 ? 1000.0 * count * cNumTrainingVectorsPerIteration / msSum : 0.0; };
            const std::string eta = numPositions < totalPositions ? FormatDuration((double)(totalPositions - numPositions) / positionsPerSecond) : "done";

            char buf[256];
            std::string str;

            snprintf(buf, sizeof(buf), "%s %.1fB/%lluB (%.1f%%) ETA %s | %.2fM pos/s | loss train %.5f",
                timeTag.c_str(), numPositions * 1.0e-9, (unsigned long long)m_options.trainingLength, 100.0 * numPositions / totalPositions,
                eta.c_str(), positionsPerSecond * 1.0e-6, sqrt(reportSums.trainingMseSum / reportSums.numIterations));
            str += buf;
            snprintf(buf, sizeof(buf), " val %.5f", validationResults.packedNetRmse);
            str += buf;

            snprintf(buf, sizeof(buf), "\n%*sLR %.3g lambda %.2f", indent, "", previous->learningRate, previous->lambda);
            str += buf;
            snprintf(buf, sizeof(buf), " | float vs pnn %.5f / %.5f", validationResults.floatNetSubsetRmse, validationResults.packedNetSubsetRmse);
            str += buf;
            snprintf(buf, sizeof(buf), " | iteration %zu\n", previous->iteration);
            str += buf;

            // a validation covers cValidationInterval iterations
            const double n = (double)reportSums.numIterations;
            const double gpuMs = reportSums.gpuMsSum / n;
            const double loaderMs = reportSums.loaderMsSum / n;
            const double validationMs = validationResults.validationMs / cValidationInterval;
            const char* limit = "training";
            if (loaderMs > gpuMs && loaderMs >= validationMs)
                limit = "data loading";
            else if (validationMs > gpuMs)
                limit = "validation";

            snprintf(buf, sizeof(buf), "%*sspeed M pos/s: GPU %.2f, training %.2f (GPU idle %.1f%%), data loading %.2f, validation %.2f | limited by %s\n", indent, "",
                taskPositionsPerSecond(reportSums.gpuMsSum, n) * 1.0e-6, taskPositionsPerSecond(reportSums.periodMsSum, n) * 1.0e-6,
                100.0 * std::max(0.0, 1.0 - reportSums.gpuMsSum / reportSums.periodMsSum),
                taskPositionsPerSecond(reportSums.loaderMsSum, n) * 1.0e-6, taskPositionsPerSecond(validationMs, 1.0) * 1.0e-6, limit);
            str += buf;

            if (m_numSkippedPositions > 0)
            {
                snprintf(buf, sizeof(buf), "%*sskipped positions with more than 32 pieces or failed reads: %llu\n", indent, "", (unsigned long long)m_numSkippedPositions.load());
                str += buf;
            }

            if (m_numThreatListFullPositions > 0)
            {
                snprintf(buf, sizeof(buf), "%*sWARNING: positions drawn again because a batch's threat feature list was full: %llu\n", indent, "", (unsigned long long)m_numThreatListFullPositions.load());
                str += buf;
            }
            if (validationResults.numClippedThreatWeights > 0)
            {
                snprintf(buf, sizeof(buf), "%*sWARNING: threat weights clipped when packing: %llu\n", indent, "", (unsigned long long)validationResults.numClippedThreatWeights);
                str += buf;
            }

            str += FormatTestPositions(validationResults.testPositionEvals, indent);

            std::cout << str << std::flush;
            m_progressFile.flush();

            lastReportTime = now;
            lastReportPositions = numPositions;
            reportSums = ProgressSums();
        }

        previous = queued;
    }

    finishValidation();

    // the last queued iteration, then the final net
    if (previous)
    {
        m_cudaNetwork.Synchronize();
        const uint32_t previousSlot = lastFinishedIteration != SIZE_MAX ? lastFinishedIteration % nn::cuda::CudaNeuralNetwork::NumIterationSlots : UINT32_MAX;
        m_cudaNetwork.FinishIteration(previous->iteration % nn::cuda::CudaNeuralNetwork::NumIterationSlots, previousSlot);
        if (previous->downloadsWeights)
            m_cudaNetwork.FinishWeightsDownload();

        m_cudaNetwork.CopyWeightsToHost();
        PackNetwork();
        m_packedNet->SaveToFile(OutputPath(".pnn").c_str());
        std::cout << "Saved final net: " << OutputPath(".pnn") << " (" << std::setprecision(4) << previous->numPositions * 1.0e-9 << "B positions)" << std::endl;
    }
    m_progressFile.flush();

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
        else if (args[i] == "--data")
            options.dataPaths.push_back(args[++i]);
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
        else if (args[i] == "--frozenVariants")
        {
            std::stringstream list(args[++i]);
            std::string item;
            while (std::getline(list, item, ','))
            {
                const uint32_t variant = (uint32_t)std::stoul(item);
                if (variant >= nn::NumVariants)
                {
                    std::cerr << "Invalid output variant: " << variant << std::endl;
                    return false;
                }
                options.frozenVariantsMask |= 1u << variant;
            }
        }
        else if (args[i] == "--batchSize")
            options.batchSize = (uint32_t)std::stoul(args[++i]);
        else if (args[i] == "--beta1")
            options.beta1 = std::stof(args[++i]);
        else if (args[i] == "--beta2")
            options.beta2 = std::stof(args[++i]);
        else if (args[i] == "--ftDecay")
            options.featureTransformerWeightDecay = std::stof(args[++i]);
        else if (args[i] == "--factorizer" && (args[i + 1] == "on" || args[i + 1] == "off"))
            options.useFactorizer = args[++i] == "on";
        else
        {
            std::cerr << "Unknown option: " << args[i] << std::endl;
            return false;
        }
    }

    // a frozen variant still receives gradients through the feature transformer and dead neuron revival
    static_assert(nn::NumVariants <= 32, "frozenVariantsMask holds one bit per variant");
    if (options.frozenVariantsMask != 0 && (!options.freezeFeatureTransformer || options.reviveDeadNeurons))
    {
        std::cerr << "--frozenVariants requires --freezeFeatureTransformer and no --reviveDeadNeurons" << std::endl;
        return false;
    }
    if (options.frozenVariantsMask == (uint32_t)((1ull << nn::NumVariants) - 1))
    {
        std::cerr << "All output variants are frozen" << std::endl;
        return false;
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
