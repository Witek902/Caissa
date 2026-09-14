#pragma once

#include "CudaCommon.hpp"
#include "CudaWeightsStorage.hpp"
#include "../TrainerCommon.hpp"
#include "../../backend/PackedNeuralNetwork.hpp"

// Input factorizer: a shared 768-feature weight block (king-bucket independent) that is added to the
// weights of every king bucket during training and folded into them when the net is packed.
// TODO make it runtime switch instead of compile-time
#define USE_FACTORIZER 1

namespace nn {
namespace cuda {

static constexpr uint32_t FactorizerInputs = 12 * 64;
#if USE_FACTORIZER
static constexpr uint32_t FeatureTransformerInputs = nn::NumNetworkInputs + FactorizerInputs;
#else
static constexpr uint32_t FeatureTransformerInputs = nn::NumNetworkInputs;
#endif // USE_FACTORIZER

// factorizer weights are clipped so that their sum with the (unclipped) bucket weights stays bounded
static constexpr float FactorizerWeightRange = 0.99f;

struct CudaBatchData
{
    CudaBuffer<TrainingEntry> trainingVectors;
    CudaBuffer<float> networkOutputs;       // post-sigmoid output
    CudaBuffer<float> outputErrors;         // dLoss / d(pre-sigmoid output)
    CudaBuffer<float> lossSum;              // sum of squared output errors, accumulated across batches

    // Forward activations. accumulatorBuffer holds the raw (pre-activation) feature transformer
    // output; every later buffer holds the post-activation value.
    CudaBuffer<float> accumulatorBuffer;    // [batch][2][AccumulatorSize]
    CudaBuffer<float> pairwiseBuffer;       // [batch][L1InputSize]
    CudaBuffer<float> l1Buffer;             // [batch][L1Size]
    CudaBuffer<float> l2Buffer;             // [batch][L2Size]
    CudaBuffer<float> l3Buffer;             // [batch] (pre-sigmoid)

    // dLoss / d(pre-activation) of the hidden layers
    CudaBuffer<float> l1PreErrors;          // [batch][L1Size]
    CudaBuffer<float> l2PreErrors;          // [batch][L2Size]

    // Weight gradients. Each must match its weight storage element count exactly,
    // (inputSize + 1) * outputSize * numVariants - the "+1" row holds the per-output biases.
    CudaBuffer<float> l1Gradients;
    CudaBuffer<float> l2Gradients;
    CudaBuffer<float> l3Gradients;
    CudaBuffer<float> featureTransformerGradients;

    uint32_t batchSize;

    void Allocate(uint32_t size)
    {
        batchSize = size;

        trainingVectors.Allocate(batchSize);
        networkOutputs.Allocate(batchSize);
        outputErrors.Allocate(batchSize);
        lossSum.Allocate(1);

        accumulatorBuffer.Allocate(batchSize * 2 * nn::AccumulatorSize);
        pairwiseBuffer.Allocate(batchSize * nn::L1InputSize);
        l1Buffer.Allocate(batchSize * nn::L1Size);
        l2Buffer.Allocate(batchSize * nn::L2Size);
        l3Buffer.Allocate(batchSize);

        l1PreErrors.Allocate(batchSize * nn::L1Size);
        l2PreErrors.Allocate(batchSize * nn::L2Size);

        l1Gradients.Allocate((nn::L1InputSize + 1) * nn::L1Size * nn::NumVariants);
        l2Gradients.Allocate((nn::L1Size + 1) * nn::L2Size * nn::NumVariants);
        l3Gradients.Allocate((nn::L2Size + 1) * 1 * nn::NumVariants);
        featureTransformerGradients.Allocate((FeatureTransformerInputs + 1) * nn::AccumulatorSize);
    }
};

class CudaNeuralNetwork
{
public:
    CudaNeuralNetwork();
    ~CudaNeuralNetwork();

    void Init(const nn::WeightsStoragePtr& featureTransformerWeights,
              const nn::WeightsStoragePtr& l1Weights,
              const nn::WeightsStoragePtr& l2Weights,
              const nn::WeightsStoragePtr& l3Weights);

    // Replace the weights with a random initialization (training from scratch)
    void InitRandomWeights(uint32_t seed);
    // Random initialization of the output subnets only, keeping the feature transformer
    // (used when warm starting the feature transformer from an existing single-layer net)
    void InitRandomOutputSubnetWeights(uint32_t seed);

    void Forward(CudaBatchData& batch);
    void Backward(CudaBatchData& batch, float learningRate);

    // Set per-layer AdamW weight decay (applied to weights only, not biases).
    void SetWeightDecay(float featureTransformerDecay, float outputSubnetDecay);

    // A frozen feature transformer keeps its weights and skips its backward pass, so only the output
    // subnets train.
    void SetFeatureTransformerFrozen(bool frozen);

    // Asynchronously copy a batch's training vectors on a dedicated copy stream. The copy waits
    // for the previous batch's last reader (FeatureTransformerGradientsKernel) so it overlaps the
    // previous batch's Adam updates; Forward waits on it before reading the buffer.
    void CopyTrainingBatchAsync(CudaBatchData& batch, const TrainingEntry* hostSrc, uint32_t count);

    // Weight management
    void CopyWeightsFromHost(const nn::WeightsStoragePtr& featureTransformerWeights,
                             const nn::WeightsStoragePtr& l1Weights,
                             const nn::WeightsStoragePtr& l2Weights,
                             const nn::WeightsStoragePtr& l3Weights);
    void CopyWeightsToHost(const nn::WeightsStoragePtr& featureTransformerWeights,
                           const nn::WeightsStoragePtr& l1Weights,
                           const nn::WeightsStoragePtr& l2Weights,
                           const nn::WeightsStoragePtr& l3Weights) const;

    // Full training state: float master weights plus both Adam moments of every layer, the Adam
    // step counters and the training position count. Unlike a packed net this resumes exactly -
    // no re-snapping onto the quantization grid and no loss of optimizer state.
    bool SaveCheckpoint(const char* path, uint64_t numTrainingVectorsPassed) const;
    bool LoadCheckpoint(const char* path, uint64_t& outNumTrainingVectorsPassed);

    // GPU-time measurement of a training iteration.
    void BeginIterationTiming();
    float EndIterationTimingMs(); // records the end marker and waits for it

    const CudaStream& GetStream() const { return m_stream; }

    // Network architecture parameters
    static constexpr uint32_t c_accumulatorSize = nn::AccumulatorSize;
    static constexpr uint32_t c_numNetworkInputs = nn::NumNetworkInputs;
    static constexpr uint32_t c_numVariants = nn::NumVariants;
    static constexpr uint32_t c_l1InputSize = nn::L1InputSize;
    static constexpr uint32_t c_l1Size = nn::L1Size;
    static constexpr uint32_t c_l2Size = nn::L2Size;

private:
    // CUDA weight storages. The feature transformer has a single variant; each output subnet
    // layer has one per output bucket.
    CudaWeightsStoragePtr m_featureTransformerWeights;
    CudaWeightsStoragePtr m_l1Weights;
    CudaWeightsStoragePtr m_l2Weights;
    CudaWeightsStoragePtr m_l3Weights;

    // CUDA streams for overlapping operations
    CudaStream m_stream;
    CudaStream m_auxStream;  // runs the FT gradient clear concurrently with the forward pass
    CudaStream m_copyStream; // prefetches the next batch's training vectors during weights update

    // Events synchronizing the FT gradient buffer clear (m_auxStream) with its use (m_stream).
    cudaEvent_t m_ftGradConsumedEvent = nullptr; // recorded on m_stream after FT Adam reads the buffer
    cudaEvent_t m_ftGradClearedEvent = nullptr;  // recorded on m_auxStream after the clear completes

    // Events synchronizing the training-vectors copy (m_copyStream) with its use (m_stream).
    cudaEvent_t m_trainConsumedEvent = nullptr; // recorded on m_stream after the last reader (FT gradients)
    cudaEvent_t m_copyDoneEvent = nullptr;      // recorded on m_copyStream after the batch copy completes

    // Timing markers bracketing an iteration's work on the main stream.
    cudaEvent_t m_iterationStartEvent = nullptr;
    cudaEvent_t m_iterationEndEvent = nullptr;
};

} // namespace cuda
} // namespace nn
