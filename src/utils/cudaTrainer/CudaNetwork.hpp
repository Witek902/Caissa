#pragma once

#include "CudaKernels.hpp"
#include "CudaWeightsStorage.hpp"

namespace nn {
namespace cuda {

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

    void Forward(CudaBatchData& batch);
    void Backward(CudaBatchData& batch, float learningRate);

    // Set per-layer AdamW weight decay (applied to weights only, not biases).
    void SetWeightDecay(float featureTransformerDecay, float outputSubnetDecay);

    void SetAdamBetas(float beta1, float beta2);

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
    // Random initialization of the output subnets, part of InitRandomWeights
    void InitRandomOutputSubnetWeights(uint32_t seed);

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
