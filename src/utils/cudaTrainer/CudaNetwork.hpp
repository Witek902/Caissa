#pragma once

#include "CudaKernels.hpp"
#include "CudaWeightsStorage.hpp"

#include <array>

namespace nn {
namespace cuda {

class CudaNeuralNetwork
{
public:
    CudaNeuralNetwork();
    ~CudaNeuralNetwork();

    // The host storages receive every weight download. Their memory is pinned until destruction, so they
    // must not be reallocated.
    void Init(const nn::WeightsStoragePtr& featureTransformerWeights,
              const nn::WeightsStoragePtr& threatWeights,
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

    // A frozen feature transformer (piece and threat weights) keeps its weights and skips its
    // backward pass, so only the output subnets train.
    void SetFeatureTransformerFrozen(bool frozen);

    // The factorizer weights are added to the piece weights only when enabled
    void SetFactorizerEnabled(bool enabled) { m_useFactorizer = enabled; }

    // Asynchronously copy a batch's training vectors and threat features into nextTrainingVectors and
    // nextThreatFeatures on a dedicated copy stream. The copy waits for the last reader of those buffers
    // (FeatureTransformerGradientsKernel of the batch before the current one), so it overlaps the whole
    // current batch.
    void CopyTrainingBatchAsync(CudaBatchData& batch, const TrainingEntry* entries, uint32_t numEntries,
        const uint16_t* threatFeatures, uint32_t numThreatFeatures);

    // Make the copied batch the current one; Forward waits for its copy before reading it.
    void SwapTrainingBatch(CudaBatchData& batch);

    // Synchronous copies between the device weights and the host storages; frozen layers keep their
    // host copy
    void CopyWeightsFromHost();
    void CopyWeightsToHost() const;

    // Full training state: float master weights plus both Adam moments of every layer, the Adam
    // step counters and the training position count. Unlike a packed net this resumes exactly -
    // no re-snapping onto the quantization grid and no loss of optimizer state.
    bool SaveCheckpoint(const char* path, uint64_t numTrainingVectorsPassed) const;
    bool LoadCheckpoint(const char* path, uint64_t& outNumTrainingVectorsPassed);

    // Training iterations are queued without waiting for the previous one. A queued iteration keeps its
    // events and loss in one of NumIterationSlots slots until FinishIteration reads them.
    static constexpr uint32_t NumIterationSlots = 3;

    struct IterationResult
    {
        float lossSum;      // sum of squared output errors over the iteration
        float gpuMs;        // from the GPU reaching the iteration to its end
        float periodMs;     // from the end of the previous iteration to the end of this one
    };

    void BeginIteration(uint32_t slot, CudaBatchData& batch);
    // after the iteration's last batch has been queued
    void EndIteration(uint32_t slot, CudaBatchData& batch);
    // waits until the iteration's batch uploads no longer read host memory
    void WaitForIterationUploads(uint32_t slot) const;
    // waits until the iteration is done; previousSlot holds the finished iteration before it, or UINT32_MAX
    IterationResult FinishIteration(uint32_t slot, uint32_t previousSlot);

    // Asynchronous weight download into the host storages, at most one at a time: the weights after the
    // queued work are copied on the GPU, then downloaded on a separate stream while training continues.
    // The host storages must not be used until FinishWeightsDownload returns.
    void StartWeightsDownload();
    void FinishWeightsDownload();

    // waits until all queued work is done
    void Synchronize() const;

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

    // CUDA weight storages. The feature transformer and the threat weights have a single variant;
    // each output subnet layer has one per output bucket.
    CudaWeightsStoragePtr m_featureTransformerWeights;
    CudaWeightsStoragePtr m_threatWeights;
    CudaWeightsStoragePtr m_l1Weights;
    CudaWeightsStoragePtr m_l2Weights;
    CudaWeightsStoragePtr m_l3Weights;

    static constexpr uint32_t NumLayers = 5;

    // device layers in a fixed order: feature transformer, threats, L1, L2, L3
    std::array<CudaWeightsStorage*, NumLayers> Layers() const
    {
        return { m_featureTransformerWeights.get(), m_threatWeights.get(), m_l1Weights.get(), m_l2Weights.get(), m_l3Weights.get() };
    }

    // host storages, in the order of Layers()
    nn::WeightsStoragePtr m_hostLayers[NumLayers];

    bool m_useFactorizer = true;

    // CUDA streams for overlapping operations
    CudaStream m_stream;
    CudaStream m_auxStream;  // runs the FT gradient clear concurrently with the forward pass
    CudaStream m_copyStream; // uploads the next batch while the current one trains

    // Events synchronizing the FT and threat gradient buffer clears (m_auxStream) with their use (m_stream).
    cudaEvent_t m_ftGradConsumedEvent = nullptr; // recorded on m_stream after FT and threat Adam read the buffers
    cudaEvent_t m_ftGradClearedEvent = nullptr;  // recorded on m_auxStream after the clear completes

    // Events synchronizing the training-vectors copies (m_copyStream) with their use (m_stream), one pair
    // per buffer, swapped together with trainingVectors and nextTrainingVectors.
    cudaEvent_t m_trainConsumedEvent = nullptr; // recorded on m_stream after the last reader (FT gradients)
    cudaEvent_t m_copyDoneEvent = nullptr;      // recorded on m_copyStream after the batch copy completes
    cudaEvent_t m_nextTrainConsumedEvent = nullptr;
    cudaEvent_t m_nextCopyDoneEvent = nullptr;

    // Per iteration slot: markers bracketing the iteration on the main stream, the end of its uploads on
    // the copy stream, and its loss sum downloaded at its end.
    cudaEvent_t m_iterationStartEvents[NumIterationSlots] = {};
    cudaEvent_t m_iterationEndEvents[NumIterationSlots] = {};
    cudaEvent_t m_uploadsDoneEvents[NumIterationSlots] = {};
    PinnedBuffer<float> m_iterationLossSums;
    bool m_slotQueued[NumIterationSlots] = {};

    // Weight snapshot (all layers back to back) and its download
    CudaStream m_downloadStream;
    CudaBuffer<float> m_weightsSnapshot;
    cudaEvent_t m_snapshotReadyEvent = nullptr;     // recorded on m_stream after the snapshot copies
    cudaEvent_t m_weightsDownloadedEvent = nullptr; // recorded on m_downloadStream after the download
    bool m_downloadInFlight = false;
};

} // namespace cuda
} // namespace nn
