#pragma once

#include "CudaCommon.hpp"
#include "CudaWeightsStorage.hpp"
#include "../TrainerCommon.hpp"
#include "../../backend/PackedNeuralNetwork.hpp"

namespace nn {
namespace cuda {

struct CudaBatchData
{
    CudaBuffer<TrainingEntry> trainingVectors;
    CudaBuffer<float> networkOutputs;
    CudaBuffer<float> outputErrors;
    CudaBuffer<float> creluErrors;

    // Intermediate buffers for forward/backward pass
    CudaBuffer<float> accumulatorBuffer;    // For sparse input accumulation
    CudaBuffer<float> hiddenBuffer;         // For hidden layer outputs

    // Temporary buffers for gradients
    CudaBuffer<float> lastLayerGradients;
    CudaBuffer<float> featureTransformerGradients;

    uint32_t batchSize;

    void Allocate(uint32_t size)
    {
        batchSize = size;

        trainingVectors.Allocate(batchSize);
        networkOutputs.Allocate(batchSize);
        outputErrors.Allocate(batchSize);
        creluErrors.Allocate(batchSize * 2 * nn::AccumulatorSize);

        // Allocate intermediate buffers based on network size
        accumulatorBuffer.Allocate(batchSize * nn::AccumulatorSize * 2); // For white and black accumulators
        hiddenBuffer.Allocate(batchSize); // Single output for final layer

        lastLayerGradients.Allocate((2 * nn::AccumulatorSize + 1) * nn::NumVariants);
        featureTransformerGradients.Allocate((nn::NumNetworkInputs + 1) * nn::AccumulatorSize);
    }
};

class CudaNeuralNetwork
{
public:
    CudaNeuralNetwork();
    ~CudaNeuralNetwork();

    void Init(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights);
    void Forward(CudaBatchData& batch);
    void Backward(CudaBatchData& batch, float learningRate, size_t iteration);

    // Set per-layer AdamW weight decay (applied to weights only, not biases).
    void SetWeightDecay(float featureTransformerDecay, float lastLayerDecay);

    // Asynchronously copy a batch's training vectors on a dedicated copy stream. The copy waits
    // for the previous batch's last reader (FeatureTransformerGradientsKernel) so it overlaps the
    // previous batch's Adam updates; Forward waits on it before reading the buffer.
    void CopyTrainingBatchAsync(CudaBatchData& batch, const TrainingEntry* hostSrc, uint32_t count);

    // Weight management
    void CopyWeightsFromHost(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights);
    void CopyWeightsToHost(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights) const;

    const CudaStream& GetStream() const { return m_stream; }

    // Network architecture parameters
    static constexpr uint32_t c_accumulatorSize = nn::AccumulatorSize;
    static constexpr uint32_t c_numNetworkInputs = nn::NumNetworkInputs;
    static constexpr uint32_t c_numVariants = nn::NumVariants;

private:
    // CUDA weight storages
    CudaWeightsStoragePtr m_featureTransformerWeights;
    CudaWeightsStoragePtr m_lastLayerWeights;

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
};

} // namespace cuda
} // namespace nn
