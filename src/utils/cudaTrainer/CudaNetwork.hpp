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

    // FT accumulator: batchSize * 2 * AccumulatorSize floats
    CudaBuffer<float> accumulatorBuffer;

    // Hidden layer outputs (post-activation)
    CudaBuffer<float> l1Buffer;  // batchSize * L1Size
    CudaBuffer<float> l2Buffer;  // batchSize * L2Size
    CudaBuffer<float> l3Buffer;  // batchSize (pre-sigmoid scalar output)

    // Weight/bias gradients per variant
    CudaBuffer<float> l3Gradients;  // (L2Size + 1) * NumVariants
    CudaBuffer<float> l2Gradients;  // (L1Size + 1) * NumVariants
    CudaBuffer<float> l1Gradients;  // (2*AccumulatorSize + 1) * NumVariants

    // Backprop error signals
    CudaBuffer<float> l2PreErrors;  // batchSize * L2Size  (dL/d(L2 pre-activation))
    CudaBuffer<float> l1PreErrors;  // batchSize * L1Size  (dL/d(L1 pre-activation))

    CudaBuffer<float> featureTransformerGradients;  // (NumNetworkInputs + 1) * AccumulatorSize

    uint32_t batchSize;

    void Allocate(uint32_t size)
    {
        batchSize = size;

        trainingVectors.Allocate(batchSize);
        networkOutputs.Allocate(batchSize);
        outputErrors.Allocate(batchSize);
        creluErrors.Allocate(batchSize * 2 * nn::AccumulatorSize);

        accumulatorBuffer.Allocate(batchSize * 2 * nn::AccumulatorSize);
        l1Buffer.Allocate(batchSize * nn::L1Size);
        l2Buffer.Allocate(batchSize * nn::L2Size);
        l3Buffer.Allocate(batchSize);

        // Gradient buffers must match each weight storage's element count, which is
        // (inputSize + 1) * outputSize * numVariants — the "+1" row is the per-output bias.
        // Omitting the outputSize factor (as the single-output last layer used to) leaves these
        // buffers far too small, so the gradient kernels write out of bounds and Adam reads
        // garbage. L3's outputSize is 1, so its size happens to already be correct.
        l3Gradients.Allocate((nn::L2Size + 1) * 1u * nn::NumVariants);
        l2Gradients.Allocate((nn::L1Size + 1) * nn::L2Size * nn::NumVariants);
        l1Gradients.Allocate((2 * nn::AccumulatorSize + 1) * nn::L1Size * nn::NumVariants);

        l2PreErrors.Allocate(batchSize * nn::L2Size);
        l1PreErrors.Allocate(batchSize * nn::L1Size);

        featureTransformerGradients.Allocate((nn::NumNetworkInputs + 1) * nn::AccumulatorSize);
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
    void Forward(CudaBatchData& batch);
    void Backward(CudaBatchData& batch, float learningRate, size_t iteration);

    // Float-net error tracking (sum of squared (output - target) over all batches
    // since the last ResetError). Forward() accumulates into a device buffer; this
    // lets the trainer report the original float-net error and compare it against the
    // quantized-net validation error to catch quantization / CPU-inference bugs.
    void ResetError();
    // Returns the accumulated sum of squared errors (synchronizes the stream).
    double GetAccumulatedSquaredError() const;

    // Weight management
    void CopyWeightsFromHost(const nn::WeightsStoragePtr& featureTransformerWeights,
                             const nn::WeightsStoragePtr& l1Weights,
                             const nn::WeightsStoragePtr& l2Weights,
                             const nn::WeightsStoragePtr& l3Weights);
    void CopyWeightsToHost(const nn::WeightsStoragePtr& featureTransformerWeights,
                           const nn::WeightsStoragePtr& l1Weights,
                           const nn::WeightsStoragePtr& l2Weights,
                           const nn::WeightsStoragePtr& l3Weights) const;

    const CudaStream& GetStream() const { return m_stream; }

    // Network architecture parameters
    static constexpr uint32_t c_accumulatorSize = nn::AccumulatorSize;
    static constexpr uint32_t c_numNetworkInputs = nn::NumNetworkInputs;
    static constexpr uint32_t c_numVariants = nn::NumVariants;

private:
    // CUDA weight storages (FT: 1 variant; L1/L2/L3: NumVariants each)
    CudaWeightsStoragePtr m_featureTransformerWeights;
    CudaWeightsStoragePtr m_l1Weights;
    CudaWeightsStoragePtr m_l2Weights;
    CudaWeightsStoragePtr m_l3Weights;

    // Device-side accumulator (single float) for the sum of squared float-net errors.
    CudaBuffer<float> m_errorAccumulator;

    // CUDA streams for overlapping operations
    CudaStream m_stream;
};

} // namespace cuda
} // namespace nn
