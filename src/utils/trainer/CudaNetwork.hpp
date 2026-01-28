#pragma once

#include "CudaCommon.hpp"
#include "CudaWeightsStorage.hpp"
#include "../../backend/PackedNeuralNetwork.hpp"

namespace nn {
namespace cuda {

struct CudaTrainingVector
{
    uint32_t variant;
    uint16_t whiteFeatures[32];  // Max features per position
    uint16_t blackFeatures[32];
    uint32_t numWhiteFeatures;
    uint32_t numBlackFeatures;
    float targetOutput;
};

struct CudaBatchData
{
    CudaBuffer<CudaTrainingVector> trainingVectors;
    CudaBuffer<float> networkOutputs;
    CudaBuffer<float> outputErrors;
    CudaBuffer<float> creluErrors;

    // Intermediate buffers for forward/backward pass
    CudaBuffer<float> accumulatorBuffer;    // For sparse input accumulation
    CudaBuffer<float> hiddenBuffer;         // For hidden layer outputs
    CudaBuffer<float> activationBuffer;     // For activation outputs

    // Temporary buffers for gradients
    CudaBuffer<double> lastLayerGradients;
    CudaBuffer<double> featureTransformerGradients;

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
        activationBuffer.Allocate(batchSize * nn::AccumulatorSize * 2);

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
    void Backward(CudaBatchData& batch, float learningRate, float weightDecay, size_t iteration);

    // Weight management
    void CopyWeightsFromHost(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights);
    void CopyWeightsToHost(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights) const;

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
};

} // namespace cuda
} // namespace nn
