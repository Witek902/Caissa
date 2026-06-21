#pragma once

#include "CudaCommon.hpp"
#include "../net/WeightsStorage.hpp"

namespace nn {
namespace cuda {

struct CudaWeightsStorage
{
public:
    CudaWeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants);
    ~CudaWeightsStorage();

    void Init(uint32_t numActiveInputs, float bias = 0.0f);

    // Copy weights from host WeightsStorage
    void CopyFromHost(const nn::WeightsStorage& hostWeights);

    // Copy weights to host WeightsStorage
    void CopyToHost(nn::WeightsStorage& hostWeights) const;

    // Update weights using gradients
    void UpdateAdam(const float* gradients, float learningRate, cudaStream_t stream);

    uint32_t m_inputSize = 0;
    uint32_t m_outputSize = 0;
    uint32_t m_numVariants = 0;
    uint32_t m_totalWeights = 0;

    float m_weightsRange = 10.0f;
    float m_biasRange = 10.0f;

    // AdamW decoupled weight decay (applied to weights only, never biases).
    float m_weightDecay = 0.0f;

    // Number of Adam steps performed so far. Used for bias correction; must count actual update
    // steps (one per batch), NOT outer training iterations, or the beta2 correction never saturates.
    size_t m_adamStep = 0;

    // Quantization-aware training: scales used to fake-quantize weights/biases on read in the
    // forward/backward kernels (straight-through estimator; the float master is left untouched).
    float m_weightQuantScale = 0.0f;
    float m_biasQuantScale = 0.0f;

    bool m_updateWeights = true;

    // Device buffers
    CudaBuffer<float> m_weights;
    CudaBuffer<float> m_moment1;  // Adam moment 1
    CudaBuffer<float> m_moment2;  // Adam moment 2

private:
    void AllocateBuffers();
};

using CudaWeightsStoragePtr = std::shared_ptr<CudaWeightsStorage>;

} // namespace cuda
} // namespace nn
