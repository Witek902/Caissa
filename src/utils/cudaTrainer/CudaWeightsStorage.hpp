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
    void UpdateAdam(const float* gradients, float learningRate, size_t iteration, cudaStream_t stream);

    uint32_t m_inputSize = 0;
    uint32_t m_outputSize = 0;
    uint32_t m_numVariants = 0;
    uint32_t m_totalWeights = 0;

    bool m_updateWeights = true;

    // Weight/bias clamp ranges, propagated from the host WeightsStorage in CopyFromHost.
    // Default to effectively no clamp until set.
    float m_weightsRange = 1.0e30f;
    float m_biasRange = 1.0e30f;

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
