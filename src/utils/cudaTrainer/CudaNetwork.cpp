#include "CudaNetwork.hpp"

#include <iostream>

namespace nn {
namespace cuda {

namespace {

constexpr uint32_t c_checkpointMagic = 0x43434B50; // 'CCKP'
constexpr uint32_t c_checkpointVersion = 1;

struct CheckpointHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t numLayers;
    uint32_t padding;
    uint64_t numTrainingVectorsPassed;
};

// Repeated per layer, so a checkpoint written for a different architecture is rejected instead of
// being read as garbage.
struct CheckpointLayerHeader
{
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t numVariants;
    uint32_t padding;
    uint64_t adamStep;
};

} // namespace

CudaNeuralNetwork::CudaNeuralNetwork()
{
    CUDA_CHECK(cudaEventCreateWithFlags(&m_ftGradConsumedEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_ftGradClearedEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_trainConsumedEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_copyDoneEvent, cudaEventDisableTiming));

    // Unlike the synchronization-only events above, these must keep timing enabled.
    CUDA_CHECK(cudaEventCreate(&m_iterationStartEvent));
    CUDA_CHECK(cudaEventCreate(&m_iterationEndEvent));

    // Pre-record so the first iteration's cross-stream waits are already satisfied.
    CUDA_CHECK(cudaEventRecord(m_ftGradConsumedEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_ftGradClearedEvent, m_auxStream.Get()));
    CUDA_CHECK(cudaEventRecord(m_trainConsumedEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_copyDoneEvent, m_copyStream.Get()));
}

CudaNeuralNetwork::~CudaNeuralNetwork()
{
    cudaEventDestroy(m_ftGradConsumedEvent);
    cudaEventDestroy(m_ftGradClearedEvent);
    cudaEventDestroy(m_trainConsumedEvent);
    cudaEventDestroy(m_copyDoneEvent);
    cudaEventDestroy(m_iterationStartEvent);
    cudaEventDestroy(m_iterationEndEvent);
}

void CudaNeuralNetwork::BeginIterationTiming()
{
    CUDA_CHECK(cudaEventRecord(m_iterationStartEvent, m_stream.Get()));
}

float CudaNeuralNetwork::EndIterationTimingMs()
{
    CUDA_CHECK(cudaEventRecord(m_iterationEndEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventSynchronize(m_iterationEndEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, m_iterationStartEvent, m_iterationEndEvent));
    return elapsedMs;
}

void CudaNeuralNetwork::InitRandomWeights(uint32_t seed)
{
    // scaled so that the accumulator of a ~32 feature position lands in the CReLU range
    m_featureTransformerWeights->Init(seed, 0.1f);

#if USE_FACTORIZER
    // the factorizer starts at zero, so the effective weights are just the bucket weights
    CUDA_CHECK(cudaMemset(
        m_featureTransformerWeights->m_weights.Get() + c_numNetworkInputs * c_accumulatorSize,
        0, FactorizerInputs * c_accumulatorSize * sizeof(float)));
#endif // USE_FACTORIZER

    InitRandomOutputSubnetWeights(seed);
}

void CudaNeuralNetwork::InitRandomOutputSubnetWeights(uint32_t seed)
{
    // He initialization, appropriate for the (clipped) ReLU activations of the output subnet
    m_l1Weights->Init(seed + 1, sqrtf(2.0f / (float)c_l1InputSize));
    m_l2Weights->Init(seed + 2, sqrtf(2.0f / (float)c_l1Size));
    m_l3Weights->Init(seed + 3, sqrtf(2.0f / (float)c_l2Size));
}

void CudaNeuralNetwork::Init(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    // Create CUDA weight storages based on host network
    m_featureTransformerWeights = std::make_shared<CudaWeightsStorage>(
        FeatureTransformerInputs, c_accumulatorSize, 1
    );
#if USE_FACTORIZER
    m_featureTransformerWeights->m_factorizerFirstWeight = c_numNetworkInputs * c_accumulatorSize;
    m_featureTransformerWeights->m_factorizerRange = FactorizerWeightRange;
#endif // USE_FACTORIZER

    m_l1Weights = std::make_shared<CudaWeightsStorage>(c_l1InputSize, c_l1Size, c_numVariants);
    m_l2Weights = std::make_shared<CudaWeightsStorage>(c_l1Size, c_l2Size, c_numVariants);
    m_l3Weights = std::make_shared<CudaWeightsStorage>(c_l2Size, 1, c_numVariants);

    CopyWeightsFromHost(featureTransformerWeights, l1Weights, l2Weights, l3Weights);

    // Quantization-aware training: weights/biases are fake-quantized on read in the forward/backward
    // kernels using the same scales as the final pack step.
    m_featureTransformerWeights->m_weightQuantScale = nn::InputLayerWeightQuantizationScale;
    m_featureTransformerWeights->m_biasQuantScale = nn::InputLayerBiasQuantizationScale;
    m_l1Weights->m_weightQuantScale = nn::HiddenLayerWeightQuantizationScale;
    m_l1Weights->m_biasQuantScale = nn::HiddenLayerBiasQuantizationScale;
    m_l2Weights->m_weightQuantScale = nn::HiddenLayerWeightQuantizationScale;
    m_l2Weights->m_biasQuantScale = nn::HiddenLayerBiasQuantizationScale;
    m_l3Weights->m_weightQuantScale = nn::OutputLayerWeightQuantizationScale;
    m_l3Weights->m_biasQuantScale = nn::OutputLayerBiasQuantizationScale;
}

void CudaNeuralNetwork::SetWeightDecay(float featureTransformerDecay, float outputSubnetDecay)
{
    m_featureTransformerWeights->m_weightDecay = featureTransformerDecay;
    m_l1Weights->m_weightDecay = outputSubnetDecay;
    m_l2Weights->m_weightDecay = outputSubnetDecay;
    m_l3Weights->m_weightDecay = outputSubnetDecay;
}

void CudaNeuralNetwork::SetFeatureTransformerFrozen(bool frozen)
{
    m_featureTransformerWeights->m_updateWeights = !frozen;
}

void CudaNeuralNetwork::CopyWeightsFromHost(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    m_featureTransformerWeights->CopyFromHost(*featureTransformerWeights);
    m_l1Weights->CopyFromHost(*l1Weights);
    m_l2Weights->CopyFromHost(*l2Weights);
    m_l3Weights->CopyFromHost(*l3Weights);
}

bool CudaNeuralNetwork::SaveCheckpoint(const char* path, uint64_t numTrainingVectorsPassed) const
{
    const CudaWeightsStoragePtr layers[] = { m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights };

    FILE* file = fopen(path, "wb");
    if (!file)
    {
        std::cerr << "Failed to save checkpoint: cannot open " << path << std::endl;
        return false;
    }

    CheckpointHeader header{};
    header.magic = c_checkpointMagic;
    header.version = c_checkpointVersion;
    header.numLayers = (uint32_t)std::size(layers);
    header.numTrainingVectorsPassed = numTrainingVectorsPassed;

    bool ok = fwrite(&header, sizeof(header), 1, file) == 1;

    std::vector<float> weights, moment1, moment2;
    for (const CudaWeightsStoragePtr& layer : layers)
    {
        if (!ok) break;

        CheckpointLayerHeader layerHeader{};
        layerHeader.inputSize = layer->m_inputSize;
        layerHeader.outputSize = layer->m_outputSize;
        layerHeader.numVariants = layer->m_numVariants;
        layerHeader.adamStep = layer->m_adamStep;

        layer->CopyStateToHost(weights, moment1, moment2);

        ok = fwrite(&layerHeader, sizeof(layerHeader), 1, file) == 1
            && fwrite(weights.data(), sizeof(float), weights.size(), file) == weights.size()
            && fwrite(moment1.data(), sizeof(float), moment1.size(), file) == moment1.size()
            && fwrite(moment2.data(), sizeof(float), moment2.size(), file) == moment2.size();
    }

    fclose(file);

    if (!ok)
        std::cerr << "Failed to save checkpoint: cannot write " << path << std::endl;

    return ok;
}

bool CudaNeuralNetwork::LoadCheckpoint(const char* path, uint64_t& outNumTrainingVectorsPassed)
{
    const CudaWeightsStoragePtr layers[] = { m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights };

    FILE* file = fopen(path, "rb");
    if (!file)
    {
        std::cerr << "Failed to load checkpoint: cannot open " << path << std::endl;
        return false;
    }

    CheckpointHeader header{};
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        header.magic != c_checkpointMagic ||
        header.version != c_checkpointVersion ||
        header.numLayers != (uint32_t)std::size(layers))
    {
        fclose(file);
        std::cerr << "Failed to load checkpoint: " << path << " is not a compatible checkpoint" << std::endl;
        return false;
    }

    std::vector<float> weights, moment1, moment2;
    for (const CudaWeightsStoragePtr& layer : layers)
    {
        CheckpointLayerHeader layerHeader{};
        if (fread(&layerHeader, sizeof(layerHeader), 1, file) != 1 ||
            layerHeader.inputSize != layer->m_inputSize ||
            layerHeader.outputSize != layer->m_outputSize ||
            layerHeader.numVariants != layer->m_numVariants)
        {
            fclose(file);
            std::cerr << "Failed to load checkpoint: " << path << " was written for a different architecture" << std::endl;
            return false;
        }

        weights.resize(layer->m_totalWeights);
        moment1.resize(layer->m_totalWeights);
        moment2.resize(layer->m_totalWeights);

        const size_t count = layer->m_totalWeights;
        if (fread(weights.data(), sizeof(float), count, file) != count ||
            fread(moment1.data(), sizeof(float), count, file) != count ||
            fread(moment2.data(), sizeof(float), count, file) != count)
        {
            fclose(file);
            std::cerr << "Failed to load checkpoint: " << path << " is truncated" << std::endl;
            return false;
        }

        layer->CopyStateFromHost(weights, moment1, moment2);
        layer->m_adamStep = (size_t)layerHeader.adamStep;
    }

    fclose(file);
    outNumTrainingVectorsPassed = header.numTrainingVectorsPassed;
    return true;
}

void CudaNeuralNetwork::CopyWeightsToHost(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights) const
{
    m_featureTransformerWeights->CopyToHost(*featureTransformerWeights);
    m_l1Weights->CopyToHost(*l1Weights);
    m_l2Weights->CopyToHost(*l2Weights);
    m_l3Weights->CopyToHost(*l3Weights);
}

static DeviceLayer GetDeviceLayer(const CudaWeightsStorage& weights)
{
    return
    {
        weights.m_weights.Get(),
        { weights.m_weightQuantScale, weights.m_biasQuantScale, 1.0f / weights.m_weightQuantScale, 1.0f / weights.m_biasQuantScale }
    };
}

void CudaNeuralNetwork::Forward(CudaBatchData& batch)
{
    const cudaStream_t stream = m_stream.Get();

    // Clear the FT gradient buffer on a separate stream so it overlaps the forward pass.
    // Wait until the previous iteration's FT Adam finished reading the buffer, clear it, then
    // signal completion for the backward FT-gradient accumulation to wait on.
    CUDA_CHECK(cudaStreamWaitEvent(m_auxStream.Get(), m_ftGradConsumedEvent, 0));
    batch.featureTransformerGradients.ClearAsync(m_auxStream.Get());
    CUDA_CHECK(cudaEventRecord(m_ftGradClearedEvent, m_auxStream.Get()));

    // The forward pass reads the training vectors; wait for this batch's copy to complete.
    CUDA_CHECK(cudaStreamWaitEvent(stream, m_copyDoneEvent, 0));

    LaunchSparseBinaryInput(batch, GetDeviceLayer(*m_featureTransformerWeights), stream);
    LaunchL1Forward(batch, GetDeviceLayer(*m_l1Weights), stream);
    LaunchL2Forward(batch, GetDeviceLayer(*m_l2Weights), stream);
    LaunchL3Forward(batch, GetDeviceLayer(*m_l3Weights), stream);
    LaunchSigmoidActivation(batch, stream);
}

void CudaNeuralNetwork::Backward(CudaBatchData& batch, float learningRate)
{
    const cudaStream_t stream = m_stream.Get();

    // The dense gradient kernels accumulate atomically across batch slices
    batch.l1Gradients.ClearAsync(stream);
    batch.l2Gradients.ClearAsync(stream);
    batch.l3Gradients.ClearAsync(stream);

    LaunchSigmoidDerivative(batch, stream);
    LaunchL3Backward(batch, GetDeviceLayer(*m_l3Weights), stream);
    LaunchL2Backward(batch, GetDeviceLayer(*m_l2Weights), stream);
    LaunchL1Gradients(batch, stream);

    // The L1 -> feature transformer backprop and the FT weight gradients are only needed to update
    // the feature transformer, so both are skipped when it is frozen.
    if (m_featureTransformerWeights->m_updateWeights)
    {
        // The buffer was cleared on the aux stream (overlapping the forward pass); wait for that
        // clear to complete before accumulating.
        CUDA_CHECK(cudaStreamWaitEvent(stream, m_ftGradClearedEvent, 0));
        LaunchFeatureTransformerGradients(batch, GetDeviceLayer(*m_l1Weights), stream);
    }

    // The training-vectors buffer is no longer read after the FT gradient accumulation (the Adam
    // updates below don't touch it), so signal that the next batch's copy may overwrite it.
    CUDA_CHECK(cudaEventRecord(m_trainConsumedEvent, stream));

    // Update output subnet weights
    m_l3Weights->UpdateAdam(batch.l3Gradients.Get(), learningRate, stream);
    m_l2Weights->UpdateAdam(batch.l2Gradients.Get(), learningRate, stream);
    m_l1Weights->UpdateAdam(batch.l1Gradients.Get(), learningRate, stream);

    // Update feature transformer weights
    m_featureTransformerWeights->UpdateAdam(
        batch.featureTransformerGradients.Get(),
        learningRate,
        stream
    );

    // Signal that the FT gradient buffer is no longer needed on the main stream, so the next
    // iteration's overlapped clear (on the aux stream) may proceed.
    CUDA_CHECK(cudaEventRecord(m_ftGradConsumedEvent, stream));
}

void CudaNeuralNetwork::CopyTrainingBatchAsync(CudaBatchData& batch, const TrainingEntry* hostSrc, uint32_t count)
{
    // Wait until the previous batch's last reader of the training-vectors buffer
    // (FeatureTransformerGradientsKernel) has finished, then copy on the dedicated copy stream so
    // it overlaps the previous batch's Adam updates. Forward waits on m_copyDoneEvent before use.
    CUDA_CHECK(cudaStreamWaitEvent(m_copyStream.Get(), m_trainConsumedEvent, 0));
    batch.trainingVectors.CopyFromHost(hostSrc, count, m_copyStream.Get());
    CUDA_CHECK(cudaEventRecord(m_copyDoneEvent, m_copyStream.Get()));
}

} // namespace cuda
} // namespace nn
