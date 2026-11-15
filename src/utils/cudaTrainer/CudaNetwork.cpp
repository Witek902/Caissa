#include "CudaNetwork.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace nn {
namespace cuda {

namespace {

constexpr uint32_t c_checkpointMagic = 0x43434B50; // 'CCKP'
constexpr uint32_t c_checkpointVersion = 2;

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
    CUDA_CHECK(cudaEventCreateWithFlags(&m_nextTrainConsumedEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_nextCopyDoneEvent, cudaEventDisableTiming));

    // Unlike the synchronization-only events, the iteration markers must keep timing enabled.
    for (uint32_t slot = 0; slot < NumIterationSlots; ++slot)
    {
        CUDA_CHECK(cudaEventCreate(&m_iterationStartEvents[slot]));
        CUDA_CHECK(cudaEventCreate(&m_iterationEndEvents[slot]));
        CUDA_CHECK(cudaEventCreateWithFlags(&m_uploadsDoneEvents[slot], cudaEventDisableTiming));
    }
    m_iterationLossSums.Allocate(NumIterationSlots);

    CUDA_CHECK(cudaEventCreateWithFlags(&m_snapshotReadyEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_weightsDownloadedEvent, cudaEventDisableTiming));

    // Pre-record so the first iteration's cross-stream waits are already satisfied.
    CUDA_CHECK(cudaEventRecord(m_ftGradConsumedEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_ftGradClearedEvent, m_auxStream.Get()));
    CUDA_CHECK(cudaEventRecord(m_trainConsumedEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_copyDoneEvent, m_copyStream.Get()));
    CUDA_CHECK(cudaEventRecord(m_nextTrainConsumedEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_nextCopyDoneEvent, m_copyStream.Get()));
}

CudaNeuralNetwork::~CudaNeuralNetwork()
{
    cudaEventDestroy(m_ftGradConsumedEvent);
    cudaEventDestroy(m_ftGradClearedEvent);
    cudaEventDestroy(m_trainConsumedEvent);
    cudaEventDestroy(m_copyDoneEvent);
    cudaEventDestroy(m_nextTrainConsumedEvent);
    cudaEventDestroy(m_nextCopyDoneEvent);
    for (uint32_t slot = 0; slot < NumIterationSlots; ++slot)
    {
        cudaEventDestroy(m_iterationStartEvents[slot]);
        cudaEventDestroy(m_iterationEndEvents[slot]);
        cudaEventDestroy(m_uploadsDoneEvents[slot]);
    }
    cudaEventDestroy(m_snapshotReadyEvent);
    cudaEventDestroy(m_weightsDownloadedEvent);
}

void CudaNeuralNetwork::BeginIteration(uint32_t slot, CudaBatchData& batch)
{
    ASSERT(slot < NumIterationSlots);
    ASSERT(!m_slotQueued[slot]);
    m_slotQueued[slot] = true;

    CUDA_CHECK(cudaEventRecord(m_iterationStartEvents[slot], m_stream.Get()));
    batch.lossSum.ClearAsync(m_stream.Get());
}

void CudaNeuralNetwork::EndIteration(uint32_t slot, CudaBatchData& batch)
{
    ASSERT(slot < NumIterationSlots);
    ASSERT(m_slotQueued[slot]);

    CUDA_CHECK(cudaMemcpyAsync(m_iterationLossSums.Get() + slot, batch.lossSum.Get(), sizeof(float), cudaMemcpyDeviceToHost, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_iterationEndEvents[slot], m_stream.Get()));

    // the copy stream holds only batch uploads, so this marks the end of the iteration's uploads
    CUDA_CHECK(cudaEventRecord(m_uploadsDoneEvents[slot], m_copyStream.Get()));
}

void CudaNeuralNetwork::WaitForIterationUploads(uint32_t slot) const
{
    ASSERT(slot < NumIterationSlots);
    ASSERT(m_slotQueued[slot]);
    CUDA_CHECK(cudaEventSynchronize(m_uploadsDoneEvents[slot]));
}

CudaNeuralNetwork::IterationResult CudaNeuralNetwork::FinishIteration(uint32_t slot, uint32_t previousSlot)
{
    ASSERT(slot < NumIterationSlots);
    ASSERT(m_slotQueued[slot]);
    ASSERT(previousSlot == UINT32_MAX || (previousSlot < NumIterationSlots && previousSlot != slot && !m_slotQueued[previousSlot]));

    CUDA_CHECK(cudaEventSynchronize(m_iterationEndEvents[slot]));

    IterationResult result;
    result.lossSum = m_iterationLossSums.Get()[slot];
    CUDA_CHECK(cudaEventElapsedTime(&result.gpuMs, m_iterationStartEvents[slot], m_iterationEndEvents[slot]));
    result.periodMs = result.gpuMs;
    if (previousSlot != UINT32_MAX)
        CUDA_CHECK(cudaEventElapsedTime(&result.periodMs, m_iterationEndEvents[previousSlot], m_iterationEndEvents[slot]));
    ASSERT(result.gpuMs >= 0.0f && result.periodMs >= 0.0f);

    m_slotQueued[slot] = false;
    return result;
}

void CudaNeuralNetwork::StartWeightsDownload()
{
    ASSERT(!m_downloadInFlight);
    m_downloadInFlight = true;

    // FinishWeightsDownload waited for the previous download, so the snapshot and the host buffer are free
    size_t offset = 0;
    for (const CudaWeightsStoragePtr& layer : { m_featureTransformerWeights, m_threatWeights, m_l1Weights, m_l2Weights, m_l3Weights })
    {
        if (layer->m_updateWeights)
        {
            CUDA_CHECK(cudaMemcpyAsync(m_weightsSnapshot.Get() + offset, layer->m_weights.Get(),
                layer->m_totalWeights * sizeof(float), cudaMemcpyDeviceToDevice, m_stream.Get()));
        }
        offset += layer->m_totalWeights;
    }
    ASSERT(offset == m_weightsSnapshot.Size() && offset == m_downloadedWeights.Size());
    CUDA_CHECK(cudaEventRecord(m_snapshotReadyEvent, m_stream.Get()));

    CUDA_CHECK(cudaStreamWaitEvent(m_downloadStream.Get(), m_snapshotReadyEvent, 0));
    CUDA_CHECK(cudaMemcpyAsync(m_downloadedWeights.Get(), m_weightsSnapshot.Get(), offset * sizeof(float), cudaMemcpyDeviceToHost, m_downloadStream.Get()));
    CUDA_CHECK(cudaEventRecord(m_weightsDownloadedEvent, m_downloadStream.Get()));
}

void CudaNeuralNetwork::FinishWeightsDownload(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& threatWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    ASSERT(m_downloadInFlight);
    CUDA_CHECK(cudaEventSynchronize(m_weightsDownloadedEvent));

    const CudaWeightsStoragePtr layers[] = { m_featureTransformerWeights, m_threatWeights, m_l1Weights, m_l2Weights, m_l3Weights };
    const nn::WeightsStoragePtr hostLayers[] = { featureTransformerWeights, threatWeights, l1Weights, l2Weights, l3Weights };

    const float* source = m_downloadedWeights.Get();
    for (size_t i = 0; i < std::size(layers); ++i)
    {
        const CudaWeightsStorage& layer = *layers[i];
        if (layer.m_updateWeights)
        {
            const uint32_t weightsPerVariant = (layer.m_inputSize + 1) * layer.m_outputSize;
            ASSERT(weightsPerVariant * layer.m_numVariants == layer.m_totalWeights);
            ASSERT(hostLayers[i]->m_variants.size() == layer.m_numVariants);
            for (uint32_t variant = 0; variant < layer.m_numVariants; ++variant)
            {
                nn::Values& hostWeights = hostLayers[i]->m_variants[variant].m_weights;
                hostWeights.resize(weightsPerVariant);
                memcpy(hostWeights.data(), source + (size_t)variant * weightsPerVariant, weightsPerVariant * sizeof(float));
            }
        }
        source += layer.m_totalWeights;
    }
    ASSERT(source == m_downloadedWeights.Get() + m_downloadedWeights.Size());

    m_downloadInFlight = false;
}

void CudaNeuralNetwork::Synchronize() const
{
    m_stream.Synchronize();
    m_auxStream.Synchronize();
    m_copyStream.Synchronize();
    m_downloadStream.Synchronize();
}

void CudaNeuralNetwork::InitRandomWeights(uint32_t seed)
{
    // scaled so that the accumulator of a ~32 feature position lands in the CReLU range
    m_featureTransformerWeights->Init(seed, 0.1f);

    // the factorizer starts at zero, so the effective weights are just the bucket weights
    CUDA_CHECK(cudaMemset(
        m_featureTransformerWeights->m_weights.Get() + c_numNetworkInputs * c_accumulatorSize,
        0, FactorizerInputs * c_accumulatorSize * sizeof(float)));

    // small, as a position has up to ~60 threat features on top of its piece features
    m_threatWeights->Init(seed + 4, 0.005f);

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
    const nn::WeightsStoragePtr& threatWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    // Create CUDA weight storages based on host network
    m_featureTransformerWeights = std::make_shared<CudaWeightsStorage>(
        FeatureTransformerInputs, c_accumulatorSize, 1
    );
    m_featureTransformerWeights->m_factorizerFirstWeight = c_numNetworkInputs * c_accumulatorSize;
    m_featureTransformerWeights->m_factorizerRange = FactorizerWeightRange;

    // no weight decay: a rarely active threat row would shrink between its updates
    m_threatWeights = std::make_shared<CudaWeightsStorage>(nn::NumThreatInputs, c_accumulatorSize, 1);

    m_l1Weights = std::make_shared<CudaWeightsStorage>(c_l1InputSize, c_l1Size, c_numVariants);
    m_l2Weights = std::make_shared<CudaWeightsStorage>(c_l1Size, c_l2Size, c_numVariants);
    m_l3Weights = std::make_shared<CudaWeightsStorage>(c_l2Size, 1, c_numVariants);

    CopyWeightsFromHost(featureTransformerWeights, threatWeights, l1Weights, l2Weights, l3Weights);

    size_t totalWeights = 0;
    for (const CudaWeightsStoragePtr& layer : { m_featureTransformerWeights, m_threatWeights, m_l1Weights, m_l2Weights, m_l3Weights })
        totalWeights += layer->m_totalWeights;
    m_weightsSnapshot.Allocate(totalWeights);
    m_downloadedWeights.Allocate(totalWeights);

    // Quantization-aware training: weights/biases are fake-quantized on read in the forward/backward
    // kernels using the same scales as the final pack step.
    m_featureTransformerWeights->m_weightQuantScale = nn::InputLayerWeightQuantizationScale;
    m_featureTransformerWeights->m_biasQuantScale = nn::InputLayerBiasQuantizationScale;
    m_threatWeights->m_weightQuantScale = nn::InputLayerWeightQuantizationScale;
    m_threatWeights->m_biasQuantScale = nn::InputLayerBiasQuantizationScale;
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

void CudaNeuralNetwork::SetAdamBetas(float beta1, float beta2)
{
    for (CudaWeightsStorage* weights : { m_featureTransformerWeights.get(), m_threatWeights.get(), m_l1Weights.get(), m_l2Weights.get(), m_l3Weights.get() })
    {
        weights->m_beta1 = beta1;
        weights->m_beta2 = beta2;
    }
}

void CudaNeuralNetwork::SetFeatureTransformerFrozen(bool frozen)
{
    m_featureTransformerWeights->m_updateWeights = !frozen;
    m_threatWeights->m_updateWeights = !frozen;
}

void CudaNeuralNetwork::CopyWeightsFromHost(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& threatWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    m_featureTransformerWeights->CopyFromHost(*featureTransformerWeights);
    m_threatWeights->CopyFromHost(*threatWeights);
    m_l1Weights->CopyFromHost(*l1Weights);
    m_l2Weights->CopyFromHost(*l2Weights);
    m_l3Weights->CopyFromHost(*l3Weights);
}

bool CudaNeuralNetwork::SaveCheckpoint(const char* path, uint64_t numTrainingVectorsPassed) const
{
    const CudaWeightsStoragePtr layers[] = { m_featureTransformerWeights, m_threatWeights, m_l1Weights, m_l2Weights, m_l3Weights };

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
    const CudaWeightsStoragePtr layers[] = { m_featureTransformerWeights, m_threatWeights, m_l1Weights, m_l2Weights, m_l3Weights };

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
    const nn::WeightsStoragePtr& threatWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights) const
{
    m_featureTransformerWeights->CopyToHost(*featureTransformerWeights);
    m_threatWeights->CopyToHost(*threatWeights);
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

    // Clear the FT and threat gradient buffers on a separate stream so it overlaps the forward pass.
    // Wait until the previous iteration's Adam updates finished reading the buffers, clear them, then
    // signal completion for the backward FT-gradient accumulation to wait on.
    CUDA_CHECK(cudaStreamWaitEvent(m_auxStream.Get(), m_ftGradConsumedEvent, 0));
    batch.featureTransformerGradients.ClearAsync(m_auxStream.Get());
    batch.threatGradients.ClearAsync(m_auxStream.Get());
    CUDA_CHECK(cudaEventRecord(m_ftGradClearedEvent, m_auxStream.Get()));

    // The forward pass reads the training vectors; wait for this batch's copy to complete.
    CUDA_CHECK(cudaStreamWaitEvent(stream, m_copyDoneEvent, 0));

    LaunchSparseBinaryInput(batch, GetDeviceLayer(*m_featureTransformerWeights), GetDeviceLayer(*m_threatWeights), m_useFactorizer, stream);
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
        LaunchFeatureTransformerGradients(batch, GetDeviceLayer(*m_l1Weights), m_useFactorizer, stream);
    }

    // The training-vectors buffer is no longer read after the FT gradient accumulation (the Adam
    // updates below don't touch it), so signal that a later batch's copy may overwrite it.
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
    m_threatWeights->UpdateAdam(batch.threatGradients.Get(), learningRate, stream);

    // Signal that the FT and threat gradient buffers are no longer needed on the main stream, so the next
    // iteration's overlapped clear (on the aux stream) may proceed.
    CUDA_CHECK(cudaEventRecord(m_ftGradConsumedEvent, stream));
}

void CudaNeuralNetwork::CopyTrainingBatchAsync(CudaBatchData& batch, const TrainingEntry* entries, uint32_t numEntries,
    const uint16_t* threatFeatures, uint32_t numThreatFeatures)
{
    // a larger batch would reallocate the buffers synchronously
    ASSERT(numEntries <= batch.nextTrainingVectors.Size());
    ASSERT(numThreatFeatures <= batch.nextThreatFeatures.Size());

#ifndef CONFIGURATION_FINAL
    // copies from pageable memory would not be asynchronous
    cudaPointerAttributes attributes;
    ASSERT(cudaPointerGetAttributes(&attributes, entries) == cudaSuccess && attributes.type == cudaMemoryTypeHost);
    ASSERT(cudaPointerGetAttributes(&attributes, threatFeatures) == cudaSuccess && attributes.type == cudaMemoryTypeHost);
#endif // CONFIGURATION_FINAL

    // The last reader of the other buffers is the batch before the current one, so the copy overlaps
    // the whole current batch.
    CUDA_CHECK(cudaStreamWaitEvent(m_copyStream.Get(), m_nextTrainConsumedEvent, 0));
    batch.nextTrainingVectors.CopyFromHost(entries, numEntries, m_copyStream.Get());
    batch.nextThreatFeatures.CopyFromHost(threatFeatures, numThreatFeatures, m_copyStream.Get());
    CUDA_CHECK(cudaEventRecord(m_nextCopyDoneEvent, m_copyStream.Get()));
}

void CudaNeuralNetwork::SwapTrainingBatch(CudaBatchData& batch)
{
    batch.trainingVectors.Swap(batch.nextTrainingVectors);
    batch.threatFeatures.Swap(batch.nextThreatFeatures);

    cudaEvent_t event = m_trainConsumedEvent;
    m_trainConsumedEvent = m_nextTrainConsumedEvent;
    m_nextTrainConsumedEvent = event;

    event = m_copyDoneEvent;
    m_copyDoneEvent = m_nextCopyDoneEvent;
    m_nextCopyDoneEvent = event;
}

} // namespace cuda
} // namespace nn
