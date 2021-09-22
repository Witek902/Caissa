#include "PackedNeuralNetwork.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nn {

bool PackedNeuralNetwork::Load(const char* filePath)
{
    (void)filePath;

    return true;
}

int32_t PackedNeuralNetwork::Run(const uint32_t numActiveInputs, const uint32_t* activeInputIndices) const
{
    (void)numActiveInputs;
    (void)activeInputIndices;

    //alignas(32) int32_t tempValuesA[MaxNeuronsInLayer];
    //alignas(32) int32_t tempValuesB[MaxNeuronsInLayer];

    const int16_t* currLayerWeights = weightsData;

    for (uint32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
    {
        const uint32_t numInputNeurons = numNeuronsInLayer[layerIdx];
        const uint32_t numOutputNeurons = numNeuronsInLayer[layerIdx + 1];
        const uint32_t totalNumWeights = (numInputNeurons + 1u) * numOutputNeurons;

        for (uint32_t outNeuronIdx = 0; outNeuronIdx < numOutputNeurons; ++outNeuronIdx)
        {
            for (uint32_t inNeuronIdx = 0; inNeuronIdx < numInputNeurons; ++inNeuronIdx)
            {

            }
        }

        currLayerWeights += totalNumWeights;
    }

    return 0;
}

} // namespace nn
