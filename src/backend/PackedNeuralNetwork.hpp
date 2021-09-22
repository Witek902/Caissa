#pragma once

#include "Common.hpp"

#include <vector>
#include <cmath>

namespace nn {

class PackedNeuralNetwork
{
public:

    static constexpr uint32_t MaxNeuronsInLayer = 256;
    static constexpr uint32_t MaxNumLayers = 5;

    // load from file
    bool Load(const char* filePath);

    // Calculate neural network output based on input
    int32_t Run(const uint32_t numActiveInputs, const uint32_t* activeInputIndices) const;

private:

    const int16_t* weightsData;
    uint32_t numLayers;
    uint32_t numNeuronsInLayer[MaxNumLayers];
};

} // namespace nn
