#include "../backend/PackedNeuralNetwork.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

// Converts a version 15 net (no half-move counter inputs) to the current version. The new half-move
// counter rows are zero, so the converted net evaluates every position exactly like the original.

bool UpgradeNet(const std::vector<std::string>& args)
{
    if (args.size() != 2)
    {
        std::cerr << "Usage: utils upgradeNet <input-v15.pnn> <output.pnn>" << std::endl;
        return false;
    }

    std::ifstream file(args[0], std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open " << args[0] << std::endl;
        return false;
    }
    const std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    constexpr size_t headerSize = sizeof(nn::PackedNeuralNetwork::Header);
    constexpr size_t weightsSize = sizeof(nn::FirstLayerWeightType) * nn::NumPieceInputs * nn::AccumulatorSize;
    constexpr size_t biasesSize = sizeof(nn::FirstLayerBiasType) * nn::AccumulatorSize;
    constexpr size_t subnetsSize = sizeof(nn::PackedNeuralNetwork::OutputSubnetVariant) * nn::NumVariants;

    // version 15 layout: header, piece input weights, biases, output subnets (padded to a cache line)
    const size_t subnetsOffset = (headerSize + weightsSize + biasesSize + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;

    nn::PackedNeuralNetwork::Header header;
    if (data.size() < headerSize)
    {
        std::cerr << args[0] << " is not a network file" << std::endl;
        return false;
    }
    memcpy(&header, data.data(), headerSize);

    if (header.magic != nn::MagicNumber || header.version != 15 || data.size() != subnetsOffset + subnetsSize)
    {
        std::cerr << args[0] << " is not a version 15 network file" << std::endl;
        return false;
    }

    auto net = std::make_unique<nn::PackedNeuralNetwork>();
    memcpy(net->accumulatorWeights, data.data() + headerSize, weightsSize);
    memset(net->accumulatorWeights + nn::NumPieceInputs * nn::AccumulatorSize, 0,
        sizeof(nn::FirstLayerWeightType) * nn::NumHmcInputs * nn::AccumulatorSize);
    memcpy(net->accumulatorBiases, data.data() + headerSize + weightsSize, biasesSize);
    memcpy(net->outputSubnetVariants, data.data() + subnetsOffset, subnetsSize);

    if (!net->SaveToFile(args[1].c_str()))
        return false;

    std::cout << "Saved " << args[1] << std::endl;
    return true;
}
