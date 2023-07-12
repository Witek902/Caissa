#pragma once

#include "Common.hpp"

#include <vector>

// multidimensional Halton sequence generator
// taken from: https://github.com/llxu2017/halton
class HaltonSequence
{
public:
    static constexpr uint32_t Width = 64;

    HaltonSequence();
    ~HaltonSequence();
    void Initialize(uint32_t mDimensions);

    uint32_t GetNumDimensions() const { return mDimensions; }

    void NextSample();
    void NextSampleLeap();

    double GetDouble(uint32_t dimension) { return rnd[dimension][0]; }

private:
    uint64_t Permute(uint32_t i, uint8_t j);

    void ClearPermutation();
    void InitPrimes();
    void InitStart();
    void InitPowerBuffer();
    void InitExpansion();
    void InitPermutation();

    uint32_t mDimensions;
    std::vector<uint64_t> mStarts;
    std::vector<uint32_t> mBase;
    std::vector<std::vector<double>> rnd;
    std::vector<std::vector<uint64_t>> digit;
    std::vector<std::vector<uint64_t>> mPowerBuffer;
    uint64_t **ppm;
};
