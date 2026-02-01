#pragma once

#include "Common.hpp"
#include "net/Network.hpp"
#include "GameCollection.hpp"

#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"

#include <array>

struct PositionEntry
{
    PackedPosition pos;
    ScoreType score = InvalidValue;
    uint8_t wdlScore = 0xFF;
    uint8_t tbScore = 0xFF;
};

struct TrainingEntry
{
    uint8_t variant;
    uint8_t numWhiteFeatures;
    uint8_t numBlackFeatures;
    uint8_t __padding;
    uint16_t whiteFeatures[32];
    uint16_t blackFeatures[32];
    float targetOutput;
};

using TrainingDataSet = std::vector<TrainingEntry>;

class TrainingDataLoader
{
public:

    // initialize the loader at given directory
    bool Init(
        std::mt19937& gen,
        const std::string& trainingDataPath = "../../../data/trainingData");

    // sample new position from the training set
    bool FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition, uint64_t kingBucketMask) const;

private:

    struct InputFileContext
    {
        std::unique_ptr<std::mutex> mutex;
        std::unique_ptr<FileInputStream> fileStream;
        std::string fileName;
        uint64_t fileSize = 0;
        float skippingProbability = 0.0f;

        bool FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition, uint64_t kingBucketMask) const;
    };

    std::vector<InputFileContext> mContexts;

    // cumulative distribution function of picking data from each file
    // (approximation based on file sizes)
    std::vector<double> mCDF;

    uint32_t SampleInputFileIndex(double u) const;
};
