#pragma once

#include "GameCollection.hpp"
#include "TrainingEntry.hpp"

#include <array>
#include <memory>

struct PositionEntry
{
    PackedPosition pos;
    ScoreType score = InvalidValue;
    uint8_t wdlScore = 0xFF;
    uint8_t tbScore = 0xFF;
};

using TrainingDataSet = std::vector<TrainingEntry>;

class TrainingDataLoader
{
public:

    // positions are grouped by pairs of piece counts (kings included)
    static constexpr uint32_t NumPiecePairs = 17;
    using PiecePairKeepProb = std::array<float, NumPiecePairs>;

    // position filters that differ from the default sampling
    struct Options
    {
        // partially skip positions whose decisive score agrees with the game result
        bool decisiveSkip = false;

        // replace the fixed piece-count skip curve with a target piece-count distribution
        bool pieceCountTarget = false;

        // skip by half-move counter only above 20, instead of sqrt(hmc / 100) everywhere
        bool hmcSkipFrom20 = false;
    };

    // initialize the loader at given directory
    bool Init(
        std::mt19937& gen,
        const Options& options,
        const std::string& trainingDataPath = "../../../data/trainingData");

    // sample new position from the training set
    bool FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition, uint64_t kingBucketMask) const;

    const std::string& GetDataPath() const { return mDataPath; }
    size_t GetNumFiles() const { return mContexts.size(); }
    uint64_t GetTotalDataSize() const { return mTotalDataSize; }
    const PiecePairKeepProb& GetPiecePairKeepProb() const { return mPiecePairKeepProb; }

private:

    struct InputFileContext
    {
        std::unique_ptr<std::mutex> mutex;
        std::unique_ptr<FileInputStream> fileStream;
        std::string fileName;
        uint64_t fileSize = 0;
        float skippingProbability = 0.0f;

        bool FetchNextPosition(std::mt19937& gen, const Options& options, const PiecePairKeepProb& piecePairKeepProb, PositionEntry& outEntry, Position& outPosition, uint64_t kingBucketMask) const;
    };

    // measure the piece count distribution left by the other filters and set the keep probabilities that turn it into the target distribution
    void InitPiecePairKeepProb(std::mt19937& gen);

    Options mOptions;
    std::string mDataPath;
    uint64_t mTotalDataSize = 0;

    std::vector<InputFileContext> mContexts;

    PiecePairKeepProb mPiecePairKeepProb;

    // cumulative distribution function of picking data from each file
    // (approximation based on file sizes)
    std::vector<double> mCDF;

    uint32_t SampleInputFileIndex(double u) const;
};
