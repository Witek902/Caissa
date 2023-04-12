#include "Common.hpp"
#include "NeuralNetwork.hpp"
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
    Position pos;
    nn::TrainingVector trainingVector;
};

class TrainingDataLoader
{
public:

    // initialize the loader at given directory
    bool Init(const std::string& trainingDataPath = "../../data/trainingData");

    // sample new position from the training set
    bool FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition);

private:

    struct InputFileContext
    {
        std::unique_ptr<FileInputStream> fileStream;
        std::string fileName;
        uint64_t fileSize = 0;

        static constexpr uint32_t BufferSize = 64;
        std::array<PositionEntry, BufferSize> buffer;
        uint32_t bufferOffset = 0;

        bool FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition);
    };

    std::vector<InputFileContext> mContexts;

    // cumulative distribution function of picking data from each file
    // (approximation based on file sizes)
    std::vector<double> mCDF;

    uint32_t SampleInputFileIndex(double u) const;
};
