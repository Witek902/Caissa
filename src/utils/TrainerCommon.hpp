#include "Common.hpp"
#include "NeuralNetwork.hpp"

#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"

struct PositionEntry
{
    PackedPosition pos;
    float score;
};

struct TrainingEntry
{
    Position pos;
    nn::TrainingVector trainingVector;
};

void LoadAllPositions(std::vector<PositionEntry>& outEntries);
