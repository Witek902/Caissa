#pragma once

#include <cstdint>

// threat features stored per position; the trainer skips positions with more
static constexpr uint32_t MaxTrainingThreatFeatures = 96;

struct TrainingEntry
{
    uint8_t variant;
    uint8_t numWhiteFeatures;
    uint8_t numBlackFeatures;
    uint8_t numThreatFeatures; // the same for both perspectives
    uint16_t whiteFeatures[32];
    uint16_t blackFeatures[32];
    uint16_t whiteThreatFeatures[MaxTrainingThreatFeatures];
    uint16_t blackThreatFeatures[MaxTrainingThreatFeatures];
    float targetOutput;
};

static_assert(sizeof(TrainingEntry) == 136 + 4 * MaxTrainingThreatFeatures, "TrainingEntry must not contain padding");
