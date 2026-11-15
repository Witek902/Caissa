#pragma once

#include <cstdint>

// every non-king piece has at most 8 threats, and positions with more than 32 pieces are skipped
static constexpr uint32_t MaxTrainingThreatFeatures = 8 * 30;

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
