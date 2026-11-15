#pragma once

#include <cstdint>

// every non-king piece has at most 8 threats, and positions with more than 32 pieces are skipped
static constexpr uint32_t MaxTrainingThreatFeatures = 8 * 30;

// Threat feature slots per position in a batch's threat feature list (both perspectives). A batch
// overflows only if its positions average more than 48 threat features per perspective; training
// positions average 15.
static constexpr uint32_t TrainingThreatFeatureBudget = 2 * 48;

struct TrainingEntry
{
    uint8_t variant;
    uint8_t numWhiteFeatures;
    uint8_t numBlackFeatures;
    uint8_t numThreatFeatures;      // the same for both perspectives
    uint32_t threatFeaturesOffset;  // in the batch's threat feature list: white features, then black features
    uint16_t whiteFeatures[32];
    uint16_t blackFeatures[32];
    float targetOutput;
};

static_assert(sizeof(TrainingEntry) == 140, "TrainingEntry must not contain padding");
