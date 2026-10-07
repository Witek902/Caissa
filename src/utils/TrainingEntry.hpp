#pragma once

#include <cstdint>

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
