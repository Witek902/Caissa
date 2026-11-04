#pragma once

#include <cstdint>

struct TrainingEntry
{
    uint8_t variant;

    // both perspectives see the same pieces, so their feature lists have the same length
    uint8_t numFeatures;

    // Half-move counter training rows active in both perspectives: bit k is row k. Training uses a
    // cumulative encoding: a position in half-move counter bucket b activates rows 0 ... b-1, which are
    // summed into the packed row of bucket b.
    uint16_t hmcInputsMask;

    uint16_t whiteFeatures[32];
    uint16_t blackFeatures[32];
    float targetOutput;
};

static_assert(sizeof(TrainingEntry) == 136, "TrainingEntry is copied to the GPU for every position, keep it small");
