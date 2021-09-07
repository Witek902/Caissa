#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include "nnue-probe/nnue.h"

bool LoadNeuralNetwork(const char* name);

int32_t ScoreQuietMove(const Position& position, const Move& move);
ScoreType Evaluate(const Position& position);
bool CheckInsufficientMaterial(const Position& position);
