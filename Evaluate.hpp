#pragma once

#include "Position.hpp"
#include "Move.hpp"

int32_t ScoreQuietMove(const Position& position, const Move& move);
int32_t Evaluate(const Position& position);
bool CheckInsufficientMaterial(const Position& position);
