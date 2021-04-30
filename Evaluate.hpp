#pragma once

#include "Position.hpp"
#include "Move.hpp"

int32_t ScoreQuietMove(const Move& move, const Color color);
int32_t Evaluate(const Position& position);
bool CheckInsufficientMaterial(const Position& position);
