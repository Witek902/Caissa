#pragma once

#include "Common.hpp"

bool HasTablebases();

void LoadTablebase(const char* path);

void UnloadTablebase();

bool ProbeTablebase_Root(const Position& pos, Move& outMove, uint32_t* outDistanceToZero = nullptr, int32_t* outWDL = nullptr);
bool ProbeTablebase_WDL(const Position& pos, int32_t* outWDL);
