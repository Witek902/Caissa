#pragma once

#include "Common.hpp"

bool HasTablebases();

void LoadTablebase(const char* path);

void UnloadTablebase();

Move ProbeTablebase_Root(const Position& pos);
