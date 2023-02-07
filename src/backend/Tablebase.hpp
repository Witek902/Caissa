#pragma once

#include "Common.hpp"

extern uint32_t g_syzygyProbeLimit;

bool HasSyzygyTablebases();
bool HasGaviotaTablebases();

void LoadSyzygyTablebase(const char* path);
void LoadGaviotaTablebase(const char* path);
void SetGaviotaCacheSize(size_t cacheSize);

void UnloadTablebase();

bool ProbeSyzygy_Root(const Position& pos, Move& outMove, uint32_t* outDTZ = nullptr, int32_t* outWDL = nullptr);
bool ProbeSyzygy_WDL(const Position& pos, int32_t* outWDL);

bool ProbeGaviota(const Position& pos, uint32_t* outDTM = nullptr, int32_t* outWDL = nullptr);
bool ProbeGaviota_Root(const Position& pos, Move& outMove, uint32_t* outDTM = nullptr, int32_t* outWDL = nullptr);