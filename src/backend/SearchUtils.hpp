#pragma once

#include "Common.hpp"

class SearchUtils
{
public:
    static void Init();

    // check if the search node has a move that draws by repetition
    // or a past position could directly reach the current position
    static bool CanReachGameCycle(const NodeInfo& node);

    // reconstruct PV line from cache, outLine must have space for MaxSearchDepth moves, returns PV length
    static uint32_t GetPvLine(const NodeInfo& rootNode, uint32_t maxLength, Move* outLine);
};
