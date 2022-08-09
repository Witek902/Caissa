#pragma once

#include <vector>

#include "Common.hpp"

class SearchUtils
{
public:
    static void Init();

    // check for repetition in the searched node
    static bool IsRepetition(const NodeInfo& node, const Game& game);

    // check if the search node has a move that draws by repetition
    // or a past position could directly reach the current position
    static bool CanReachGameCycle(const NodeInfo& node);

    // reconstruct PV line from cache
    static void GetPvLine(const NodeInfo& rootNode, uint32_t maxLength, std::vector<Move>& outLine);
};
