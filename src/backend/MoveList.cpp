#include "MoveList.hpp"
#include "TranspositionTable.hpp"
#include "MoveOrderer.hpp"
#include "Position.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>

uint32_t MoveList::AssignTTScores(const TTEntry& ttEntry)
{
    uint32_t numAssignedMoves = 0;

    for (uint32_t j = 0; j < TTEntry::NumMoves; ++j)
    {
        if (!ttEntry.moves[j].IsValid())
        {
            continue;
        }

        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i] == ttEntry.moves[j])
            {
                scores[i] = MoveOrderer::TTMoveValue - j;
                numAssignedMoves++;
                break;
            }
        }
    }

    return numAssignedMoves;
}

void MoveList::Sort()
{
    uint8_t indices[MaxMoves];
    Move movesCopy[MaxMoves];
    int32_t scoresCopy[MaxMoves];

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        indices[i] = static_cast<uint8_t>(i);
        movesCopy[i] = moves[i];
        scoresCopy[i] = scores[i];
    }

    std::sort(indices, indices + numMoves, [this](const uint8_t a, const uint8_t b) { return scores[a] > scores[b]; });

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        moves[i] = movesCopy[indices[i]];
        scores[i] = scoresCopy[indices[i]];
    }
}

void MoveList::Print(const Position& pos) const
{
    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i];

        if (!pos.IsMoveLegal(move)) continue;

        std::cout
            << std::right << std::setw(3) << (i + 1) << ". "
            << move.ToString() << "\t("
            << pos.MoveToString(move, MoveNotation::SAN) << ")\t"
            << scores[i];

        if (!pos.StaticExchangeEvaluation(move))
        {
            std::cout << " [negative SSE]";
        }
        std::cout << std::endl;
    }
}
