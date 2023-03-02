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
            if (entries[i].move == ttEntry.moves[j])
            {
                entries[i].score = MoveOrderer::TTMoveValue - j;
                numAssignedMoves++;
                break;
            }
        }
    }

    return numAssignedMoves;
}

void MoveList::Sort()
{
    std::sort(entries, entries + numMoves, [this](const Entry& a, const Entry& b) { return a.score > b.score; });
}

void MoveList::Print(const Position& pos) const
{
    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = entries[i].move;

        if (!pos.IsMoveLegal(move)) continue;

        std::cout
            << std::right << std::setw(3) << (i + 1) << ". "
            << move.ToString() << "\t("
            << pos.MoveToString(move, MoveNotation::SAN) << ")\t"
            << entries[i].score;

        if (!pos.StaticExchangeEvaluation(move))
        {
            std::cout << " [negative SSE]";
        }
        std::cout << std::endl;
    }
}
