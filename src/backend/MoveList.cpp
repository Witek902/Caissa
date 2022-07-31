#include "MoveList.hpp"
#include "TranspositionTable.hpp"
#include "MoveOrderer.hpp"
#include "Position.hpp"

#include <algorithm>
#include <cstring>
#include <random>
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
            if (moves[i].move == ttEntry.moves[j])
            {
                moves[i].score = MoveOrderer::TTMoveValue - j;
                numAssignedMoves++;
                break;
            }
        }
    }

    return numAssignedMoves;
}

void MoveList::Shuffle()
{
    static std::atomic<uint32_t> shuffleSeed = 0;
    std::shuffle(moves, moves + numMoves, std::default_random_engine(shuffleSeed++));
}

void MoveList::Print(const Position& pos) const
{
    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i].move;

        if (!pos.IsMoveLegal(move)) continue;

        std::cout
            << std::right << std::setw(3) << (i + 1) << ". "
            << move.ToString() << "\t("
            << pos.MoveToString(moves[i].move, MoveNotation::SAN) << ")\t"
            << moves[i].score;

        if (!pos.StaticExchangeEvaluation(move))
        {
            std::cout << " [negative SSE]";
        }
        std::cout << std::endl;
    }
}
