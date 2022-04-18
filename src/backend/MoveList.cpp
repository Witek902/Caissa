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

void MoveList::AssignPVScore(const Move pvMove)
{
    if (!pvMove.IsValid())
    {
        return;
    }

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        if (moves[i].move == pvMove)
        {
            moves[i].score = MoveOrderer::PVMoveValue;
            return;
        }
    }

    ASSERT(!"Invalid PV move");
}

void MoveList::Shuffle()
{
    static std::atomic<uint32_t> shuffleSeed = 0;
    std::shuffle(moves, moves + numMoves, std::default_random_engine(shuffleSeed++));
}

void MoveList::RemoveMove(const Move& move)
{
    for (uint32_t i = 0; i < numMoves; ++i)
    {
        if (moves[i].move == move)
        {
            std::swap(moves[i], moves[numMoves - 1]);
            numMoves--;
            i--;
        }
    }
}

void MoveList::Print(const Position& pos, bool sorted) const
{
    MoveEntry movesCopy[MaxMoves];
    memcpy(movesCopy, moves, sizeof(MoveEntry) * numMoves);

    if (sorted)
    {
        std::sort(movesCopy, movesCopy + numMoves, [](const MoveEntry& a, const MoveEntry& b)
        {
            return a.score > b.score;
        });
    }

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = movesCopy[i].move;

        if (!pos.IsMoveLegal(move)) continue;

        std::cout
            << std::right << std::setw(3) << (i + 1) << ". "
            << move.ToString() << "\t("
            << pos.MoveToString(movesCopy[i].move, MoveNotation::SAN) << ")\t"
            << movesCopy[i].score;

        if (!pos.StaticExchangeEvaluation(move))
        {
            std::cout << " [negative SSE]";
        }
        std::cout << std::endl;
    }
}
