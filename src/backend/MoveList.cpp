#include "MoveList.hpp"
#include "TranspositionTable.hpp"
#include "MoveOrderer.hpp"

#include <algorithm>
#include <cstring>
#include <random>

const Move MoveList::AssignTTScores(const TTEntry& ttEntry)
{
    Move ttMove = Move::Invalid();

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
                if (!ttMove.IsValid())
                {
                    ttMove = moves[i].move;
                }
                break;
            }
        }
    }

    return Move::Invalid();
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

void MoveList::Print(bool sorted) const
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
        std::cout << movesCopy[i].move.ToString() << " " << movesCopy[i].score << std::endl;
    }
}
