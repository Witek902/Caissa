#include "MoveList.hpp"

#include <algorithm>

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
