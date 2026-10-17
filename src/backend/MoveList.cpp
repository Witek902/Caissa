#include "MoveList.hpp"
#include "Position.hpp"

#include <stdio.h>

void PrintMoveList(const Position& pos, const MoveList& moves)
{
    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.entries[i].move;

        printf("%3u. ", i + 1);

        if (!pos.IsMoveLegal(move))
        {
            printf("%s\t[illegal move generated]", move.ToString().CStr());
        }
        else
        {
            printf("%s\t(%s)\t%d", move.ToString().CStr(), pos.MoveToString(move, MoveNotation::SAN).CStr(), moves.entries[i].score);

            if (!pos.StaticExchangeEvaluation(move))
            {
                printf(" [negative SSE]");
            }
        }

        printf("\n");
    }
}
