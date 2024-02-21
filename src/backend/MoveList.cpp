#include "MoveList.hpp"
#include "Position.hpp"

#include <iomanip>
#include <iostream>

void PrintMoveList(const Position& pos, const MoveList& moves)
{
    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.entries[i].move;

        if (!pos.IsMoveLegal(move)) continue;

        std::cout
            << std::right << std::setw(3) << (i + 1) << ". "
            << move.ToString() << "\t("
            << pos.MoveToString(move, MoveNotation::SAN) << ")\t"
            << moves.entries[i].score;

        if (!pos.StaticExchangeEvaluation(move))
        {
            std::cout << " [negative SSE]";
        }
        std::cout << std::endl;
    }
}
