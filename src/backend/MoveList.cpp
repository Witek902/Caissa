#include "MoveList.hpp"
#include "Position.hpp"

#include <iomanip>

void PrintMoveList(const Position& pos, const MoveList& moves)
{
    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.entries[i].move;

        std::cout << std::right << std::setw(3) << (i + 1) << ". ";

        if (!pos.IsMoveLegal(move))
        {
            std::cout << move.ToString() << "\t[illegal move generated]";
        }
        else
        {
            std::cout
                << move.ToString() << "\t("
                << pos.MoveToString(move, MoveNotation::SAN) << ")\t"
                << moves.entries[i].score;

            if (!pos.StaticExchangeEvaluation(move))
            {
                std::cout << " [negative SSE]";
            }
        }

        std::cout << std::endl;
    }
}
