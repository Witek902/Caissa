#include "MoveList.hpp"
#include "TranspositionTable.hpp"
#include "MoveOrderer.hpp"
#include "Position.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>

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
