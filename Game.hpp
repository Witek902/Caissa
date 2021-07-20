#pragma once

#include "Position.hpp"

#include <vector>
#include <unordered_map>

// represents whole game
class Game
{
public:
    const Position& GetInitialPosition() const { return mInitPosition; }
    const Position& GetPosition() const { return mPosition; }
    Color GetSideToMove() const { return mPosition.GetSideToMove(); }

    const std::vector<Move>& GetMoves() const { return mMoves; }

    void Reset(const Position& pos);

    bool DoMove(const Move& move);

    uint32_t GetRepetitionCount(const Position& position) const;

    bool IsDrawn() const;

    // convert moves list to PGN string
    std::string ToPGN() const;

private:

    void RecordBoardPosition(const Position& position);

    Position mInitPosition;
    Position mPosition;
    std::vector<Move> mMoves;

    // store some simplified position state instead of full struct
    using HistoryPosition = std::pair<Position, uint32_t>;
    using HistoryPositions = std::vector<HistoryPosition>;
    std::unordered_map<uint64_t, HistoryPositions> mHistoryGamePositions;
};