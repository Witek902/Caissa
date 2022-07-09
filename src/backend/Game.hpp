#pragma once

#include "Position.hpp"

#include <vector>
#include <unordered_map>

namespace std {

template<>
struct hash<Position>
{
    size_t operator()(const Position& k) const
    {
        return static_cast<size_t>(k.GetHash());
    }
};

} // std

struct GameMetadata
{
    uint32_t roundNumber = 1;
};

// represents whole game
class Game
{
public:
    enum class Score : uint8_t
    {
        Draw        = 0,
        WhiteWins   = 1,
        BlackWins   = 2,
        Unknown     = 0xFF,
    };

    Game();

    const Position& GetInitialPosition() const { return mInitPosition; }
    const Position& GetPosition() const { return mPosition; }
    Color GetSideToMove() const { return mPosition.GetSideToMove(); }

    void SetMetadata(const GameMetadata& metadata) { mMetadata = metadata; }

    bool operator == (const Game& rhs) const;
    bool operator != (const Game& rhs) const;

    const std::vector<Move>& GetMoves() const { return mMoves; }
    const std::vector<ScoreType>& GetMoveScores() const { return mMoveScores; }
    Score GetForcedScore() const { return mForcedScore; }

    void Reset(const Position& pos);
    void SetScore(Score score);
    bool DoMove(const Move& move);
    bool DoMove(const Move& move, ScoreType score);

    uint32_t GetRepetitionCount(const Position& position) const;

    Score GetScore() const;

    bool IsDrawn() const;

    // convert moves list to PGN string
    std::string ToPGNMoveList(bool includeScores = false) const;

    // print whole game as PNG string
    std::string ToPGN(bool includeScores = false) const;

private:

    Score CalculateScore() const;

    void RecordBoardPosition(const Position& position);

    GameMetadata mMetadata;

    Position mInitPosition;
    Position mPosition;
    Score mForcedScore;
    std::vector<Move> mMoves;
    std::vector<ScoreType> mMoveScores;

    // TODO store some simplified position state instead of full struct
    std::unordered_map<Position, uint16_t> mHistoryGamePositions;
};