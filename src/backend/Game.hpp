#pragma once

#include "Position.hpp"
#include "containers/DynArray.hpp"

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

    const DynArray<Move>& GetMoves() const { return mMoves; }
    const DynArray<ScoreType>& GetMoveScores() const { return mMoveScores; }
    Score GetForcedScore() const { return mForcedScore; }

    void Reset(const Position& pos);
    void SetScore(Score score);
    bool DoMove(const Move& move);
    bool DoMove(const Move& move, ScoreType score);

    uint32_t GetRepetitionCount(const Position& position) const;

    Score GetScore() const;

    bool IsDrawn() const;

    // convert moves list to PGN string
    String ToPGNMoveList(bool includeScores = false) const;

    // print whole game as PNG string
    String ToPGN(bool includeScores = false) const;

private:

    Score CalculateScore() const;

    void RecordBoardPosition(const Position& position);

    GameMetadata mMetadata;

    Position mInitPosition;
    Position mPosition;
    Score mForcedScore;
    DynArray<Move> mMoves;
    DynArray<ScoreType> mMoveScores;

    struct HistoryPosition
    {
        Position position;
        uint32_t count;
    };

    static constexpr uint32_t GameHistoryBuckets = 32;
    DynArray<HistoryPosition> mHistoryGamePositions[GameHistoryBuckets];
};